# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Code (MBPP-style) RLVR environment with a unit-test EXECUTION verifier + controlled fp/fn noise.

Mirrors math_environment (Ray actor + verify workers + EnvironmentReturn) but scores by executing
the candidate against the hidden asserts (nemo_rl.environments.code_verify.score_one) instead of
math-equivalence. Same i.i.d. fp/fn Bernoulli reward-flip so the noisy-verifier study transfers to
a DIFFERENT verifier type. ground_truth (from metadata) = JSON {"tests":[...],"setup":"..."}.
"""
import random
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.code_verify import score_one
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers


class CodeEnvConfig(TypedDict):
    num_workers: int
    fp: NotRequired[float]  # P(flip wrong→rewarded)
    fn: NotRequired[float]  # P(flip right→unrewarded)
    timeout: NotRequired[float]


class CodeEnvMetadata(TypedDict):
    ground_truth: str  # JSON {"tests":[...],"setup":"..."}


@ray.remote  # pragma: no cover
class CodeVerifyWorker:
    def __init__(self) -> None:
        # seeded so fp/fn noise is reproducible per run (matches HFVerifyWorker)
        self.rng = random.Random(42)

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs: Any,
    ) -> list[float]:
        fp = kwargs.get("fp", 0.0)
        fn = kwargs.get("fn", 0.0)
        timeout = kwargs.get("timeout", 8.0)
        results: list[float] = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            clean = int(score_one(response, ground_truth, timeout=timeout))  # 1 if all asserts pass
            # i.i.d. verifier noise (identical to math_environment.py)
            if clean == 0 and self.rng.random() < fp:
                score = 1.0  # false positive
            elif clean == 1 and self.rng.random() < fn:
                score = 0.0  # false negative
            else:
                score = float(clean)
            results.append(score)
        if return_extracted_answer:
            return results, [None] * len(results)
        return results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class CodeVerifierEnvironment(EnvironmentInterface):
    def __init__(self, cfg: CodeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            CodeVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CodeEnvMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        # extract the assistant's generated code from each conversation
        assistant_response_batch = []
        for conversation in message_log_batch:
            responses = [
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_resp = chunk_list_to_workers(assistant_response_batch, self.num_workers)
        chunked_gt = chunk_list_to_workers(ground_truths, self.num_workers)

        futures = [
            self.workers[i].verify.remote(
                chunk, gt_chunk, return_extracted_answer,
                fp=self.cfg.get("fp", 0.0),
                fn=self.cfg.get("fn", 0.0),
                timeout=self.cfg.get("timeout", 8.0),
            )
            for i, (chunk, gt_chunk) in enumerate(zip(chunked_resp, chunked_gt))
        ]
        worker_results = ray.get(futures)

        results: list[float] = []
        extracted: list | None = [] if return_extracted_answer else None
        for wr in worker_results:
            if return_extracted_answer:
                scores, answers = wr
                extracted.extend(answers)
            else:
                scores = wr
            results.extend(scores)

        observations = [
            {"role": "environment",
             "content": "Environment: correct" if r else "Environment: incorrect"}
            for r in results
        ]
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=extracted,
        )

    def global_post_process_and_metrics(self, batch):
        """Accuracy = mean reward over ended sequences (mirrors math env)."""
        rewards = batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        rewards = rewards * batch["is_end"]
        metrics = {"accuracy": rewards.mean().item()}
        return batch, metrics
