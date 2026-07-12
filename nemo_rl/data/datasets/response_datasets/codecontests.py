# Copyright (c) 2026. Apache-2.0.
"""CodeContests (deepmind/code_contests) — competitive-programming code task with ADVERSARIAL
tests. Unlike APPS (weak tests -> reward-hacking collapse), CodeContests ships up to ~100
`generated_tests` per problem, machine-generated to reject wrong-but-passing solutions, so the
execution reward is far harder to game. All problems are stdin/stdout.

Reuses the execution verifier (task_name='code' -> code_verify.run_apps stdin/stdout path, which
now runs every candidate inside the bwrap sandbox). Ground truth = {inputs, outputs} test spec.
Difficulty tier via CC_DIFFICULTY_MAX / CC_DIFFICULTY_MIN env (numeric; 0 = 'unknown', excluded by
default). Train split = train (~13k); validation split = valid.
"""
import json
import os
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

_STDIO_PROMPT = (
    "You are an expert competitive programmer. Solve this problem.\n\n{q}\n\n"
    "Read input from standard input and write the answer to standard output. "
    "Provide the COMPLETE program in a single ```python code block."
)
_MAX_CASES = 15          # cap test cases per problem (bounds verification time; generated tests are the strong ones)
_MAX_DESC_CHARS = 8000   # skip pathologically long problem statements
_MAX_IO_CHARS = 4000     # skip a test case whose input/output is huge (slow to run/compare)


class CodeContestsDataset(RawDataset):
    """CodeContests stdin/stdout code task; execution-verified against public + generated tests."""

    def __init__(self, split: str = "train", system_prompt_file: str | None = None, **kwargs) -> None:
        self.task_name = "code"
        hf_split = "train" if split in ("train",) else "valid"  # train / valid (test also exists)
        ds = load_dataset("deepmind/code_contests", split=hf_split)
        # difficulty tier: numeric; 0 = UNKNOWN (excluded by default). Tune via env for a learnable base rate.
        dmin = int(os.environ.get("CC_DIFFICULTY_MIN", "1"))
        dmax = int(os.environ.get("CC_DIFFICULTY_MAX", "8"))
        ds = ds.filter(lambda x: dmin <= int(x.get("difficulty") or 0) <= dmax)
        self.dataset = ds.map(self.format_data, remove_columns=ds.column_names)
        self.dataset = self.dataset.filter(
            lambda x: x["messages"][1]["content"] not in (None, "", "{}")
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        q = (data.get("description") or "").strip()
        gt = "{}"
        if q and len(q) <= _MAX_DESC_CHARS:
            ins: list[str] = []
            outs: list[str] = []
            # public tests first (canonical), then generated (adversarial) to fill the cap
            for bucket in ("public_tests", "generated_tests", "private_tests"):
                b = data.get(bucket) or {}
                for i, o in zip(b.get("input") or [], b.get("output") or []):
                    if len(ins) >= _MAX_CASES:
                        break
                    if i is None or o is None or len(str(i)) > _MAX_IO_CHARS or len(str(o)) > _MAX_IO_CHARS:
                        continue
                    ins.append(i)
                    outs.append(o)
            if ins and outs:
                gt = json.dumps({"inputs": ins, "outputs": outs})
        return {
            "messages": [
                {"role": "user", "content": _STDIO_PROMPT.format(q=q)},
                {"role": "assistant", "content": gt},
            ],
            "task_name": self.task_name,
        }
