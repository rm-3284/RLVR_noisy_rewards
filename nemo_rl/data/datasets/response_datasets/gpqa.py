# Copyright (c) 2026. Apache-2.0.
"""GPQA (Idavidrein/gpqa) as a 4-choice MCQ, verified via the existing string/equivalence
verifier: options are NUMBERED 1-4 and the model answers with \\boxed{N}, so math_verify
handles it exactly like a GSM8K integer answer (no new verifier needed).

Random-guess floor is 1/4 = 0.25 (anchor/normalize the acc(r,m) form there, not at 0).
Choice order is shuffled deterministically per-question (reproducible) to avoid position bias.
Config via GPQA_CONFIG env (gpqa_main [default,448] / gpqa_diamond / gpqa_extended).
"""
import glob
import os
import random, hashlib
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

_GPQA_DIR = "/scratch/gpfs/GRIFFITHS/aw2418/huggingface/hub/datasets--Idavidrein--gpqa"


def _csv_path(cfg: str) -> str:
    hits = glob.glob(f"{_GPQA_DIR}/snapshots/*/{cfg}.csv")
    if not hits:
        raise FileNotFoundError(f"GPQA csv for {cfg} not found under {_GPQA_DIR}")
    return hits[0]


class GPQADataset(RawDataset):
    """GPQA multiple-choice; ground truth is the (shuffled) option NUMBER of the correct answer."""

    def __init__(
        self,
        split: str = "train",
        system_prompt_file: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "math"  # reuse the string/equivalence verifier on \boxed{N}
        cfg = os.environ.get("GPQA_CONFIG", "gpqa_main").strip()
        ds = load_dataset("csv", data_files=_csv_path(cfg))["train"]
        # deterministic 80/20 train/val split by index (GPQA ships a single set)
        keep_val = split in ("test", "validation")
        ds = ds.filter(lambda x, i: (i % 5 == 0) == keep_val, with_indices=True)
        self.dataset = ds.map(self.format_data, remove_columns=ds.column_names)
        self.dataset = self.dataset.filter(
            lambda x: x["messages"][1]["content"] not in (None, "")
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        q = (data.get("Question") or "").strip()
        correct = (data.get("Correct Answer") or "").strip()
        wrong = [
            (data.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)
        ]
        if not q or not correct or any(w == "" for w in wrong):
            return {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}], "task_name": self.task_name}
        # deterministic shuffle of the 4 options, seeded by the question text
        opts = [correct] + wrong
        order = list(range(4))
        random.Random(int(hashlib.md5(q.encode()).hexdigest()[:8],16)).shuffle(order)
        shuffled = [opts[i] for i in order]
        gt_num = str(shuffled.index(correct) + 1)  # 1-based option number of the correct answer
        lines = "\n".join(f"{i + 1}) {opt}" for i, opt in enumerate(shuffled))
        user = (
            f"{q}\n\nOptions:\n{lines}\n\n"
            "Reason step by step, then give the number of the correct option as \\boxed{N}."
        )
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": gt_num},
            ],
            "task_name": self.task_name,
        }
