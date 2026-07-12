# Copyright (c) 2026. Apache-2.0.
"""APPS (codeparrot/apps) — a much larger execution-verified code task than MBPP (~10k problems,
difficulty-tiered), so far less memorization-prone and with real compute headroom for the
compute-supervision tradeoff sweep.

Reuses the execution verifier (task_name='code' -> CodeVerifierEnvironment -> code_verify.score_one,
which auto-detects the APPS io format). Ground truth = the input_output test spec (stdin/stdout OR
call-based via fn_name). Difficulty via APPS_DIFFICULTY env (comma-sep of
introductory,interview,competition; default 'interview' = the learnable middle tier).
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
_CALL_PROMPT = (
    "You are an expert Python programmer. Solve this problem.\n\n{q}\n\n"
    "Complete this solution (keep the class/signature):\n{starter}\n\n"
    "Provide the COMPLETE solution in a single ```python code block."
)
_MAX_IO_CHARS = 60_000  # skip problems whose test spec is huge (slow to verify)
_MAX_CASES = 15  # cap #test cases per problem to bound verification time


class APPSDataset(RawDataset):
    """APPS code task. Ground truth = JSON input_output (stdin/stdout or call-based); execution-verified."""

    def __init__(self, split: str = "train", system_prompt_file: str | None = None, **kwargs) -> None:
        self.task_name = "code"
        ds = load_dataset("codeparrot/apps", revision="refs/convert/parquet")[
            "test" if split in ("test", "validation") else "train"
        ]
        levels = os.environ.get("APPS_DIFFICULTY", "interview").strip()
        keep = {x.strip() for x in levels.split(",")} if levels else None
        if keep:
            ds = ds.filter(lambda x: x.get("difficulty") in keep)
        self.dataset = ds.map(self.format_data, remove_columns=ds.column_names)
        self.dataset = self.dataset.filter(
            lambda x: x["messages"][1]["content"] not in (None, "", "{}")
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        io_raw = data.get("input_output") or ""
        q = (data.get("question") or "").strip()
        starter = (data.get("starter_code") or "").strip()
        gt = "{}"
        if io_raw and len(io_raw) <= _MAX_IO_CHARS and q:
            try:
                io = json.loads(io_raw)
                ins, outs = io.get("inputs") or [], io.get("outputs") or []
                if ins and outs:
                    io_capped = {"inputs": ins[:_MAX_CASES], "outputs": outs[:_MAX_CASES]}
                    if io.get("fn_name"):
                        io_capped["fn_name"] = io["fn_name"]
                    gt = json.dumps(io_capped)
            except Exception:
                gt = "{}"
        prompt = (
            _CALL_PROMPT.format(q=q, starter=starter)
            if starter
            else _STDIO_PROMPT.format(q=q)
        )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": gt},
            ],
            "task_name": self.task_name,
        }
