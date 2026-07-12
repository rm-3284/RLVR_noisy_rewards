# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MBPP (Mostly Basic Python Problems) — code task with a unit-test (execution) verifier.

Different verifier TYPE from math (string/equivalence): reward = candidate code passes the
hidden asserts. The "ground truth" carried in the assistant message is the JSON-encoded test
list, which CodeVerifierEnvironment executes the candidate against (+ i.i.d. fp/fn noise).
Weak models on MBPP give a LOW base rate → the low-precision FP-worse regime, off math.
"""

import json
import os
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

# MBPP prompt: problem text + the asserts (gives the expected function name/signature).
_PROMPT = (
    "You are an expert Python programmer. Write a Python function to solve this task.\n"
    "Task: {text}\n"
    "Your function must pass these tests:\n{tests}\n"
    "Return ONLY the function definition in a ```python code block."
)


class MBPPDataset(RawDataset):
    """MBPP code task. Ground truth = JSON-encoded test_list (asserts); verified by execution."""

    def __init__(
        self,
        split: str = "train",
        config: str = "full",
        system_prompt_file: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "code"
        ds = load_dataset("google-research-datasets/mbpp", config)[split]
        # Optional difficulty/size control via env (kept simple: cap #problems if set).
        cap = os.environ.get("MBPP_MAX", "").strip()
        if cap:
            ds = ds.select(range(min(int(cap), len(ds))))
        self.dataset = ds.map(self.format_data, remove_columns=ds.column_names)
        # drop rows with no usable tests
        self.dataset = self.dataset.filter(
            lambda x: x["messages"][1]["content"] not in (None, "", "[]")
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # 'full' config uses 'text'; 'sanitized' uses 'prompt'.
        problem = data.get("text") or data.get("prompt") or ""
        tests = list(data.get("test_list") or [])
        setup = data.get("test_setup_code") or ""
        # ground truth passed to the verifier = JSON of {tests, setup}
        gt = json.dumps({"tests": tests, "setup": setup})
        prompt = _PROMPT.format(text=problem.strip(), tests="\n".join(tests))
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": gt},
            ],
            "task_name": self.task_name,
        }
