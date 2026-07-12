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

import os
from typing import Any

from datasets import concatenate_datasets, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

# EleutherAI/hendrycks_math is split into 7 subject configs; concatenate for the full MATH set.
_MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _extract_boxed_answer(solution: str) -> str | None:
    """Return the content of the last \\boxed{...} in a MATH solution (brace-matched)."""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return None
    i = solution.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1 : j].strip()
    return None


class MATHDataset(RawDataset):
    """Hendrycks MATH (competition math). Unique-answer; ground truth is the boxed final answer.

    Verified with math_verify (equivalence-aware) — the near-clean baseline for controlled fp/fn.
    """

    def __init__(
        self,
        split: str = "train",
        extract_answer: bool = True,
        system_prompt_file: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "math"
        self.extract_answer = extract_answer

        parts = [load_dataset("EleutherAI/hendrycks_math", c)[split] for c in _MATH_CONFIGS]
        self.dataset = concatenate_datasets(parts)
        # Full MATH (all levels) is too hard for small base models (~1.5% for Qwen2.5-1.5B),
        # leaving no learning signal. Restrict difficulty via MATH_LEVELS (comma-sep, e.g. "1,2,3").
        levels_env = os.environ.get("MATH_LEVELS", "").strip()
        if levels_env:
            keep = {f"Level {n.strip()}" for n in levels_env.split(",")}
            self.dataset = self.dataset.filter(lambda x: x.get("level") in keep)
        self.dataset = self.dataset.map(
            self.format_data, remove_columns=self.dataset.column_names
        )
        # drop rows where no boxed answer could be extracted
        self.dataset = self.dataset.filter(
            lambda x: x["messages"][1]["content"] not in (None, "")
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.extract_answer:
            answer = _extract_boxed_answer(data["solution"])
        else:
            answer = data["solution"]

        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": answer},
            ],
            "task_name": self.task_name,
        }
