# CAIS: Qwen full-epoch redo — 0.5B / 1.5B / 3B MATH — 2026-08-08

**Standard: FULL-EPOCH completion.** A cell is done only if a run reached **step 234** (1 MATH epoch =
dataset ÷ batch-32; fixed by config, same for every size and rollout). Anything below 234 — timeout,
cut-short, MIG-fail, early-crash — is a redo. No peak bar, no slack. Truth = W&B `_step`, dedup max.

## Sweeps (all on CAIS; pull checkout-only, do NOT merge the branch)
```
git fetch origin
git checkout origin/backup/analysis-sweeps-infra-2026-07-11 -- \
  sweeps/cais_v4q3_remainder.txt sweeps/cais_v4q15_remainder.txt sweeps/cais_v4q05_remainder.txt PUSH_CAIS_QWEN_REDO.md
```
| sweep | model | cells |
|---|---|---|
| `sweeps/cais_v4q3_remainder.txt`  | `MODELS/Qwen2.5-3B`   | **150** |
| `sweeps/cais_v4q15_remainder.txt` | `MODELS/Qwen2.5-1.5B` | **56**  |
| `sweeps/cais_v4q05_remainder.txt` | `MODELS/Qwen2.5-0.5B` | **81**  |

- **Model staging:** Qwen2.5-3B is staged on CAIS. **Confirm `MODELS/Qwen2.5-1.5B` and `MODELS/Qwen2.5-0.5B`
  exist** (`hf download Qwen/Qwen2.5-1.5B` / `-0.5B` if not).
- Launch like the earlier CAIS MATH sweeps (MATH override, `BASE_CONFIG=grpo_gsm8k_1B_rollout32_batch32.yaml`).
- **WALLTIME — the one risk:** r128 must reach step 234. **3B r128 is currently 0/80 at full-epoch** (all
  timed out ~213 on Della's 48h). Falcon-3B r128 completed on CAIS, so CAIS should get there — but confirm
  the r128 step-rate projects to 234 under CAIS's 48h cap; if not, r128 needs a longer-walltime QOS.
- Re-audit after: `_step >= 234` for every cell, or it's not done.
