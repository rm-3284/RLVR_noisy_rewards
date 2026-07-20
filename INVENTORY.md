# RLVR Noisy-Rewards — USABLE GRID INVENTORY (all tasks, incl. active v4 campaign)
_cells = distinct (fp,fn); off-diag (fp≠fn) needed for asymmetry + strict collapse; ≥2 rollouts needed for β1._
_NOTE: counts are attempted grid cells on disk; not all cells are fully converged/usable — run the fit scripts to confirm._

## GSM8K

| model | rollouts | cells@each | max off-diag | β1 | collapse |
|---|---|---|---|---|---|
| OLMo-1B | [8, 32, 128] | r8:16, r32:16, r128:16 | 12 | YES | YES |
| Qwen-0.5B | [8, 32, 128] | r8:2, r32:6, r128:2 | 2 | YES | weak |
| Qwen-1.5B | [4, 16, 32, 64, 128, 256, 512] | r4:2, r16:2, r32:7, r64:2, r128:2, r256:2, r512:2 | 4 | YES | YES |
| Qwen-3B | [8, 32, 128] | r8:3, r32:5, r128:3 | 2 | YES | weak |
| Qwen-7B | [32] | r32:5 | 2 | no | weak |

## MATH

| model | rollouts | cells@each | max off-diag | β1 | collapse |
|---|---|---|---|---|---|
| Gemma-2B | [8] | r8:3 | 0 | no | no |
| Llama-1B | [32] | r32:3 | 2 | no | weak |
| OLMo-1B | [8, 32, 128] | r8:16, r32:16, r128:16 | 12 | YES | YES |
| OLMo-7B | [8, 128] | r8:1, r128:5 | 4 | YES | YES |
| Qwen-0.5B | [8, 32, 64, 128] | r8:1, r32:6, r64:3, r128:16 | 12 | YES | YES |
| Qwen-1.5B | [8, 32, 128] | r8:2, r32:5, r128:16 | 12 | YES | YES |
| Qwen-3B | [8, 32] | r8:3, r32:3 | 2 | YES | weak |

## CODE

| model | rollouts | cells@each | max off-diag | β1 | collapse |
|---|---|---|---|---|---|
| Qwen-0.5B | [8, 16] | r8:1, r16:3 | 2 | YES | weak |
| Qwen-1.5B | [16, 32] | r16:5, r32:1 | 2 | YES | weak |
| Qwen-3B | [8] | r8:1 | 0 | no | no |

## GPQA

| model | rollouts | cells@each | max off-diag | β1 | collapse |
|---|---|---|---|---|---|
| Qwen-1.5B | [32] | r32:1 | 0 | no | no |
