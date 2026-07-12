# Della AI-Lab (H200) Cluster — Practical Usage Guide

A self-contained how-to for running GPU jobs on Della's **AI-Lab partition**. Verified against the live
cluster 2026-07-08. For general Della docs see `della_guide.md`; this file is the AI-Lab-specific slice.

---

## 1. What the AI-Lab partition is

- **144 H200 GPUs** = **18 nodes × 8 H200 each** (node names `della-i19g1`, `della-i20g2`, …).
- Per **GPU**: 141 GB VRAM, FP8 + bf16 support.
- Per **node**: 64 Intel (Emerald Rapids) CPU-cores, **1.5 TB** CPU RAM, 8 H200.
- Exclusive to **approved AI-Lab members**. Access is via a project **account** (here: `griffith`).

Use it for: large models needing multi-GPU (FSDP/TP) — 7B/9B+ — or many mid-size single-GPU jobs
that want H200 memory headroom. Smaller models (≤3B) can also run here single-GPU.

---

## 2. Access pattern (the two directives that matter)

```bash
#SBATCH --account=griffith        # your approved AI-Lab project account
#SBATCH --partition=ailab         # REQUIRED to land on H200 nodes
```

**Gotcha:** on Della's *general* `gpu` partition you must NOT pass `--partition` (the submit plugin
rejects explicit `gpu`, and auto-routes by QOS). The AI-Lab partition is the opposite — you **DO**
pass `--partition=ailab`. Don't copy the "no --partition" rule from general-gpu scripts to AI-Lab.

---

## 3. QOS is auto-assigned by `--time` — do NOT hardcode `--qos`

| QOS         | set it by `--time` ≤ | max wall | concurrent GPUs/user | max submitted tasks/user | max jobs/user |
|-------------|----------------------|----------|----------------------|--------------------------|---------------|
| gpu-short   | `1-00:00:00` (24h)   | 24h      | 44                   | 1100                     | 44            |
| gpu-medium  | `3-00:00:00` (72h)   | 3 days   | 20                   | 1100                     | 24            |
| gpu-long    | `6-00:00:00` (144h)  | 6 days   | 16                   | 100                      | 10            |

- Pick `--time` to fall in the band you want; the scheduler assigns the QOS. Setting `--qos` by hand
  can conflict and get rejected.
- **"Concurrent GPUs/user"** is the real throughput ceiling (e.g. 20 on gpu-medium): you can have at
  most that many GPUs running at once in that QOS, no matter how many tasks you submit.
- AI-Lab jobs typically land in **gpu-medium** (48–72h is plenty for training runs).

---

## 4. Job templates (copy-paste, verified working)

### 4a. Single-GPU job
```bash
sbatch --job-name=myrun --account=griffith --partition=ailab \
  --time=2-00:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=180G \
  myscript.sh
```

### 4b. Multi-GPU (4-GPU node-share, for FSDP / tensor-parallel of 7B–9B+)
```bash
sbatch --job-name=myrun --account=griffith --partition=ailab \
  --time=2-00:00:00 --gres=gpu:4 --cpus-per-task=32 --mem=360G \
  myscript.sh
```
- Request `--gres=gpu:N` for N∈{1..8} (a node has 8 H200). For a *whole* node use `--gres=gpu:8`.
- Scale `--cpus-per-task` (~8/GPU) and `--mem` (≤1.5 TB/node) with the GPU count.
- For a training framework, pass the parallelism to match: e.g. 4-GPU tensor-parallel vLLM +
  FSDP policy → `tensor_parallel_size=4`, `gpus_per_node=4`.

### 4c. Array sweep (many cells)
```bash
sbatch --job-name=sweep --account=griffith --partition=ailab \
  --time=2-00:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=180G \
  --array=0-239%3 \
  --export=ALL,PARAMS_FILE=params.txt,EXTRA=... myscript.sh
# %3 = at most 3 array tasks running at once (concurrency throttle)
```

---

## 5. The submit-cap gotcha (READ THIS before big sweeps)

`MaxSubmitPU` counts **every pending array element**, not just running ones. The `%N` concurrency
throttle limits how many *run*, but the rest sit **PENDING and still count** toward the 1100 cap.

- A `--array=0-899%2` (900 cells) consumes **~900** of your 1100 submit budget the instant it's queued,
  even though only 2 run at a time.
- Consequence: you can queue at most ~1 large (≈900-cell) sweep per QOS at once. Feed additional
  sweeps as room frees, or split across QOS (gpu-short vs gpu-medium have independent 1100 budgets).
- Symptom when full: `sbatch: error: QOSMaxSubmitJobPerUserLimit`.
- Check your usage (expanded): `squeue -u $USER -h -r -t all -o '%q' | sort | uniq -c`

---

## 6. Air-gapped compute nodes (no internet on the GPU nodes)

Compute nodes cannot reach the network. Do network work on the **login node**, run offline on the node:

```bash
# inside the job script:
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
```
- **Pre-download** models + datasets on the login node (which has internet) before submitting.
  HF-gated models (Llama, Gemma) need a token accepted once: `huggingface-cli download <repo> --local-dir <path>`.
- **W&B**: runs log offline into the job dir; sync from the login node afterward with `wandb sync <offline-run-dir>`.
- Point caches at scratch (persist across nodes): `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`
  under `/scratch/gpfs/<PROJECT>/<user>`.

---

## 7. Monitoring & control

```bash
squeue -u $USER                              # your jobs (compact; arrays collapse)
squeue -u $USER -r                           # expanded (one line per array task)
sinfo -p ailab -o "%P %D %t %G"              # partition capacity/state
sinfo -p ailab -N -o "%N %G %t"              # per-node GPU state (idle/mix/alloc)
scancel <jobid>                              # cancel one job/array
scancel <jobid>_[10-899]                     # cancel a pending array tail (keep running cells)
scancel --state=PENDING -n <name> -u $USER   # cancel only pending elements of an array
```
- No fully-idle nodes ≠ broken: H200 nodes are usually `mix` (shared). Jobs queue on `Priority`/`Resources`
  until GPUs free; freeing your own held jobs is the main lever to speed your own queue.

---

## 8. Quick gotcha checklist

- ✅ `--account=griffith --partition=ailab` (both needed).
- ✅ Choose `--time` to pick the QOS band; never hardcode `--qos`.
- ✅ Big array? Remember pending cells eat the 1100 submit cap — feed in chunks.
- ✅ Everything offline on the node; download + wandb-sync on the login node.
- ✅ Mem/CPU scale with GPU count; a node caps at 8 GPU / 64 CPU / 1.5 TB.
- ⚠️ `find -newermt` / `sacct -S` use **local time** — don't query a future instant by mistake.
- ⚠️ Mass `scancel -u $USER` may be gated by tooling; run it yourself in an interactive shell if needed.
