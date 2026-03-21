# GRPO post-training with NeMo RL

This repository is based on [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL): a Ray-based library for large-scale LLM post-training, including **GRPO** (Group Relative Policy Optimization). Use it when you want many comparable GRPO runs (different models, datasets, hyperparameters, or custom reward environments) with a consistent workflow.

For the upstream overview and full feature list, see [`README_legacy.md`](README_legacy.md). This document focuses on **practical usage**: environment setup, swapping models, defining new RL environments, and organizing many training jobs.

---

## 1. One-time setup

### Clone and submodules

```sh
git clone <your-fork-url> RLVR_noisy_rewards --recursive
cd RLVR_noisy_rewards
```

If you already cloned without submodules:

```sh
git submodule update --init --recursive
```

### Python environment with `uv`

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```sh
uv venv
```

Do **not** pass `-p` / `--python`; the repo pins the version in `.python-version`.

Run everything through `uv run` (avoid only activating the venv) so dependencies stay aligned with the lockfile:

```sh
uv run python examples/run_grpo.py
```

After dependency changes in `pyproject.toml`, force a clean venv on the next run:

```sh
NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_grpo.py
```

### Environment variables

Typical settings:

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Hugging Face cache (models and datasets) |
| `HF_DATASETS_CACHE` | Optional separate dataset cache |
| `WANDB_API_KEY` | Weights & Biases (if `logger.wandb_enabled=true`; cluster networking: [below](#weights--biases-on-clusters)) |

Gated models (e.g. Llama) require `huggingface-cli login`.

### GPU / CUDA

You need NVIDIA drivers and a CUDA stack compatible with the pinned PyTorch build. Megatron training on bare metal may need extra system packages (e.g. cuDNN); see [`README_legacy.md`](README_legacy.md#prerequisites).

---

## 2. Minimal GRPO run

Entry point: [`examples/run_grpo.py`](examples/run_grpo.py).

Default config: [`examples/configs/grpo_math_1B.yaml`](examples/configs/grpo_math_1B.yaml) (math GRPO on `Qwen/Qwen2.5-1.5B` with the DTensor + vLLM path).

```sh
uv run python examples/run_grpo.py
```

Use another YAML as a base:

```sh
uv run python examples/run_grpo.py --config examples/configs/grpo_math_8B.yaml
```

Configs use [OmegaConf](https://omegaconf.readthedocs.io/) style overrides after `--config`:

```sh
uv run python examples/run_grpo.py \
  cluster.gpus_per_node=8 \
  grpo.num_prompts_per_step=64
```

---

## 3. Using a new model

### Hugging Face ID (DTensor path, default in `grpo_math_1B.yaml`)

Set the policy and (usually) the tokenizer:

```sh
uv run python examples/run_grpo.py \
  policy.model_name="meta-llama/Llama-3.2-3B-Instruct" \
  policy.tokenizer.name="meta-llama/Llama-3.2-3B-Instruct"
```

Also adjust **memory-related** fields so training and rollout fit on your hardware:

- `policy.max_total_sequence_length` — upper bound on packed length; keep consistent with generation limits.
- `policy.train_global_batch_size` / `policy.train_micro_batch_size` — throughput vs. memory.
- `policy.generation.vllm_cfg.tensor_parallel_size` — vLLM TP for large models.
- `policy.generation.vllm_cfg.max_model_len` — often tied to `${policy.max_total_sequence_length}` in recipes.
- `policy.dtensor_cfg.tensor_parallel_size`, `policy.dtensor_cfg.sequence_parallel`, `policy.dtensor_cfg.activation_checkpointing` — training-side scaling.

Optional: `policy.hf_config_overrides: {}` for small HF config tweaks without forking the model code.

### Megatron backend

Use a Megatron-oriented recipe (e.g. [`examples/configs/grpo_math_1B_megatron.yaml`](examples/configs/grpo_math_1B_megatron.yaml)) and set `policy.megatron_cfg.converter_type` to match your architecture (see existing recipes under `examples/configs/recipes/`).

### Validate a new model

GRPO assumes rollout (e.g. vLLM) and training log-probs stay aligned. Before long sweeps, follow [`docs/adding-new-models.md`](docs/adding-new-models.md) and the diagnostic scripts under `tools/model_diagnostics/`.

---

## 4. Data and tasks

GRPO expects chat-formatted examples with a per-sample `task_name`. Built-in dataset classes live in [`nemo_rl/data/datasets/response_datasets/`](nemo_rl/data/datasets/response_datasets/). See [`docs/guides/grpo.md`](docs/guides/grpo.md) for the full data model, processors, and multi-dataset setups.

### Quick patterns

- **HF dataset by name** — e.g. `dataset_name: OpenMathInstruct-2` under `data.train` (as in `grpo_math_1B.yaml`).
- **JSONL / local or HF path** — use `ResponseDataset` with `data_path`, `input_key`, `output_key`; see the `data:` example in [`docs/guides/grpo.md`](docs/guides/grpo.md#dataset).
- **Multiple training sources** — [`examples/configs/grpo_multiple_datasets.yaml`](examples/configs/grpo_multiple_datasets.yaml) and the “Multiple dataloaders” section in the same guide.

Each dataset entry picks up defaults from `data.default` (prompt files, `processor`, `env_name`, etc.).

---

## 5. RL environments: config wiring

Environments compute **rewards** (and optional multi-turn state) from the model’s messages. They run as **Ray actors**.

### Built-in environment names

Registered names are defined in [`nemo_rl/environments/utils.py`](nemo_rl/environments/utils.py) (`ENV_REGISTRY`), including for example:

- `math` / `math_default` — math verification (e.g. [`MathEnvironment`](nemo_rl/environments/math_environment.py))
- `math_multi_reward` — multi-component rewards (e.g. GDPO-style)
- `code`, `code_jaccard` — code-style tasks
- `reward_model` — reward-model-based scoring
- `vlm` — vision-language tasks
- `nemo_gym` — NeMo-Gym integration

### YAML wiring

1. **`data.default.env_name`** (or per-dataset `env_name`) must match a key in the top-level **`env:`** block.
2. That key’s value is the **constructor config** passed to the environment actor.

Example from [`examples/configs/grpo_math_1B.yaml`](examples/configs/grpo_math_1B.yaml):

```yaml
data:
  default:
    processor: "math_hf_data_processor"
    env_name: "math"

env:
  math:
    num_workers: 8
    math_verify_impl: "hf_math_verify"   # or "dapo_math_verify"
```

For a different registered environment, change `env_name` and add a sibling block under `env:` with the parameters your actor expects.

---

## 6. Creating a new environment

### Step 1 — Implement the interface

Subclasses implement [`EnvironmentInterface`](nemo_rl/environments/interfaces.py): mainly `step(...)` → [`EnvironmentReturn`](nemo_rl/environments/interfaces.py) (observations, rewards, `terminateds`, etc.) and `global_post_process_and_metrics(...)`.

Follow the patterns in:

- [`nemo_rl/environments/math_environment.py`](nemo_rl/environments/math_environment.py) (single-turn verification)
- [`nemo_rl/environments/code_environment.py`](nemo_rl/environments/code_environment.py) (tooling / feedback-style loops)

The class must be a **`@ray.remote`** actor, like the existing environments.

### Step 2 — Register the name

Either:

- **Fork-local:** add an entry to `ENV_REGISTRY` in [`nemo_rl/environments/utils.py`](nemo_rl/environments/utils.py), or  
- **Runtime:** call [`register_env`](nemo_rl/environments/utils.py) **before** `setup_response_data` runs — e.g. import your module from a small wrapper script that mirrors [`examples/run_grpo.py`](examples/run_grpo.py) and register first.

### Step 3 — Data processor (if needed)

Map dataset rows into a [`DatumSpec`](nemo_rl/data/interfaces.py) (message log, `extra_env_info` for ground truth, etc.). Math uses processors in [`nemo_rl/data/processors.py`](nemo_rl/data/processors.py); add your own and reference it from `data.default.processor` (string import path).

### Step 4 — Point the YAML at it

Set `data.default.env_name` (or per-dataset) to your registered name and add matching `env.<your_name>: { ... }` with any kwargs your actor’s `__init__` expects.

### Optional: isolated Python for the actor

Some stacks use a separate uv environment per actor class (see [`docs/design-docs/dependency-management.md`](docs/design-docs/dependency-management.md)). For standard GRPO math/code this is usually unnecessary.

---

## 7. Running many GRPO experiments

### Separate outputs per run

[`examples/run_grpo.py`](examples/run_grpo.py) calls `get_next_experiment_dir` on `logger.log_dir`, creating numbered subdirectories `exp_001`, `exp_002`, … under that base. Point `logger.log_dir` at a study-specific folder so runs do not clash.

**Checkpoints** do not auto-increment: set a **unique** `checkpointing.checkpoint_dir` per experiment when you run in parallel or want clean restarts:

```sh
uv run python examples/run_grpo.py \
  checkpointing.checkpoint_dir="results/grpo/qwen15b_run42" \
  logger.log_dir="logs/grpo/qwen15b_run42" \
  logger.wandb_enabled=true \
  logger.wandb.name="qwen15b-run42"
```

### Resuming training

If `checkpointing.enabled` is true and `checkpointing.checkpoint_dir` points at a directory that already contains `step_*` checkpoints, [`CheckpointManager`](nemo_rl/utils/checkpoint.py) loads the latest state (policy, dataloader position, etc.). Keep the **same** config family as the original run when resuming.

### Recipe YAMLs

Under [`examples/configs/recipes/`](examples/configs/recipes/) there are large-scale, multi-node-oriented GRPO recipes. Copy one close to your hardware and edit model paths and cluster sections rather than starting from scratch.

### Sweeps

For many jobs, generate config files or shell loops that only change overrides (model, seeds, reward noise, batch sizes). Slurm/Kubernetes patterns are described in [`docs/cluster.md`](docs/cluster.md).

### Weights & Biases on clusters

With `logger.wandb_enabled=true`, the **training driver** (the process that runs `wandb.init`, typically the node where you launch `uv run … run_grpo.py` in a Ray job) must reach the W&B API over **HTTPS**—by default the public service (`api.wandb.ai`), unless you use a [self-hosted](https://docs.wandb.ai/guides/hosting/) deployment via `WANDB_BASE_URL`. You do **not** usually need every GPU worker node to have its own outbound path to W&B; only the process that owns the run does.

- **Direct internet from compute** — If your compute nodes can open HTTPS to W&B, set `WANDB_API_KEY` and run as usual.
- **Egress only through an HTTP(S) proxy** — Configure `https_proxy` / `HTTPS_PROXY` (and `http_proxy` if required) so Python’s `wandb` client can reach the API; follow your site’s proxy policy and [W&B’s networking docs](https://docs.wandb.ai/guides/track/environment-variables) if you need extra settings.
- **No internet on compute nodes** — Use **`WANDB_MODE=offline`** so metrics are written under your `logger.log_dir`; later run **`wandb sync`** from a machine that can reach W&B (e.g. a login node), or prefer **`logger.tensorboard_enabled=true`** / MLflow to a path on shared storage and inspect or upload from elsewhere.

The same networking considerations apply to **downloading models or datasets** from Hugging Face on the nodes that perform those downloads; plan cache directories (`HF_HOME`, etc.) and prefetch if compute has no WAN access.

---

## 8. Evaluation and HF export

After training, you may convert DTensor checkpoints to Hugging Face format and run eval; see the “Evaluation” section in [`README_legacy.md`](README_legacy.md#evaluation) and [`docs/guides/eval.md`](docs/guides/eval.md).

---

## 9. Further reading

| Topic | Document |
|--------|-----------|
| GRPO internals, loss, datasets | [`docs/guides/grpo.md`](docs/guides/grpo.md) |
| New models & logprob checks | [`docs/adding-new-models.md`](docs/adding-new-models.md) |
| Training / generation backends | [`docs/design-docs/training-backends.md`](docs/design-docs/training-backends.md), [`docs/design-docs/generation.md`](docs/design-docs/generation.md) |
| Clusters | [`docs/cluster.md`](docs/cluster.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

---

## License

NeMo RL is licensed under the Apache License 2.0; see the `LICENSE` file in the repository.
