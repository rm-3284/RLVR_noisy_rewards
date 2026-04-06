import os
import argparse
import itertools

DEFAULT_BATCH_SIZES = [16, 32, 64]
DEFAULT_FPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_FNS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_ROLLOUTS = [4, 8, 16, 32, 64]

template = """defaults: grpo_math_1B.yaml

checkpointing:
  checkpoint_dir: "results/{run_name}"

grpo:
  num_prompts_per_step: {batch}
  num_generations_per_prompt: {rollout}

policy:
  model_name: "{model_path}"
  train_global_batch_size: {total_batch}

data:
  _override_: true
  max_input_seq_length: ${{policy.max_total_sequence_length}}
  shuffle: true
  num_workers: 1
  use_multiple_dataloader: false

  train:
    dataset_name: "{dataset}"
    split: train

  validation:
    dataset_name: "{dataset}"
    split: test

  default:
    prompt_file: null
    system_prompt_file: "examples/prompts/{dataset}.txt"
    processor: "math_hf_data_processor"
    env_name: "math"

env:
  math:
    fp: {fp}
    fn: {fn}

logger:
  log_dir: "{log_dir}"
  wandb:
    name: "{run_name}"
"""


def parse_floats(s):
    return [float(v) for v in s.split(",")]


def parse_ints(s):
    return [int(v) for v in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Generate GRPO config files from a sweep over hyperparameters."
    )
    parser.add_argument(
        "--batch-sizes", type=parse_ints, default=DEFAULT_BATCH_SIZES,
        metavar="B1,B2,...",
        help=f"Comma-separated batch sizes (default: {DEFAULT_BATCH_SIZES})",
    )
    parser.add_argument(
        "--rollouts", type=parse_ints, default=DEFAULT_ROLLOUTS,
        metavar="R1,R2,...",
        help=f"Comma-separated rollouts per prompt (default: {DEFAULT_ROLLOUTS})",
    )
    parser.add_argument(
        "--fps", type=parse_floats, default=DEFAULT_FPS,
        metavar="FP1,FP2,...",
        help=f"Comma-separated false positive rates (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--fns", type=parse_floats, default=DEFAULT_FNS,
        metavar="FN1,FN2,...",
        help=f"Comma-separated false negative rates (default: {DEFAULT_FNS})",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="/n/fs/vision-mix/rm4411/hf_models/Qwen2.5-0.5B",
        help="Path to the HF model",
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        help="Dataset name (default: gsm8k)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="generated_configs",
        help="Output directory for generated configs (default: generated_configs)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Logger log_dir written into each config (default: logs)",
    )
    parser.add_argument(
        "--run-prefix", type=str, default="grpo",
        help="Prefix for run names and checkpoint dirs (default: grpo)",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    for batch, rollout, fp, fn in itertools.product(
        args.batch_sizes, args.rollouts, args.fps, args.fns
    ):
        total_batch = batch * rollout
        fp_str = f"{fp:.1f}".replace(".", "")
        fn_str = f"{fn:.1f}".replace(".", "")

        run_name = f"{args.run_prefix}-{args.dataset}-b{batch}-r{rollout}-fp{fp_str}-fn{fn_str}"
        filename = run_name.replace("-", "_").lstrip("_") + ".yaml"
        filepath = os.path.join(args.out_dir, filename)

        with open(filepath, "w") as f:
            f.write(
                template.format(
                    batch=batch,
                    rollout=rollout,
                    fp=fp,
                    fn=fn,
                    total_batch=total_batch,
                    model_path=args.model_path,
                    dataset=args.dataset,
                    log_dir=args.log_dir,
                    run_name=run_name,
                )
            )
        count += 1

    print(f"Generated {count} configs in '{args.out_dir}'")


if __name__ == "__main__":
    main()
