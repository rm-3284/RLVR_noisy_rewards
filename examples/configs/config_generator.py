import os
import itertools

batch_sizes = [16, 32, 64]
ps = [0.0, 0.1, 0.2, 0.3, 0.4]
rollouts = [4, 8, 16, 32, 64]

out_dir = "generated_configs"
os.makedirs(out_dir, exist_ok=True)

template = """defaults: grpo_math_1B.yaml

checkpointing:
  checkpoint_dir: "results/0.5Bgrpo_gsm8k_b{batch}_r{rollout}_p{p}"

grpo:
  num_prompts_per_step: {batch}
  num_generations_per_prompt: {rollout}

policy:
  train_global_batch_size: {total_batch}

data:
  _override_: true
  max_input_seq_length: ${{policy.max_total_sequence_length}}
  shuffle: true
  num_workers: 1
  use_multiple_dataloader: false

  train:
    dataset_name: "gsm8k"
    split: train

  validation:
    dataset_name: "gsm8k"
    split: test

  default:
    prompt_file: null
    system_prompt_file: "examples/prompts/gsm8k.txt"
    processor: "math_hf_data_processor"
    env_name: "math"

env:
  math:
    p: {p}

logger:
  log_dir: "logs_0.5B"
  wandb:
    name: "grpo0.5B-gsm8k-b{batch}-r{rollout}-p{p}"
"""

count = 0
for batch, p, rollout in itertools.product(batch_sizes, ps, rollouts):
    total_batch = batch * rollout
    p_str = str(p).replace(".", "")

    filename = f"grpo_gsm8k_b{batch}_r{rollout}_p{p_str}.yaml"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w") as f:
        f.write(template.format(batch=batch, rollout=rollout, p=p, total_batch=total_batch))

    count += 1

print(f"Generated {count} configs in '{out_dir}'")