"""Extract metrics for batch-32 and Gemma rollout runs from WandB and output to CSV.

Collects: false positive rate, false negative rate, final validation accuracy,
final trained accuracy, best (max) validation and train accuracy, and num rollouts.

Writes:
  batch32_metrics.csv  — Qwen batch-32 runs (grpo-gsm8k-b32-* / grpo0.5B-*)
  gemma_metrics.csv    — Gemma runs (grpo1b-* / grpo270m-*)

Usage:
    python extract_metrics.py
"""

import csv

import wandb

ENTITY = "rm4411-princeton-university"
PROJECT = "RLVR"
OUTPUT_FILE = "batch32_metrics.csv"
GEMMA_OUTPUT_FILE = "gemma_metrics.csv"

FIELDNAMES = [
    "run_name",
    "false_positive_rate",
    "false_negative_rate",
    "num_rollouts",
    "final_validation_accuracy",
    "final_train_accuracy",
    "best_validation_accuracy",
    "best_train_accuracy",
]


def _collect_run(api, entity, project, name, rid):
    """Fetch a single run and return a metrics dict, or None on skip."""
    run = api.run(f"{entity}/{project}/{rid}")
    config = run.config

    env_math = config.get("env", {}).get("math", {})
    if "fp" in env_math or "fn" in env_math:
        fp = env_math.get("fp", 0.0)
        fn = env_math.get("fn", 0.0)
    else:
        # Older runs use p (overall error rate) with fp=fn=p
        p = env_math.get("p", 0.0)
        fp = p
        fn = p
    num_rollouts = config.get("grpo", {}).get("num_generations_per_prompt")

    val_accuracy = run.summary.get("validation/accuracy")
    train_accuracy = run.summary.get("train/reward")

    best_val_accuracy = None
    best_train_accuracy = None
    for row in run.scan_history(keys=["validation/accuracy"]):
        v = row.get("validation/accuracy")
        if v is not None and (best_val_accuracy is None or v > best_val_accuracy):
            best_val_accuracy = v
    for row in run.scan_history(keys=["train/reward"]):
        v = row.get("train/reward")
        if v is not None and (best_train_accuracy is None or v > best_train_accuracy):
            best_train_accuracy = v

    return {
        "run_name": name,
        "false_positive_rate": fp,
        "false_negative_rate": fn,
        "num_rollouts": num_rollouts,
        "final_validation_accuracy": val_accuracy,
        "final_train_accuracy": train_accuracy,
        "best_validation_accuracy": best_val_accuracy,
        "best_train_accuracy": best_train_accuracy,
    }


def _write_csv(rows, output_file):
    rows.sort(key=lambda r: (r["false_positive_rate"], r["false_negative_rate"], r["num_rollouts"] or 0))
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_file}")


def extract_metrics():
    api = wandb.Api()

    path = f"{ENTITY}/{PROJECT}"
    print(f"Fetching runs from: {path}")
    runs = api.runs(path)

    # Collect run IDs first (paginator doesn't load configs)
    run_ids = [(r.name, r.id) for r in runs]
    print(f"Found {len(run_ids)} total runs")

    batch32_rows = []
    gemma_rows = []

    for name, rid in run_ids:
        is_gemma = name.startswith("grpo1b-") or name.startswith("grpo270m-")
        is_batch32 = ("-b32-" in name or "batch32" in name) and not is_gemma

        if not is_gemma and not is_batch32:
            continue

        # Skip runs with 'x' numeric marker (e.g. p01-x025)
        if "-x0" in name or "-x1" in name:
            continue

        row = _collect_run(api, ENTITY, PROJECT, name, rid)
        print(f"  Collected: {name} (fp={row['false_positive_rate']}, fn={row['false_negative_rate']}, "
              f"rollouts={row['num_rollouts']}, best_val={row['best_validation_accuracy']}, "
              f"best_train={row['best_train_accuracy']})")

        if is_gemma:
            gemma_rows.append(row)
        else:
            batch32_rows.append(row)

    if batch32_rows:
        _write_csv(batch32_rows, OUTPUT_FILE)
    else:
        print("No batch-32 runs found.")

    if gemma_rows:
        _write_csv(gemma_rows, GEMMA_OUTPUT_FILE)
    else:
        print("No Gemma runs found.")


if __name__ == "__main__":
    extract_metrics()
