"""Extract metrics for batch-32 runs from WandB and output to CSV.

Collects: false positive rate, false negative rate, final validation accuracy,
final trained accuracy, best (max) validation and train accuracy, and num rollouts.

Usage:
    python extract_metrics.py
"""

import csv

import wandb

ENTITY = "rm4411-princeton-university"
PROJECT = "RLVR"
OUTPUT_FILE = "batch32_metrics.csv"


def extract_metrics():
    api = wandb.Api()

    path = f"{ENTITY}/{PROJECT}"
    print(f"Fetching runs from: {path}")
    runs = api.runs(path)

    # Collect run IDs first (paginator doesn't load configs)
    run_ids = [(r.name, r.id) for r in runs]
    print(f"Found {len(run_ids)} total runs")

    rows = []
    for name, rid in run_ids:
        # Filter to batch-32 runs by name
        if "-b32-" not in name and "batch32" not in name:
            continue

        # Skip runs with 'x' numeric marker (e.g. p01-x025)
        if "-x0" in name or "-x1" in name:
            continue

        # Fetch full run to get config
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
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

        rows.append({
            "run_name": name,
            "false_positive_rate": fp,
            "false_negative_rate": fn,
            "num_rollouts": num_rollouts,
            "final_validation_accuracy": val_accuracy,
            "final_train_accuracy": train_accuracy,
            "best_validation_accuracy": best_val_accuracy,
            "best_train_accuracy": best_train_accuracy,
        })

        print(f"  Collected: {name} (fp={fp}, fn={fn}, rollouts={num_rollouts}, "
              f"best_val={best_val_accuracy}, best_train={best_train_accuracy})")

    if not rows:
        print("No batch-32 runs found.")
        return

    rows.sort(key=lambda r: (r["false_positive_rate"], r["false_negative_rate"], r["num_rollouts"] or 0))

    fieldnames = [
        "run_name",
        "false_positive_rate",
        "false_negative_rate",
        "num_rollouts",
        "final_validation_accuracy",
        "final_train_accuracy",
        "best_validation_accuracy",
        "best_train_accuracy",
    ]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_metrics()
