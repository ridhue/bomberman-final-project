import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "task1"

AGENTS = [
    "tpl_agent",
    "peaceful_agent",
    "coin_collector_agent",
    "rule_based_agent",
]

def read_summary(agent):
    summary_file = (
        RESULTS_DIR
        / agent
        / f"{agent}_summary.csv"
    )

    metrics = {}

    with open(summary_file, "r") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            metrics[row["metric"]] = {
                "mean": float(row["mean"]),
                "sd": float(row["standard_deviation"]),
            }

    return metrics


rows = []

for agent in AGENTS:
    metrics = read_summary(agent)

    row = {
        "agent": agent,

        "coins_mean": metrics["coins_per_round"]["mean"],
        "coins_sd": metrics["coins_per_round"]["sd"],

        "completion_mean": metrics["completion_rate"]["mean"],
        "completion_sd": metrics["completion_rate"]["sd"],

        "self_kills_mean": metrics["self_kills_per_round"]["mean"],
        "self_kills_sd": metrics["self_kills_per_round"]["sd"],

        "invalid_mean": metrics["invalid_per_round"]["mean"],
        "invalid_sd": metrics["invalid_per_round"]["sd"],

        "bombs_mean": metrics["bombs_per_round"]["mean"],
        "bombs_sd": metrics["bombs_per_round"]["sd"],

        "steps_mean": metrics["steps_per_round"]["mean"],
        "steps_sd": metrics["steps_per_round"]["sd"],
    }

    rows.append(row)


output_file = RESULTS_DIR / "task1_agent_comparison.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
        delimiter=";"
    )
    writer.writeheader()
    writer.writerows(rows)


print("\n========== TASK 1 COMPARISON ==========\n")

for row in rows:
    print(row["agent"])
    print(
        f"  Coins:       "
        f"{row['coins_mean']:.2f} ± {row['coins_sd']:.2f}"
    )
    print(
        f"  Completion:  "
        f"{row['completion_mean']:.1%} ± "
        f"{row['completion_sd']:.1%}"
    )
    print(
        f"  Self-kills:  "
        f"{row['self_kills_mean']:.2f} ± "
        f"{row['self_kills_sd']:.2f}"
    )
    print(
        f"  Invalid:     "
        f"{row['invalid_mean']:.2f} ± "
        f"{row['invalid_sd']:.2f}"
    )
    print()


print(f"Saved comparison to: {output_file}")