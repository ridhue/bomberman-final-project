import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--opponent",
    required=True,
    help="Opponent used in the Stage 3 evaluation",
)

parser.add_argument(
    "--agents",
    nargs="+",
    required=True,
    help="Agents to compare",
)

args = parser.parse_args()

OPPONENT = args.opponent
AGENTS = args.agents

RESULTS_ROOT = PROJECT_ROOT / "results" / "task3" / OPPONENT


# --------------------------------------------------------------
# Read summaries
# --------------------------------------------------------------

comparison_rows = []

for agent in AGENTS:
    summary_file = RESULTS_ROOT / agent / f"{agent}_summary.csv"

    if not summary_file.exists():
        print(f"WARNING: summary not found for {agent}:")
        print(summary_file)
        continue

    with open(summary_file, newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            comparison_rows.append({
                "agent": agent,
                "metric": row["metric"],
                "mean": float(row["mean"]),
                "standard_deviation": float(row["standard_deviation"]),
            })


if not comparison_rows:
    raise RuntimeError("No valid Stage 3 summary files were found.")


# --------------------------------------------------------------
# Save combined comparison
# --------------------------------------------------------------

output_file = RESULTS_ROOT / "task3_agent_comparison.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "agent",
            "metric",
            "mean",
            "standard_deviation",
        ],
        delimiter=";",
    )

    writer.writeheader()
    writer.writerows(comparison_rows)


# --------------------------------------------------------------
# Print comparison
# --------------------------------------------------------------

print(f"\n========== TASK 3 COMPARISON VS {OPPONENT} ==========\n")

important_metrics = [
    "kills_per_round",
    "average_score",
    "self_kills_per_round",
    "steps_per_round",
]

for metric in important_metrics:
    print(metric)

    for row in comparison_rows:
        if row["metric"] == metric:
            print(
                f"  {row['agent']:25} "
                f"{row['mean']:.2f} ± {row['standard_deviation']:.2f}"
            )

    print()


print(f"Saved comparison to: {output_file}")