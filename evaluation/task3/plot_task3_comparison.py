import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--opponent",
    required=True,
    help="Opponent used in the Stage 3 evaluation",
)

args = parser.parse_args()

OPPONENT = args.opponent

RESULTS_DIR = PROJECT_ROOT / "results" / "task3" / OPPONENT
COMPARISON_FILE = RESULTS_DIR / "task3_agent_comparison.csv"


if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"Comparison file not found: {COMPARISON_FILE}\n"
        "Run compare_task3_agents.py first."
    )


# --------------------------------------------------------------
# Read comparison CSV
# --------------------------------------------------------------

rows = []

with open(COMPARISON_FILE, newline="") as file:
    reader = csv.DictReader(file, delimiter=";")

    for row in reader:
        rows.append({
            "agent": row["agent"],
            "metric": row["metric"],
            "mean": float(row["mean"]),
            "standard_deviation": float(row["standard_deviation"]),
        })


# --------------------------------------------------------------
# Metrics selected for Stage 3
# --------------------------------------------------------------

plots = {
    "kills_per_round": "Kills per Round",
    "average_score": "Average Score per Round",
    "self_kills_per_round": "Self-Kills per Round",
    "steps_per_round": "Steps per Round",
}


for metric, ylabel in plots.items():

    metric_rows = [
        row for row in rows
        if row["metric"] == metric
    ]

    if not metric_rows:
        print(f"Skipping missing metric: {metric}")
        continue

    agents = [row["agent"] for row in metric_rows]
    means = [row["mean"] for row in metric_rows]
    stds = [row["standard_deviation"] for row in metric_rows]

    plt.figure(figsize=(8, 5))

    plt.bar(
        agents,
        means,
        yerr=stds,
        capsize=5,
    )

    plt.ylabel(ylabel)
    plt.xlabel("Agent")
    plt.title(f"Stage 3 vs {OPPONENT}: {ylabel}")

    plt.tight_layout()

    output_file = RESULTS_DIR / f"task3_{metric}.png"

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(f"Saved plot: {output_file}")