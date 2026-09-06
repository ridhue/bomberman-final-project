import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "task2"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--agents",
    nargs="+",
    required=True,
    help="Agents whose Task 2 summaries should be compared",
)
args = parser.parse_args()

AGENTS = args.agents


comparison_rows = []

for agent in AGENTS:
    summary_file = RESULTS_ROOT / agent / f"{agent}_summary.csv"

    if not summary_file.exists():
        raise FileNotFoundError(
            f"No summary found for '{agent}': {summary_file}\n"
            f"Run run_task2_evaluation.py for this agent first."
        )

    metrics = {}

    with open(summary_file, "r", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            metrics[row["metric"]] = {
                "mean": float(row["mean"]),
                "standard_deviation": float(row["standard_deviation"]),
            }

    comparison_rows.append({
        "agent": agent,

        "average_score_mean":
            metrics["average_score"]["mean"],
        "average_score_sd":
            metrics["average_score"]["standard_deviation"],

        "coins_per_round_mean":
            metrics["coins_per_round"]["mean"],
        "coins_per_round_sd":
            metrics["coins_per_round"]["standard_deviation"],

        "crates_per_round_mean":
            metrics["crates_per_round"]["mean"],
        "crates_per_round_sd":
            metrics["crates_per_round"]["standard_deviation"],

        "self_kills_per_round_mean":
            metrics["self_kills_per_round"]["mean"],
        "self_kills_per_round_sd":
            metrics["self_kills_per_round"]["standard_deviation"],

        "bombs_per_round_mean":
            metrics["bombs_per_round"]["mean"],
        "bombs_per_round_sd":
            metrics["bombs_per_round"]["standard_deviation"],

        "crates_per_bomb_mean":
            metrics["crates_per_bomb"]["mean"],
        "crates_per_bomb_sd":
            metrics["crates_per_bomb"]["standard_deviation"],

        "invalid_per_round_mean":
            metrics["invalid_per_round"]["mean"],
        "invalid_per_round_sd":
            metrics["invalid_per_round"]["standard_deviation"],

        "steps_per_round_mean":
            metrics["steps_per_round"]["mean"],
        "steps_per_round_sd":
            metrics["steps_per_round"]["standard_deviation"],
    })


OUTPUT_FILE = RESULTS_ROOT / "task2_agent_comparison.csv"

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=comparison_rows[0].keys(),
        delimiter=";",
    )

    writer.writeheader()
    writer.writerows(comparison_rows)


print("\n========== TASK 2 AGENT COMPARISON ==========\n")

for result in comparison_rows:
    print(result["agent"])
    print(
        f"  score:       "
        f"{result['average_score_mean']:.2f} "
        f"± {result['average_score_sd']:.2f}"
    )
    print(
        f"  coins:       "
        f"{result['coins_per_round_mean']:.2f} "
        f"± {result['coins_per_round_sd']:.2f}"
    )
    print(
        f"  crates:      "
        f"{result['crates_per_round_mean']:.2f} "
        f"± {result['crates_per_round_sd']:.2f}"
    )
    print(
        f"  self-kills:  "
        f"{result['self_kills_per_round_mean']:.2f} "
        f"± {result['self_kills_per_round_sd']:.2f}"
    )
    print(
        f"  crates/bomb: "
        f"{result['crates_per_bomb_mean']:.2f} "
        f"± {result['crates_per_bomb_sd']:.2f}"
    )
    print()


print(f"Saved comparison to: {OUTPUT_FILE}")