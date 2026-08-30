import csv
import json
import statistics
import subprocess
import sys
import argparse
from pathlib import Path


ROUNDS = 100
SEEDS = [1, 2, 3, 4, 5]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Read command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent",
    required=True,
    help="Name of the agent to evaluate"
)
args = parser.parse_args()

AGENT = args.agent


# Create result folder for this agent
RESULTS_DIR = PROJECT_ROOT / "results" / "task1" / AGENT
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def per_round(stats, key):
    return stats.get(key, 0) / stats["rounds"]


results = []

for seed in SEEDS:
    print(f"\nRunning seed {seed}...")

    json_file = RESULTS_DIR / f"seed_{seed}.json"

    command = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        AGENT,
        "--scenario",
        "coin-heaven",
        "--n-rounds",
        str(ROUNDS),
        "--no-gui",
        "--seed",
        str(seed),
        "--save-stats",
        str(json_file),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    with open(json_file, "r") as file:
        data = json.load(file)

    stats = data["by_agent"][AGENT]
    
    round_stats = list(data["by_round"].values())

    completed_rounds = [
        round_data
        for round_data in round_stats
        if round_data.get("coins", 0) == 50
    ]

    completion_rate = len(completed_rounds) / len(round_stats)

    if completed_rounds:
        mean_steps_to_complete = statistics.mean(
            round_data["steps"]
            for round_data in completed_rounds
        )
    else:
        mean_steps_to_complete = None

    result = {
    "seed": seed,
    "average_score": per_round(stats, "score"),
    "coins_per_round": per_round(stats, "coins"),
    "completion_rate": completion_rate,
    "mean_steps_to_complete": mean_steps_to_complete,
    "self_kills_per_round": per_round(stats, "suicides"),
    "invalid_per_round": per_round(stats, "invalid"),
    "bombs_per_round": per_round(stats, "bombs"),
    "steps_per_round": per_round(stats, "steps"),
    }

    results.append(result)

    print(
    f"Seed {seed}: "
    f"coins={result['coins_per_round']:.2f}, "
    f"completion={result['completion_rate']:.2%}, "
    f"self-kills={result['self_kills_per_round']:.2f}"
    )


# Save results from each individual run
runs_file = RESULTS_DIR / f"{AGENT}_runs.csv"

with open(runs_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys(),
        delimiter=";"
    )
    writer.writeheader()
    writer.writerows(results)


# Calculate mean and standard deviation across the five runs
metric_names = [
    "average_score",
    "coins_per_round",
    "completion_rate",
    "self_kills_per_round",
    "invalid_per_round",
    "bombs_per_round",
    "steps_per_round",
]

summary_rows = []

print("\n========== MULTI-RUN SUMMARY ==========\n")

for metric in metric_names:
    values = [result[metric] for result in results]

    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values)

    summary_rows.append({
        "metric": metric,
        "mean": mean_value,
        "standard_deviation": std_value,
    })

    print(
        f"{metric:25} "
        f"Mean: {mean_value:.2f}   "
        f"SD: {std_value:.2f}"
    )


summary_file = RESULTS_DIR / f"{AGENT}_summary.csv"

with open(summary_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["metric", "mean", "standard_deviation"],
        delimiter=";"
    )
    writer.writeheader()
    writer.writerows(summary_rows)


print(f"\nSaved individual runs to: {runs_file}")
print(f"Saved aggregate summary to: {summary_file}")