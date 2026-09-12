import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROUNDS = 100
SEEDS = [1, 2, 3, 4, 5]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------
# Command-line arguments
# --------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--agent",
    required=True,
    help="Agent to evaluate",
)

parser.add_argument(
    "--opponent",
    required=True,
    help="Opponent used for the Stage 3 hunting evaluation",
)

args = parser.parse_args()

AGENT = args.agent
OPPONENT = args.opponent


# --------------------------------------------------------------
# Result directory
# --------------------------------------------------------------

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task3"
    / OPPONENT
    / AGENT
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def per_round(stats, key):
    return stats.get(key, 0) / stats["rounds"]


# --------------------------------------------------------------
# Run evaluation
# --------------------------------------------------------------

results = []

for seed in SEEDS:
    print(f"\nRunning seed {seed} against {OPPONENT}...")

    json_file = RESULTS_DIR / f"seed_{seed}.json"

    command = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        AGENT,
        OPPONENT,
        "--scenario",
        "classic",
        "--n-rounds",
        str(ROUNDS),
        "--no-gui",
        "--seed",
        str(seed),
        "--save-stats",
        str(json_file),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    with open(json_file, "r") as file:
        data = json.load(file)

    stats = data["by_agent"][AGENT]

    bombs = stats.get("bombs", 0)
    kills = stats.get("kills", 0)

    if bombs > 0:
        kills_per_bomb = kills / bombs
    else:
        kills_per_bomb = 0.0

    result = {
        "seed": seed,
        "average_score": per_round(stats, "score"),
        "kills_per_round": per_round(stats, "kills"),
        "coins_per_round": per_round(stats, "coins"),
        "crates_per_round": per_round(stats, "crates"),
        "self_kills_per_round": per_round(stats, "suicides"),
        "bombs_per_round": per_round(stats, "bombs"),
        "kills_per_bomb": kills_per_bomb,
        "invalid_per_round": per_round(stats, "invalid"),
        "steps_per_round": per_round(stats, "steps"),
    }

    results.append(result)

    print(
        f"Seed {seed}: "
        f"score={result['average_score']:.2f}, "
        f"kills={result['kills_per_round']:.2f}, "
        f"self-kills={result['self_kills_per_round']:.2f}, "
        f"coins={result['coins_per_round']:.2f}"
    )


# --------------------------------------------------------------
# Save individual seed results
# --------------------------------------------------------------

runs_file = RESULTS_DIR / f"{AGENT}_runs.csv"

with open(runs_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys(),
        delimiter=";",
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------------------
# Aggregate results
# --------------------------------------------------------------

metric_names = [
    "average_score",
    "kills_per_round",
    "coins_per_round",
    "crates_per_round",
    "self_kills_per_round",
    "bombs_per_round",
    "kills_per_bomb",
    "invalid_per_round",
    "steps_per_round",
]

summary_rows = []

print(
    f"\n========== TASK 3 SUMMARY VS {OPPONENT} ==========\n"
)

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
        fieldnames=[
            "metric",
            "mean",
            "standard_deviation",
        ],
        delimiter=";",
    )

    writer.writeheader()
    writer.writerows(summary_rows)


print(f"\nSaved individual runs to: {runs_file}")
print(f"Saved aggregate summary to: {summary_file}")