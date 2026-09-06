import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "task2"

COMPARISON_FILE = RESULTS_ROOT / "task2_agent_comparison.csv"


if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"Comparison file not found: {COMPARISON_FILE}\n"
        "Run compare_task2_agents.py first."
    )


rows = []

with open(COMPARISON_FILE, "r", newline="") as file:
    reader = csv.DictReader(file, delimiter=";")

    for row in reader:
        rows.append(row)


agents = [row["agent"] for row in rows]


# --------------------------------------------------------------
# Plot 1: crates destroyed per round
# --------------------------------------------------------------

crate_means = [
    float(row["crates_per_round_mean"])
    for row in rows
]

crate_sds = [
    float(row["crates_per_round_sd"])
    for row in rows
]

plt.figure(figsize=(8, 5))

plt.bar(
    agents,
    crate_means,
    yerr=crate_sds,
    capsize=5,
)

plt.ylabel("Crates destroyed per round")
plt.title("Task 2: Crate Destruction")
plt.xticks(rotation=20)
plt.tight_layout()

crate_plot = RESULTS_ROOT / "task2_crates_comparison.png"

plt.savefig(crate_plot, dpi=200)
plt.close()


# --------------------------------------------------------------
# Plot 2: self-kills per round
# --------------------------------------------------------------

self_kill_means = [
    float(row["self_kills_per_round_mean"])
    for row in rows
]

self_kill_sds = [
    float(row["self_kills_per_round_sd"])
    for row in rows
]

plt.figure(figsize=(8, 5))

plt.bar(
    agents,
    self_kill_means,
    yerr=self_kill_sds,
    capsize=5,
)

plt.ylabel("Self-kills per round")
plt.title("Task 2: Bomb Safety")
plt.xticks(rotation=20)
plt.tight_layout()

safety_plot = RESULTS_ROOT / "task2_self_kills_comparison.png"

plt.savefig(safety_plot, dpi=200)
plt.close()


# --------------------------------------------------------------
# Plot 3: crates destroyed per bomb
# --------------------------------------------------------------

efficiency_means = [
    float(row["crates_per_bomb_mean"])
    for row in rows
]

efficiency_sds = [
    float(row["crates_per_bomb_sd"])
    for row in rows
]

plt.figure(figsize=(8, 5))

plt.bar(
    agents,
    efficiency_means,
    yerr=efficiency_sds,
    capsize=5,
)

plt.ylabel("Crates destroyed per bomb")
plt.title("Task 2: Bomb Efficiency")
plt.xticks(rotation=20)
plt.tight_layout()

efficiency_plot = RESULTS_ROOT / "task2_crates_per_bomb_comparison.png"

plt.savefig(efficiency_plot, dpi=200)
plt.close()


print(f"Saved crate comparison to: {crate_plot}")
print(f"Saved self-kill comparison to: {safety_plot}")
print(f"Saved bomb-efficiency comparison to: {efficiency_plot}")