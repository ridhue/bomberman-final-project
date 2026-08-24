import csv
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "task1"

INPUT_FILE = RESULTS_DIR / "task1_agent_comparison.csv"
OUTPUT_FILE = RESULTS_DIR / "task1_coins_comparison.png"


agents = []
coins_mean = []
coins_sd = []


with open(INPUT_FILE, "r") as file:
    reader = csv.DictReader(file, delimiter=";")

    for row in reader:
        agents.append(row["agent"])
        coins_mean.append(float(row["coins_mean"]))
        coins_sd.append(float(row["coins_sd"]))


# Make agent names nicer for the graph
display_names = [
    name.replace("_", " ").title()
    for name in agents
]


plt.figure(figsize=(10, 6))

plt.bar(
    display_names,
    coins_mean,
    yerr=coins_sd,
    capsize=5
)

plt.ylabel("Coins collected per round")
plt.xlabel("Agent")
plt.title("Task 1: Coin Collection Performance")

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Saved figure to: {OUTPUT_FILE}")