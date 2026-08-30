import csv
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "task1"

INPUT_FILE = RESULTS_DIR / "task1_agent_comparison.csv"
OUTPUT_FILE = RESULTS_DIR / "task1_completion_comparison.png"


agents = []
completion_mean = []
completion_sd = []


with open(INPUT_FILE, "r") as file:
    reader = csv.DictReader(file, delimiter=";")

    for row in reader:
        agents.append(row["agent"])

        # Convert 0–1 proportions to percentages
        completion_mean.append(
            float(row["completion_mean"]) * 100
        )
        completion_sd.append(
            float(row["completion_sd"]) * 100
        )


display_names = [
    name.replace("_", " ").title()
    for name in agents
]


plt.figure(figsize=(10, 6))

plt.bar(
    display_names,
    completion_mean,
    yerr=completion_sd,
    capsize=5
)

plt.ylabel("Task completion rate (%)")
plt.xlabel("Agent")
plt.title("Task 1: Completion Rate")

plt.ylim(0, 110)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Saved figure to: {OUTPUT_FILE}")