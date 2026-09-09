"""Launch a training run with scenario/seed/stage captured for the config snapshot.

train.py can't see main.py's --scenario/--seed directly (game_state doesn't
carry them), so this wrapper passes them through as env vars that
setup_training() reads. --stage also controls which checkpoints/<stage>/
folder gets written to, so runs for different stages don't overwrite each
other's checkpoints/metrics. Use this instead of calling main.py directly
when you need the run to be reproducible.

Usage:
    python -m agent_code.team_agent.run_training --stage stage2 --scenario classic --seed 42 --n-rounds 20000
    python -m agent_code.team_agent.run_training --stage stage3 --scenario classic --seed 42 --n-rounds 20000 --opponents coin_collector_agent
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="e.g. stage1, stage2 - picks checkpoints/<stage>/")
    parser.add_argument("--scenario", default="coin-heaven")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-rounds", type=int, default=20000)
    parser.add_argument(
        "--opponents", nargs="*", default=[],
        help="e.g. coin_collector_agent peaceful_agent - needed for Stage 3+, omit for solo training",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["TRAIN_STAGE"] = args.stage
    env["TRAIN_SCENARIO"] = args.scenario
    env["TRAIN_SEED"] = str(args.seed)
    env["TRAIN_N_ROUNDS"] = str(args.n_rounds)

    command = [
        sys.executable, "main.py", "play",
        "--agents", "team_agent", *args.opponents,
        "--train", "1",
        "--scenario", args.scenario,
        "--seed", str(args.seed),
        "--n-rounds", str(args.n_rounds),
        "--no-gui",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
