"""Train with recorded scenario, seed, and stage configuration.

Run parameters are passed to setup_training through environment variables.
Each stage writes checkpoints and metrics to its own directory.

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
    parser.add_argument('--resume-from', type=Path, help='Warm-start from this existing checkpoint')
    parser.add_argument('--output-model', type=Path, help='Save to a separate model file')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--epsilon-start', type=float)
    parser.add_argument('--epsilon-min', type=float)
    parser.add_argument('--reward-config', default='S3_C')
    parser.add_argument('--legal-mask', choices=['on', 'off'], default='off')
    parser.add_argument('--enhanced-features', action='store_true')
    parser.add_argument('--survival-mask', action='store_true')
    parser.add_argument('--bomb-margin', type=int, choices=range(4), default=0)
    parser.add_argument('--revisit-penalty', type=float, default=0)
    parser.add_argument('--in-process', action='store_true', help='Run the same headless world directly instead of launching main.py')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-step logging in direct mode')
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
    env['TRAIN_OPPONENTS'] = ','.join(args.opponents)
    env['REWARD_CONFIG'] = args.reward_config
    env['LEGAL_ACTION_MASK'] = '1' if args.legal_mask == 'on' else '0'
    env['ENHANCED_FEATURES'] = '1' if args.enhanced_features else '0'
    env['SURVIVAL_ACTION_MASK'] = '1' if args.survival_mask else '0'
    env['BOMB_ESCAPE_MARGIN'] = str(args.bomb_margin)
    env['REVISIT_PENALTY'] = str(args.revisit_penalty)
    for key, value in [('LEARNING_RATE', args.learning_rate), ('EPSILON_START', args.epsilon_start), ('EPSILON_MIN', args.epsilon_min)]:
        if value is not None:
            env[key] = str(value)
    if args.resume_from and not args.output_model:
        parser.error('--resume-from requires --output-model to preserve the source checkpoint')
    if args.output_model:
        checkpoint_dir = PROJECT_ROOT / 'agent_code' / 'team_agent' / 'checkpoints' / args.stage
        if checkpoint_dir.exists():
            parser.error(f'Checkpoint directory already exists: {checkpoint_dir}; choose a new stage name')
        target = args.output_model.resolve()
        if target.exists():
            parser.error(f'Output model already exists: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        env['MODEL_FILE_OVERRIDE'] = str(target)
        if args.resume_from:
            import shutil
            source = args.resume_from.resolve()
            if not source.is_file():
                parser.error(f'Initial checkpoint not found: {source}')
            shutil.copy2(source, target)
            env['RESUME_TRAINING'] = '1'
            env['INITIAL_MODEL'] = str(source)

    command = [
        sys.executable, "main.py", "play",
        "--agents", "team_agent", *args.opponents,
        "--train", "1",
        "--scenario", args.scenario,
        "--seed", str(args.seed),
        "--n-rounds", str(args.n_rounds),
        "--no-gui",
    ]
    if args.in_process:
        import logging
        import random
        import numpy as np
        os.environ.update(env)
        os.chdir(PROJECT_ROOT)
        import settings as s
        if args.quiet:
            s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = logging.ERROR
        from environment import BombeRLeWorld, WorldArgs
        log_dir = PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        world_args = WorldArgs(no_gui=True, fps=30, turn_based=False, update_interval=0,
                               save_replay=False, replay=None, make_video=False,
                               continue_without_training=False, log_dir=str(log_dir),
                               save_stats=False, match_name=None, seed=args.seed,
                               silence_errors=False, scenario=args.scenario)
        world = BombeRLeWorld(world_args, [('team_agent', True)] + [(a, False) for a in args.opponents])
        # Baseline opponents reseed NumPy in setup; seed after setup as well.
        random.seed(args.seed)
        np.random.seed(args.seed)
        for round_idx in range(args.n_rounds):
            world.new_round()
            while world.running:
                world.do_step()
            if (round_idx + 1) % 25 == 0:
                print(f'Finished {round_idx + 1}/{args.n_rounds} rounds', flush=True)
        world.end()
    else:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
