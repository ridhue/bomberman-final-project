"""Watch a saved evaluation round with the game replay interface."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True, help='Evaluation output folder containing replays.')
    parser.add_argument('--seed', type=int, help='Choose a particular recorded round rather than the first self-kill.')
    parser.add_argument('--turn-based', action='store_true', help='Press Enter to advance each recorded step.')
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.seed is None:
        data_file = next(folder.glob('*_mask_*.json'))
        rows = json.loads(data_file.read_text())['episodes']
        chosen = next((r for r in rows if r['suicides']), rows[0])
        seed = chosen['seed']
    else:
        seed = args.seed
    replay = folder / 'replays' / f'seed_{seed}.pt'
    if not replay.is_file():
        raise FileNotFoundError(replay)
    print(f'Watching recorded seed {seed}: {replay}', flush=True)
    print('Watch team_agent. This repeats saved actions without running or training either model.', flush=True)
    import main as game
    log_dir = folder / 'replay_view_logs'
    log_dir.mkdir(exist_ok=True)
    argv = ['replay', str(replay), '--update-interval', '0.25', '--log-dir', str(log_dir)]
    if args.turn_based:
        argv.append('--turn-based')
        print('Press Enter for each step.', flush=True)
    game.main(argv)


if __name__ == '__main__':
    main()
