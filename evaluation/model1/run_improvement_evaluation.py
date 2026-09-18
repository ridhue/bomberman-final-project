"""Reproducible model-1 comparisons using the existing game mechanics.

Each episode reseeds the world, Python random and NumPy random after setup.
Opponents share those latter streams, as in the supplied sequential framework.
Common seeds give the same initial maps, not identical opponent trajectories
after policies diverge. No game rules or decision time limits are changed.
"""
import argparse
from collections import Counter, deque
import csv
import hashlib
import json
import logging
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import settings as s
from environment import BombeRLeWorld, WorldArgs
from agent_code.team_agent import callbacks

AGENT_DIR = ROOT / 'agent_code' / 'team_agent'


def summarize(rows):
    n = len(rows)
    metrics = ['score', 'coins', 'kills', 'suicides', 'deaths', 'crates',
               'bombs', 'invalid', 'steps', 'loop_fraction', 'survived']
    result = {'rounds': n}
    for metric in metrics:
        values = [r[metric] for r in rows]
        result[metric] = statistics.mean(values)
        result[metric + '_se'] = (statistics.stdev(values) / n ** 0.5 if n > 1 else 0.0)
    result['max_act_seconds'] = max(r['max_act_seconds'] for r in rows)
    result['mean_act_seconds'] = sum(r['act_seconds'] for r in rows) / max(1, sum(r['steps'] for r in rows))
    result['score_wins'] = statistics.mean(r['score_win'] for r in rows)
    return result


def evaluate(model_path, mask, scenario, opponents, seeds, log_dir, bomb_margin=0, replay_dir=None, persistent_escape_margin=False):
    callbacks.MODEL_FILE = str(model_path.resolve())
    args = WorldArgs(no_gui=True, fps=30, turn_based=False, update_interval=0,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=str(log_dir),
                     save_stats=False, match_name=None, seed=seeds[0],
                     silence_errors=False, scenario=scenario)
    world = BombeRLeWorld(args, [('team_agent', False)] + [(a, False) for a in opponents])
    team = world.agents[0]
    policy = team.backend.runner.fake_self
    policy.use_legal_action_mask = mask in ('on', 'survival')
    policy.use_survival_action_mask = mask == 'survival'
    policy.bomb_escape_margin = bomb_margin
    policy.use_persistent_escape_margin = persistent_escape_margin
    rows = []
    for seed in seeds:
        if replay_dir is not None:
            world.args = world.args._replace(save_replay=str(replay_dir / f'seed_{seed}.pt'))
        world.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        world.new_round()
        positions = deque(maxlen=8)
        loops = 0
        actions = Counter()
        max_act = 0.0
        opponent_max_act = [0.0 for _ in world.agents[1:]]
        while world.running:
            was_alive = not team.dead
            if was_alive:
                positions.append((team.x, team.y))
                # Eight consecutive alternating positions (A,B,A,B,A,B,A,B).
                if (len(positions) == 8 and positions[0] != positions[1]
                        and all(positions[i] == positions[i % 2] for i in range(8))):
                    loops += 1
            old_time = team.statistics['time']
            opponent_old_times = [a.statistics['time'] for a in world.agents[1:]]
            world.do_step()
            for i, a in enumerate(world.agents[1:]):
                opponent_max_act[i] = max(opponent_max_act[i], a.statistics['time'] - opponent_old_times[i])
            if was_alive:
                actions[team.last_action] += 1
                max_act = max(max_act, team.statistics['time'] - old_time)
        stats = team.statistics
        rows.append({
            'seed': seed, 'score': team.score,
            **{k: stats[k] for k in ['coins', 'kills', 'suicides', 'crates', 'bombs', 'invalid', 'steps']},
            'deaths': int(team.dead), 'survived': int(not team.dead),
            'loop_fraction': loops / max(1, stats['steps']),
            'act_seconds': stats['time'], 'max_act_seconds': max_act,
            'score_win': int(bool(opponents) and team.score > max(a.score for a in world.agents[1:])),
            'opponent_scores': [a.score for a in world.agents[1:]],
            'opponent_statistics': [dict(name=a.name, code_name=a.code_name,
                max_act_seconds=opponent_max_act[i],
                **{k: a.statistics[k] for k in ['coins', 'kills', 'suicides', 'time', 'steps']})
                for i, a in enumerate(world.agents[1:])],
            'actions': dict(actions),
        })
        if len(rows) % 10 == 0:
            print(f'  evaluated {len(rows)}/{len(seeds)} episodes', flush=True)
    world.end()
    # The original framework adds handlers on every setup. Close them between
    # variants so a large comparison does not accumulate open log files.
    for name in ['BombeRLeWorld'] + [a.name + suffix for a in world.agents for suffix in ['_code', '_wrapper']]:
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['team-agent-model.pt'])
    parser.add_argument('--masks', nargs='+', choices=['off', 'on', 'survival'], default=['off', 'on'])
    parser.add_argument('--scenario', choices=s.SCENARIOS, default='classic')
    parser.add_argument('--opponents', nargs='*', default=['rule_based_agent'] * 3)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=10001)
    parser.add_argument('--bomb-margin', type=int, choices=range(4), default=0)
    parser.add_argument('--save-replays', action='store_true', help='Save each evaluated game for trajectory investigation.')
    parser.add_argument('--persistent-escape-margin', action='store_true', help='Experimental one-step ongoing escape buffer for tracked own bombs.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or len(args.opponents) > 3:
        parser.error('Need positive rounds and at most three opponents')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    replay_dir = output / 'replays' if args.save_replays else None
    if replay_dir is not None:
        if len(args.models) != 1 or len(args.masks) != 1:
            parser.error('--save-replays currently requires exactly one model and one mask')
        replay_dir.mkdir(exist_ok=False)
    # Suppress log I/O without changing physics, the board or time limits.
    s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = logging.ERROR
    seeds = list(range(args.seed_start, args.seed_start + args.rounds))
    protocol = {'scenario': args.scenario, 'opponents': args.opponents, 'seeds': seeds,
                'loop_definition': '8 consecutive positions alternating between two distinct tiles',
                'mask_definitions': {'off': 'unrestricted historical policy', 'on': 'observed physical legality',
                                     'survival': 'legality plus survival path under observed bomb schedule; static crates/opponents'},
                'python': sys.version, 'numpy': np.__version__}
    protocol['bomb_escape_margin'] = args.bomb_margin
    protocol['save_replays'] = args.save_replays
    protocol['persistent_escape_margin'] = args.persistent_escape_margin
    protocol['source_sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in AGENT_DIR.glob('*.py')
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    summaries = []
    for model in args.models:
        path = Path(model)
        if not path.is_absolute():
            path = AGENT_DIR / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for mask_name in args.masks:
            label = str(path.relative_to(AGENT_DIR)).replace('\\', '/').replace('/', '__') + '_mask_' + mask_name
            if args.bomb_margin:
                label += '_margin_' + str(args.bomb_margin)
            if args.persistent_escape_margin:
                label += '_persistent_margin'
            start = time.monotonic()
            rows = evaluate(path, mask_name, args.scenario, args.opponents, seeds, logs, args.bomb_margin, replay_dir, args.persistent_escape_margin)
            summary = {'variant': label, 'model': str(path), 'sha256': digest,
                       'mask': mask_name, 'bomb_margin': args.bomb_margin, **summarize(rows)}
            summaries.append(summary)
            (output / (label + '.json')).write_text(json.dumps({'summary': summary, 'episodes': rows}, indent=2))
            # Save after every variant so partial progress is reviewable.
            (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
            with (output / 'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
                writer.writeheader()
                writer.writerows(summaries)
            print(f"{label}: score={summary['score']:.3f}, self-kill={summary['suicides']:.1%}, "
                  f"loop={summary['loop_fraction']:.1%}, bombs={summary['bombs']:.2f}, "
                  f"{time.monotonic()-start:.1f}s", flush=True)


if __name__ == '__main__':
    main()
