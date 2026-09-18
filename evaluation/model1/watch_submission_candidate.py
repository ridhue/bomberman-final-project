"""Verify the deployed checkpoint and watch matches against three rule agents."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / 'agent_code' / 'team_agent'
CANDIDATE = AGENT_DIR / 'team-agent-model.pt'
EXPECTED_SHA256 = json.loads((AGENT_DIR / 'MODEL_INFO.json').read_text(encoding='utf-8'))['deployment_checkpoint_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seed', type=int, default=81001)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--headless', action='store_true', help='Run a normal-framework smoke check without a window.')
    parser.add_argument('--persistent-escape-margin', action=argparse.BooleanOptionalAction,
                        default=True, help='Keep the ongoing own-bomb escape buffer enabled.')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    digest = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f'Candidate differs from the evaluated checkpoint: {digest}')
    sys.path.insert(0, str(ROOT))
    os.environ.update(MODEL_FILE_OVERRIDE=str(CANDIDATE), LEGAL_ACTION_MASK='1', SURVIVAL_ACTION_MASK='1', BOMB_ESCAPE_MARGIN='1')
    os.environ['PERSISTENT_ESCAPE_MARGIN'] = '1' if args.persistent_escape_margin else '0'
    import numpy as np
    with CANDIDATE.open('rb') as f:
        model = pickle.load(f)
    if model.weights.shape != (6, 57) or not np.isfinite(model.weights).all():
        raise RuntimeError('Unexpected model dimensions or non-finite weights')
    from agent_code.team_agent.callbacks import setup
    policy = SimpleNamespace(train=False, logger=logging.getLogger('candidate-verification'))
    setup(policy)
    assert policy.model.n_features == 57
    assert policy.use_legal_action_mask and policy.use_survival_action_mask
    assert policy.bomb_escape_margin == 1
    assert policy.use_persistent_escape_margin == args.persistent_escape_margin
    assert np.array_equal(policy.model.weights, model.weights)
    print(f'CHECK PASSED: exact evaluated candidate SHA-256 {digest}', flush=True)
    print('Policy: 57 features, legal + survival masks, bomb margin 1, training OFF.', flush=True)
    print(f'Ongoing escape buffer: {"ON" if args.persistent_escape_margin else "OFF"}', flush=True)
    print(f'Model: {CANDIDATE}', flush=True)
    if args.verify_only:
        return
    os.chdir(ROOT)
    import main as game
    log_dir = ROOT / 'results' / 'model1_improvement' / 'candidate_watch' / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / 'logs'
    log_dir.mkdir(parents=True)
    argv = ['play', '--agents', 'team_agent', 'rule_based_agent', 'rule_based_agent', 'rule_based_agent', '--train', '0', '--scenario', 'classic', '--n-rounds', str(args.rounds), '--seed', str(args.seed), '--update-interval', '0.1', '--log-dir', str(log_dir)]
    if args.headless:
        argv.append('--no-gui')
    else:
        print('Watch team_agent. At each round end press an arrow key or Enter to continue. Close the window to stop.', flush=True)
    game.main(argv)
    if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Candidate checkpoint changed during play')


if __name__ == '__main__':
    main()
