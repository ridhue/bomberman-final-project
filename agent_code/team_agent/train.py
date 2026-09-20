import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import events as e
from .config import ACTIONS, EPSILON_MIN, EPSILON_DECAY, MODEL_FILE, REWARD_CONFIG, REVISIT_PENALTY
from .callbacks import policy_features, policy_action_mask
from .rewards import add_custom_events, reward_from_events

CHECKPOINT_DIR = Path("checkpoints") / os.environ.get("TRAIN_STAGE", "stage1")
CHECKPOINT_INTERVAL = 500
BEST_SCORE_WINDOW = 10

# Periodic greedy evaluation is logged separately from exploratory training.
EVAL_INTERVAL = 1000
EVAL_ROUNDS = 20


def setup_training(self):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    self.round_count = 0
    self.recent_scores = []
    self.best_score = float("-inf")
    self.td_errors = []
    self.coins_this_round = 0

    self.eval_mode = False
    self.eval_rounds_remaining = 0
    self.eval_batch = []
    self.pre_eval_epsilon = None

    # Set by run_training.py; falls back to "unknown" if main.py is invoked directly.
    with open(CHECKPOINT_DIR / "config_snapshot.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "scenario": os.environ.get("TRAIN_SCENARIO", "unknown"),
            "seed": os.environ.get("TRAIN_SEED", "unknown"),
            "learning_rate": self.model.alpha,
            "discount_factor": self.model.gamma,
            "epsilon_start": self.epsilon,
            "epsilon_min": EPSILON_MIN,
            "epsilon_decay": EPSILON_DECAY,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "best_score_window": BEST_SCORE_WINDOW,
            "eval_interval": EVAL_INTERVAL,
            "eval_rounds": EVAL_ROUNDS,
            "reward_config": REWARD_CONFIG,
            "revisit_penalty": REVISIT_PENALTY,
            "n_features": self.model.n_features,
            "legal_action_mask": self.use_legal_action_mask,
            "survival_action_mask": self.use_survival_action_mask,
            "bomb_escape_margin": self.bomb_escape_margin,
            "initial_model": os.environ.get('INITIAL_MODEL', 'fresh'),
            "opponents": os.environ.get('TRAIN_OPPONENTS', 'unknown'),
        }, f, indent=2)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    _learn_step(self, old_game_state, self_action, new_game_state, events, done=False, learn=not self.eval_mode)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    was_eval_round = self.eval_mode
    _learn_step(self, last_game_state, last_action, None, events, done=True, learn=not was_eval_round)

    self.round_count += 1
    score = last_game_state["self"][1]

    if was_eval_round:
        self.eval_batch.append({
            "score": score,
            "coins": self.coins_this_round,
            "self_kill": events.count(e.KILLED_SELF),
        })
        self.coins_this_round = 0
        self.eval_rounds_remaining -= 1
        if self.eval_rounds_remaining == 0:
            _flush_eval_batch(self)
            self.eval_mode = False
            self.epsilon = self.pre_eval_epsilon
            self.model.epsilon = self.epsilon
        return

    # Decay exploration toward EPSILON_MIN after each training round.
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.model.epsilon = self.epsilon

    # metrics, one line per round
    mean_td = sum(self.td_errors) / len(self.td_errors) if self.td_errors else 0.0
    with open(CHECKPOINT_DIR / "metrics.jsonl", "a") as f:
        f.write(json.dumps({
            "round": self.round_count,
            "score": score,
            "coins": self.coins_this_round,
            "self_kills": events.count(e.KILLED_SELF),
            "epsilon": round(self.epsilon, 6),
            "mean_td_error": round(mean_td, 6),
        }) + "\n")
    self.td_errors = []
    self.coins_this_round = 0

    # best model, on a rolling window
    self.recent_scores.append(score)
    self.recent_scores = self.recent_scores[-BEST_SCORE_WINDOW:]
    if len(self.recent_scores) == BEST_SCORE_WINDOW:
        window_mean = sum(self.recent_scores) / BEST_SCORE_WINDOW
        if window_mean > self.best_score:
            self.best_score = window_mean
            with open(CHECKPOINT_DIR / "checkpoint_best.pt", "wb") as f:
                pickle.dump(self.model, f)

    # periodic snapshot
    if self.round_count % CHECKPOINT_INTERVAL == 0:
        with open(CHECKPOINT_DIR / f"checkpoint_{self.round_count:06d}.pt", "wb") as f:
            pickle.dump(self.model, f)

    # Save the current model.
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)

    # kick off the next greedy eval batch
    if self.round_count % EVAL_INTERVAL == 0:
        self.eval_mode = True
        self.eval_rounds_remaining = EVAL_ROUNDS
        self.eval_batch = []
        self.pre_eval_epsilon = self.epsilon
        self.epsilon = 0.0
        self.model.epsilon = 0.0


def _flush_eval_batch(self):
    n = len(self.eval_batch)
    if n == 0:
        return

    self_kills = sum(r["self_kill"] for r in self.eval_batch)
    avg_score = sum(r["score"] for r in self.eval_batch) / n
    avg_coins = sum(r["coins"] for r in self.eval_batch) / n

    with open(CHECKPOINT_DIR / "eval_metrics.jsonl", "a") as f:
        f.write(json.dumps({
            "round": self.round_count,
            "n_eval_rounds": n,
            "self_kill_rate": round(self_kills / n, 4),
            "avg_score": round(avg_score, 3),
            "avg_coins": round(avg_coins, 3),
        }) + "\n")


def _learn_step(self, old_state, action, new_state, events: List[str], done: bool, learn: bool = True):
    if old_state is None or action is None or action not in ACTIONS:
        return

    self.coins_this_round += events.count(e.COIN_COLLECTED)

    if not learn:
        return

    features = self.action_features
    next_features = None if done else policy_features(self, new_state)
    action_idx = ACTIONS.index(action)
    all_events = add_custom_events(old_state, action, new_state, events)

    reward = reward_from_events(all_events, config_name=REWARD_CONFIG)
    # Optional controlled experiment: discourage repeated safe positions.
    # Apply only to successful moves outside danger, so escaping is not punished.
    if (REVISIT_PENALTY and new_state is not None and e.INVALID_ACTION not in events
            and action in ('UP', 'RIGHT', 'DOWN', 'LEFT')
            and features[15] == 0 and next_features[15] == 0
            and new_state['self'][3] in self.position_history):
        reward -= REVISIT_PENALTY

    next_mask = policy_action_mask(self, new_state) if not done else None
    td_error = self.model.update(features, action_idx, reward, next_features, done, next_mask)
    self.td_errors.append(td_error)
    self.logger.debug(f"TD error: {td_error:.3f}, reward: {reward}")
