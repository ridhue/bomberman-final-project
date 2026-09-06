import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import events as e
from .config import ACTIONS, EPSILON_MIN, EPSILON_DECAY, MODEL_FILE
from .features import state_to_features
from .rewards import add_custom_events, reward_from_events

CHECKPOINT_DIR = Path("checkpoints/stage1")
CHECKPOINT_INTERVAL = 500
BEST_SCORE_WINDOW = 10


def setup_training(self):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    self.round_count = 0
    self.recent_scores = []
    self.best_score = float("-inf")
    self.td_errors = []
    self.coins_this_round = 0

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
        }, f, indent=2)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    _learn_step(self, old_game_state, self_action, new_game_state, events, done=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    _learn_step(self, last_game_state, last_action, None, events, done=True)

    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.model.epsilon = self.epsilon

    self.round_count += 1
    score = last_game_state["self"][1]

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

    # live model, unchanged
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)


def _learn_step(self, old_state, action, new_state, events: List[str], done: bool):
    if old_state is None or action is None or action not in ACTIONS:
        return

    self.coins_this_round += events.count(e.COIN_COLLECTED)

    features = state_to_features(old_state)
    next_features = None if done else state_to_features(new_state)
    action_idx = ACTIONS.index(action)
    all_events = add_custom_events(old_state, action, new_state, events)

    reward = reward_from_events(all_events, config_name="C")

    td_error = self.model.update(features, action_idx, reward, next_features, done)
    self.td_errors.append(td_error)
    self.logger.debug(f"TD error: {td_error:.3f}, reward: {reward}")