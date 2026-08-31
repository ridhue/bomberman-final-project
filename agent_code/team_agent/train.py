import pickle
from typing import List

from .config import ACTIONS, EPSILON_MIN, EPSILON_DECAY, MODEL_FILE
from .features import state_to_features
from .rewards import add_custom_events, reward_from_events


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    _learn_step(self, old_game_state, self_action, new_game_state, events, done=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    _learn_step(self, last_game_state, last_action, None, events, done=True)

    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)


def _learn_step(self, old_state, action, new_state, events: List[str], done: bool):
    if old_state is None or action is None or action not in ACTIONS:
        return

    features = state_to_features(old_state)
    next_features = None if done else state_to_features(new_state)
    action_idx = ACTIONS.index(action)
    all_events = add_custom_events(old_state, action, new_state, events)

    reward = reward_from_events(all_events, config_name="C")

    td_error = self.model.update(features, action_idx, reward, next_features, done)
    self.logger.debug(f"TD error: {td_error:.3f}, reward: {reward}")