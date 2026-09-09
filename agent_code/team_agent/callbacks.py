import os
import pickle

import numpy as np

from .config import ACTIONS, EPSILON_START, LEARNING_RATE, DISCOUNT_FACTOR, MODEL_FILE, EPSILON_MIN, RESUME_TRAINING
from .features import state_to_features, N_FEATURES
from .model import LinearQModel


def setup(self):
    self.epsilon = EPSILON_START

    # TRAIN_SEED (set by run_training.py) seeds weight init + exploration so a
    # given --seed reproduces a full run, not just the map/coin layout.
    train_seed = os.environ.get("TRAIN_SEED")
    seed = int(train_seed) if self.train and train_seed and train_seed != "unknown" else None
    self.rng = np.random.default_rng(seed)

    fresh_start = self.train and not RESUME_TRAINING
    if fresh_start or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up fresh linear Q-model.")
        self.model = LinearQModel(len(ACTIONS), N_FEATURES, LEARNING_RATE, DISCOUNT_FACTOR, seed=seed)
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)
        if self.train:
            self.epsilon = getattr(self.model, "epsilon", EPSILON_START)


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    if self.train and self.rng.random() < self.epsilon:
        self.logger.debug("Exploring: random action.")
        return str(self.rng.choice(ACTIONS))

    action_idx = self.model.best_action(features)
    return ACTIONS[action_idx]
