import os
import pickle

import numpy as np

from .config import ACTIONS, EPSILON_START, LEARNING_RATE, DISCOUNT_FACTOR, MODEL_FILE
from .features import state_to_features, N_FEATURES
from .model import DoubleQModel


def setup(self):
    self.epsilon = EPSILON_START

    if self.train or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up fresh double Q-model.")
        self.model = DoubleQModel(len(ACTIONS), N_FEATURES, LEARNING_RATE, DISCOUNT_FACTOR)
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    if self.train and np.random.random() < self.epsilon:
        self.logger.debug("Exploring: random action.")
        return str(np.random.choice(ACTIONS))

    action_idx = self.model.best_action(features)
    return ACTIONS[action_idx]
