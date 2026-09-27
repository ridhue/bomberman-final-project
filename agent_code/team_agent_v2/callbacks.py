import os
import pickle

import numpy as np

from .config import ACTIONS, EPSILON_START, EPSILON_RESUME, LEARNING_RATE, DISCOUNT_FACTOR, MODEL_FILE, RESUME_TRAINING
from .features import state_to_features, N_FEATURES
from .model import DoubleQModel


def setup(self):
    fresh_start = self.train and not RESUME_TRAINING

    if fresh_start or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up fresh double Q-model.")
        self.model = DoubleQModel(len(ACTIONS), N_FEATURES, LEARNING_RATE, DISCOUNT_FACTOR)
        self.epsilon = EPSILON_START
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)
        if self.train:
            self.epsilon = EPSILON_RESUME
        else:
            self.epsilon = 0.0

def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    if self.train and np.random.random() < self.epsilon:
        self.logger.debug("Exploring: random action.")
        return str(np.random.choice(ACTIONS))

    action_idx = self.model.best_action(features)
    return ACTIONS[action_idx]
