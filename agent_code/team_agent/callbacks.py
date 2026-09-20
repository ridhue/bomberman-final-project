import os
import pickle
import sys
from collections import deque

import numpy as np

from .config import (
    ACTIONS, EPSILON_START, LEARNING_RATE, DISCOUNT_FACTOR, MODEL_FILE,
    RESUME_TRAINING, USE_LEGAL_ACTION_MASK, ENHANCED_FEATURES,
    USE_SURVIVAL_ACTION_MASK, BOMB_ESCAPE_MARGIN,
)
from .features import state_to_features, N_FEATURES, N_ENHANCED_FEATURES, enhanced_features, legal_action_mask
from .safety import survival_action_mask, buffered_survival_action_mask
from .model import LinearQModel

# Saved checkpoints pickle LinearQModel instances, which embed the class's
# module path (e.g. "agent_code.team_agent.model") at save time. The
# submission process copies this directory under the team's own name (e.g.
# "agent_code.finetune"), which breaks that reference on load. Alias the
# original path to wherever this package actually lives, so old and renamed
# checkpoints both unpickle correctly regardless of the folder name.
sys.modules.setdefault("agent_code.team_agent", sys.modules[__package__])
sys.modules.setdefault("agent_code.team_agent.model", sys.modules[__package__ + ".model"])


def setup(self):
    self.epsilon = EPSILON_START
    self.use_legal_action_mask = USE_LEGAL_ACTION_MASK
    self.use_survival_action_mask = USE_SURVIVAL_ACTION_MASK
    self.bomb_escape_margin = BOMB_ESCAPE_MARGIN
    self.position_history = deque(maxlen=8)
    self.history_round = None
    self.own_bomb_position = None
    self.use_persistent_escape_margin = os.environ.get('PERSISTENT_ESCAPE_MARGIN', '0') == '1'

    # TRAIN_SEED (set by run_training.py) seeds weight init + exploration so a
    # given --seed reproduces a full run, not just the map/coin layout.
    train_seed = os.environ.get("TRAIN_SEED")
    seed = int(train_seed) if self.train and train_seed and train_seed != "unknown" else None
    self.rng = np.random.default_rng(seed)

    fresh_start = self.train and not RESUME_TRAINING
    if not self.train and not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f'Trained model not found: {MODEL_FILE}')
    if fresh_start or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up fresh linear Q-model.")
        self.model = LinearQModel(len(ACTIONS), N_FEATURES, LEARNING_RATE, DISCOUNT_FACTOR, seed=seed)
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)
        if self.train:
            self.epsilon = getattr(self.model, "epsilon", EPSILON_START)
            if 'EPSILON_START' in os.environ:
                self.epsilon = EPSILON_START
            if 'LEARNING_RATE' in os.environ:
                self.model.alpha = LEARNING_RATE
    if self.train and ENHANCED_FEATURES:
        self.model.expand_features(N_ENHANCED_FEATURES)
    if self.model.n_features not in (N_FEATURES, N_ENHANCED_FEATURES):
        raise ValueError(f'Unsupported model feature count: {self.model.n_features}')
    # Deployment settings travel with a selected model. Explicit environment
    # overrides still support historical comparisons and controlled training.
    if 'LEGAL_ACTION_MASK' not in os.environ:
        self.use_legal_action_mask = getattr(self.model, 'use_legal_action_mask', USE_LEGAL_ACTION_MASK)
    if 'SURVIVAL_ACTION_MASK' not in os.environ:
        self.use_survival_action_mask = getattr(self.model, 'use_survival_action_mask', USE_SURVIVAL_ACTION_MASK)
    if 'BOMB_ESCAPE_MARGIN' not in os.environ:
        self.bomb_escape_margin = getattr(self.model, 'bomb_escape_margin', BOMB_ESCAPE_MARGIN)
    if 'PERSISTENT_ESCAPE_MARGIN' not in os.environ:
        self.use_persistent_escape_margin = getattr(self.model, 'use_persistent_escape_margin', False)


def policy_features(self, game_state):
    base = state_to_features(game_state)
    if self.model.n_features == N_ENHANCED_FEATURES:
        return enhanced_features(game_state, self.position_history, base)
    return base


def policy_action_mask(self, game_state):
    if self.use_survival_action_mask:
        if getattr(self, 'use_persistent_escape_margin', False):
            return buffered_survival_action_mask(game_state, ACTIONS, getattr(self, 'own_bomb_position', None), self.bomb_escape_margin)
        return survival_action_mask(game_state, ACTIONS, self.bomb_escape_margin)
    if self.use_legal_action_mask:
        return legal_action_mask(game_state, ACTIONS)
    return None


def act(self, game_state: dict) -> str:
    if self.history_round != game_state['round']:
        self.position_history.clear()
        self.history_round = game_state['round']
        self.own_bomb_position = None
    if getattr(self, 'own_bomb_position', None) is not None and not any(pos == self.own_bomb_position for pos, _ in game_state['bombs']):
        self.own_bomb_position = None
    features = policy_features(self, game_state)
    # Training must use exactly the features on which this action was chosen.
    self.action_features = features
    self.position_history.append(game_state['self'][3])
    mask = policy_action_mask(self, game_state)

    if self.train and self.rng.random() < self.epsilon:
        self.logger.debug("Exploring: random action.")
        choices = np.asarray(ACTIONS)[mask] if mask is not None else ACTIONS
        action = str(self.rng.choice(choices))
    else:
        action_idx = self.model.best_action(features, mask)
        action = ACTIONS[action_idx]
    if action == 'BOMB':
        self.own_bomb_position = game_state['self'][3]
    return action
