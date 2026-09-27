import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Select a checkpoint for evaluation or continuation training.
MODEL_FILE = os.environ.get("MODEL_FILE_OVERRIDE", "team-agent-model.pt")

# Default parameters for fresh training runs.
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.01'))


DISCOUNT_FACTOR = 0.95
EPSILON_START = float(os.environ.get('EPSILON_START', '1.0'))
EPSILON_MIN = float(os.environ.get('EPSILON_MIN', '0.05'))
EPSILON_DECAY = 0.999

# Resume loads the saved weights and exploration rate.
RESUME_TRAINING = os.environ.get('RESUME_TRAINING', '0') == '1'

# Mask and feature overrides for controlled experiments.
USE_LEGAL_ACTION_MASK = os.environ.get('LEGAL_ACTION_MASK', '0') == '1'
ENHANCED_FEATURES = os.environ.get('ENHANCED_FEATURES', '0') == '1'
REWARD_CONFIG = os.environ.get('REWARD_CONFIG', 'S3_C')
REVISIT_PENALTY = float(os.environ.get('REVISIT_PENALTY', '0'))
USE_SURVIVAL_ACTION_MASK = os.environ.get('SURVIVAL_ACTION_MASK', '0') == '1'
BOMB_ESCAPE_MARGIN = int(os.environ.get('BOMB_ESCAPE_MARGIN', '0'))
