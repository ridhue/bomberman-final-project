ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# bump when the feature vector shape/meaning changes
FEATURE_VERSION = "v1_stage1_nav_coin"

MODEL_FILE = "team-agent-model.pt"

# untuned baseline
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995


# Stage 1: Baseline training, no opponent
STAGE = 1
OPPONENT = None
ROUNDS = 20000