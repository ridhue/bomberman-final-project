ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# bump when the feature vector shape/meaning changes
FEATURE_VERSION = "v1_stage1_nav_coin"

MODEL_FILE = "team-agent-model.pt"

# untuned baseline
LEARNING_RATE = 0.05


DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

# False (default): every training run starts from a fresh model/epsilon.
# True: training continues from the saved MODEL_FILE and its saved epsilon.
# Needed for deliberate multi-stage/curriculum training; keep False for
# one-off experiments so results stay comparable.
RESUME_TRAINING = False
