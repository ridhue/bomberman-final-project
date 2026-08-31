ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# bump when the feature vector shape/meaning changes
FEATURE_VERSION = "v1_stage1_nav_coin"

MODEL_FILE = "team-agent-v2-model.pt"

# same values as team_agent's model1 baseline, kept identical on purpose
# so the only difference between the two models is the algorithm itself
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
