ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_FILE = "team-agent-v2-model.pt"

# same values as team_agent's model1 baseline, kept identical on purpose
# so the only difference between the two models is the algorithm itself
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

RESUME_TRAINING = False
