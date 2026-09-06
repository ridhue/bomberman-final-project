ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
