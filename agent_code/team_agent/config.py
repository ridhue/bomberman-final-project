import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Override lets a one-off eval point at a specific checkpoint without
# touching the shared live model file - safe to change on disk even while
# other processes are mid-run, since they already have this constant
# cached in memory from their own startup import.
MODEL_FILE = os.environ.get("MODEL_FILE_OVERRIDE", "team-agent-model.pt")

# untuned baseline
LEARNING_RATE = 0.01


DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

# False (default): every training run starts from a fresh model/epsilon.
# True: training continues from the saved MODEL_FILE and its saved epsilon.
# Needed for deliberate multi-stage/curriculum training; keep False for
# one-off experiments so results stay comparable.
RESUME_TRAINING = False
