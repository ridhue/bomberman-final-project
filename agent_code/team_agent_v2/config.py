ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_FILE = "team-agent-v2-model.pt"

# same values as team_agent's model1 baseline, kept identical on purpose
# so the only difference between the two models is the algorithm itself
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

RESUME_TRAINING = True
EPSILON_RESUME = 0.15

#for stage 2 : warm-start Stage 1 weights into Stage 2 (classic — crates + bombs). Curriculum, not scratch — same recipe that rescued Model 1 (70%→16% self-kill).