# Evaluation

## Purpose

This folder contains the evaluation infrastructure for comparing
different agents and reward/model configurations independently of
the training code.

All commands should be run from the main repository directory.

## Task 1

Task 1 evaluates basic navigation and visible coin collection in the
`coin-heaven` scenario.

## Quick test

For debugging only:

20 rounds with one seed.

## Development evaluation

Current protocol:

- Scenario: coin-heaven
- 100 rounds per seed
- Seeds: 1, 2, 3, 4, 5
- Total: 500 rounds per agent
- No training during evaluation
- GUI disabled

Run:

```bash
python evaluation/task1/run_task1_evaluation.py --agent AGENT_NAME
```

Example:
```bash
python evaluation/task1/run_task1_evaluation.py --agent coin_collector_agent
```

## Metrics
### Primary metrics
- Coins per round
- Task completion rate

### Efficiency
- Steps to completion for successful rounds

### Diagnostic metrics
- Self-kills per round
- Invalid actions per round
- Bombs dropped per round

Raw episode length should not be interpreted as efficiency by itself,
because an agent may have a short episode simply because it dies early.

### Output

Results are stored in:

results/task1/AGENT_NAME/

The evaluation creates:

- JSON files for individual seeds
- CSV file containing individual run results
- CSV summary containing mean and standard deviation

### Reference agents

Current Task 1 reference agents:

- peaceful_agent
- coin_collector_agent
- rule_based_agent
- tpl_agent as the untrained/template baseline

### Comparison

After evaluating the agents, run:

```bash
python evaluation/task1/compare_task1_agents.py
```

This creates: results/task1/task1_agent_comparison.csv

### Plots

Coin collection: python evaluation/task1/plot_task1_comparison.py
Completion rate: python evaluation/task1/plot_task1_completion.py

## Planned Task 1 reward comparison

Reward definitions are located in:

`agent_code/team_agent/rewards.py`

Three initial reward configurations are defined.

### A - Sparse baseline

- COIN_COLLECTED: +10
- KILLED_SELF: -20

Purpose: test whether Task 1 can be learned mainly from outcome-based
feedback.

### B - Behaviour penalties

Same outcome rewards, plus:

- INVALID_ACTION: -1
- WAITED: -0.5
- BOMB_DROPPED: -2

Purpose: test whether discouraging unnecessary Task 1 behaviour
improves learning.

### C - Directional shaping

Same as B, plus:

- MOVED_TOWARD_COIN: +0.2
- MOVED_AWAY_FROM_COIN: -0.2

Purpose: test whether intermediate directional feedback improves
navigation learning.

The reward configurations should be compared using the same model,
features, training duration and evaluation protocol.

## Task 2 - Crates, Bombs and Escape Behaviour

### Purpose

Task 2 extends the Stage 1 navigation problem by introducing crates,
hidden coins and bomb usage.

The reward and evaluation setup therefore focuses on two goals:

1. using bombs effectively to destroy crates and reveal/collect coins
2. surviving the resulting explosions

The Stage 2 experiments use the `loot-crate` scenario.

---

### Stage 2 Reward Configurations

The reward configurations are defined in:

`agent_code/team_agent/rewards.py`

Three configurations are prepared for a controlled comparison.

#### S2_A - Outcome Rewards

The first configuration mainly rewards direct task outcomes.

- `COIN_COLLECTED`: +10
- `CRATE_DESTROYED`: +2
- `COIN_FOUND`: +2
- `KILLED_SELF`: -20

This serves as the basic Stage 2 reward baseline.

#### S2_B - Bomb Placement Shaping

S2_B extends the outcome rewards with behaviour-related penalties and
bomb-placement feedback.

Additional custom events distinguish between bombs that can hit a crate
and bombs that are unlikely to contribute to crate destruction.

- `INVALID_ACTION`: -1
- `WAITED`: -0.5
- `USEFUL_BOMB_DROPPED`: +0.5
- `USELESS_BOMB_DROPPED`: -0.5

The purpose is to test whether directly rewarding useful bomb placement
improves crate destruction efficiency.

#### S2_C - Bomb Escape Shaping

S2_C additionally rewards safe behaviour around bombs.

Custom events:

- `ESCAPED_DANGER`: +1.0
- `ENTERED_DANGER`: -1.0
- `STAYED_IN_DANGER`: -0.2

Danger information is based on the Stage 2 bomb and explosion representation
used by the shared `team_agent`.

The purpose is to test whether explicit escape-related shaping reduces
self-kills while preserving effective bomb usage.

---

### Controlled Stage 2 Comparison

The planned comparison is:

`S2_A vs S2_B vs S2_C`

The following should remain constant between reward configurations:

- Stage 2 feature representation
- linear Q-model architecture
- training scenario
- training duration
- learning rate
- discount factor
- epsilon schedule
- evaluation procedure

Each reward configuration should be trained from a fresh model.

Where possible, training should be repeated using controlled or multiple
training seeds.

---

### Task 2 Evaluation

Task 2 evaluation is implemented in:

`evaluation/task2/`

Run an evaluation from the repository root with:

```bash
python evaluation/task2/run_task2_evaluation.py --agent AGENT_NAME
```

### Metrics

The following metrics are recorded:

### Performance
-average score per round
-coins collected per round
-crates destroyed per round

### Safety
-self-kills per round

### Bomb Behaviour
-bombs dropped per round
-crates destroyed per bomb

### Diagnostics
-invalid actions per round
-steps per round

crates_per_bomb is included as an additional diagnostic measure of bomb
efficiency. A higher value indicates that fewer bombs are required to
destroy the same number of crates.

### Comparing Agents

```bash
python evaluation/task2/compare_task2_agents.py --agents AGENT_1 AGENT_2 AGENT_3
```

The comparison is saved to: results/task2/task2_agent_comparison.csv

Plots can then be generated with:

```bash
python evaluation/task2/plot_task2_comparison.py
```

The plotting script currently produces comparisons for crates destroyed per round, self-kills per round and crates destroyed per bomb