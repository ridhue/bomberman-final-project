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