# Model 1

The deployed model is `agent_code/team_agent/team-agent-model.pt`. It uses the frozen 500-round continuation weights, 57 features, legal and survival masks, placement margin 1, and the ongoing own-bomb escape buffer. These defaults are stored in the checkpoint. No extra training is needed to use it.

## Files

| File | Purpose |
|---|---|
| `watch_submission_candidate.py` | Verify the deployed checkpoint and watch games against three rule-based agents |
| `run_improvement_evaluation.py` | Evaluate checkpoints on fixed starting seeds; save statistics and optional replays |
| `view_diagnostic_replay.py` | Watch a saved evaluation round |
| `test_action_mask.py` | Check legality, countdown, survival filtering, feature padding, and history handling |

Model identity and settings are recorded in `agent_code/team_agent/MODEL_INFO.json`. The frozen training checkpoint is `checkpoints/model1_extra_seed31/model.pt`; the original reference checkpoint is `checkpoints/before_improvements/team-agent-model.pt`. Paths passed to `--models` are relative to `agent_code/team_agent`. The deployment file has the same learned weights as the frozen training checkpoint; its hash differs because the selected inference settings are stored in metadata.

The original task1/task2/task3 evaluation folders are unchanged. 

## Watch

Run from the project root:

```bat
.\.venv\Scripts\python.exe evaluation/model1/watch_submission_candidate.py --rounds 10
```

## Evaluate

```bat
.\.venv\Scripts\python.exe evaluation/model1/run_improvement_evaluation.py --models team-agent-model.pt --masks survival --bomb-margin 1 --persistent-escape-margin --rounds 200 --seed-start 240001 --save-replays --output results/model1/evaluation_200
```

