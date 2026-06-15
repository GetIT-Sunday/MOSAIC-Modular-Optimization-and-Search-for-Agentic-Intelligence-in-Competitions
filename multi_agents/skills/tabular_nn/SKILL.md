---
name: tabular_nn
description: Use when AutoKaggle needs a neural-network style tabular model such as MLP, residual MLP, categorical embeddings, or transformer-like tabular experiments.
---

# Tabular NN

Use this skill after a tabular GBDT baseline is trustworthy and the next
experiment needs a different inductive bias. The goal is not to replace GBDT by
default; it is to create OOF diversity or capture smooth continuous
interactions that tree splits may miss.

## Required Context

Read:

- `data_manifest.json`
- `metric_spec.json`
- latest GBDT `validation_report.json`
- `per_class_oof_report.json`
- `oof_diversity_report.json`
- feature list and `feature_importance.csv`

## Workflow

1. Confirm the task is tabular classification or regression.
2. Build a numeric/categorical preprocessing pipeline.
3. Prefer PyTorch when available; otherwise use sklearn MLP fallback.
4. Use stratified OOF validation for classification.
5. Write OOF predictions and a validated submission.
6. Judge value by score and OOF diversity, not score alone.

## Output Contract

```json
{
  "skill_used": "tabular_nn",
  "harness": "tabular_nn_oof_harness",
  "runner_kind": "tabular_mlp",
  "hypothesis": "An MLP may capture smooth spectral/color interactions differently from GBDT and add ensemble diversity.",
  "validation_plan": "5-fold stratified OOF validation, train-valid gap, OOF disagreement against champion",
  "evidence_needed": ["nn_training_report.json", "model_config.json", "oof_predictions.csv", "validator_result.json"],
  "promotion_gate": {
    "validator_must_pass": true,
    "min_local_score": 0.962,
    "manual_submit_allowed": false
  }
}
```
