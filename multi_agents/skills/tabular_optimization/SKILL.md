---
name: tabular_optimization
description: Use when planning tabular Kaggle model improvements, feature engineering, GBDT/NN experiments, or score-gain hypotheses after baseline.
---

# Tabular Optimization

Use this skill after intake and baseline are valid. The goal is to generate
rank-oriented hypotheses, not a list of models.

## Required Context

Read:

- `data_manifest.json`
- `metric_spec.json`
- `baseline_review.json`
- `leaderboard_target.json` when available
- latest validation reports and validator results
- drift/leakage/stability audits when available

## Decision Workflow

1. Characterize the table:
   - row count, feature count, target type
   - categorical vs numeric columns
   - missingness and high-cardinality features
   - class imbalance
2. Interpret the baseline:
   - linear/logistic high score means strong separability
   - low score means missing nonlinear interactions or preprocessing issues
   - unstable fold scores mean validation must be fixed first
3. Generate hypotheses by model family:
   - GBDT for nonlinear interactions and mixed feature types
   - CatBoost for categorical interactions and low-friction categorical handling
   - LightGBM/XGBoost for fast boosted-tree search and feature importance
   - ExtraTrees/RandomForest for robust nonparametric sanity checks
   - residual MLP or FT-Transformer only when GBDT plateau is plausible or
     continuous interactions look important
   - TabPFN only when data size and feature shape fit its limits
4. Select the first experiment that maximizes expected information value, not
   just expected score.
5. Require promotion gates before any submission package.

## Feature Hypothesis Patterns

Prefer hypotheses that explain why a feature change may work:

- interaction: color-band ratios, differences, group-wise contrasts
- categorical handling: target encoding, CatBoost native categories
- calibration: probability calibration if metric uses probability quality
- imbalance: class weights or balanced sampling
- domain transform: physically meaningful transforms when column names support it
- pruning: remove drifted or leakage-suspicious columns

## Model Architecture Guidance

- Start with GBDT before neural networks for medium/large tabular data.
- Use residual MLP when continuous numeric interactions dominate and GBDT gains
  plateau.
- Use categorical embeddings when categorical variables are meaningful and not
  too high-cardinality.
- Blend neural and GBDT models only if OOF predictions are not too correlated.

## Output Contract

Every recommendation should include:

```json
{
  "skill_used": "tabular_optimization",
  "harness": "stratified_gbdt_oof_harness",
  "hypothesis": "CatBoost may capture spectral_type x redshift interactions missed by the linear baseline.",
  "model_family": "catboost",
  "runner_kind": "catboost",
  "expected_gain": "+0.002 to +0.006 accuracy",
  "risk": "overfit to categorical proxy or unstable public/private split",
  "validation_plan": "5-fold stratified OOF validation with confusion matrix",
  "promotion_gate": {
    "min_cv_gain": 0.002,
    "validator_must_pass": true,
    "max_fold_std_increase": 0.001
  }
}
```
