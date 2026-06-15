---
name: validation_risk
description: Use when auditing whether local validation can be trusted, especially before leaderboard chasing, model promotion, or manual submission.
---

# Validation Risk

Use this skill whenever a score looks high, surprising, unstable, or
leaderboard feedback disagrees with local validation.

## Required Context

Read:

- `data_manifest.json`
- `baseline_review.json`
- validation reports
- OOF predictions when available
- `leaderboard_feedback.json`
- `leaderboard_gap_audit.json`
- drift/leakage audit artifacts

## Workflow

1. Confirm metric correctness:
   - official metric vs inferred metric
   - prediction format
   - higher/lower is better
2. Confirm split design:
   - stratified K-fold for classification
   - group split when entity leakage is plausible
   - time split when time ordering exists
3. Measure stability:
   - fold mean and std
   - seed mean and std when stochastic
   - OOF confusion matrix
   - per-class recall and precision
4. Check leakage:
   - ID-like columns
   - target-derived features
   - duplicate rows across train/test
   - train/test distribution mismatch
5. Compare leaderboard feedback:
   - local vs public gap
   - public rank vs target rank
   - repeated submissions risk
6. Decide next action:
   - `continue_optimization`
   - `run_stability_audit`
   - `run_drift_audit`
   - `pause_submission`

## Risk Levels

- `low`: fold std small, no obvious leakage, public feedback consistent.
- `medium`: one risk signal exists but score still plausible.
- `high`: public score materially worse than local CV, fold instability is high,
  or leakage/drift is likely.

## Output Contract

```json
{
  "skill_used": "validation_risk",
  "harness": "stratified_cv_stability_harness",
  "hypothesis": "The 0.927 holdout score may be optimistic; 5-fold OOF will test stability and class-level failure modes.",
  "runner_kind": "cv_stability_audit",
  "expected_gain": "risk_reduction",
  "risk": "single holdout may not represent leaderboard split",
  "validation_plan": "5-fold stratified CV, fold std, OOF confusion matrix, train/test drift summary",
  "promotion_gate": {
    "max_fold_std": 0.01,
    "validator_must_pass": true,
    "manual_submit_allowed": false
  }
}
```

## Harness Selection

- Use `stratified_cv_stability_harness` for first post-baseline audit.
- Use `distribution_shift_harness` when train/test drift is suspected.
- Use `overfitting_audit_harness` when local score greatly exceeds public score.
