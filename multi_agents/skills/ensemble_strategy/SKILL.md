---
name: ensemble_strategy
description: Use when multiple validated models exist and AutoKaggle needs to plan OOF blending, stacking, rank averaging, or submission promotion.
---

# Ensemble Strategy

Use this skill after at least two validated candidates exist. An ensemble should
be justified by complementary errors, not by leaderboard superstition.

## Required Context

Read:

- `champion_selection.json`
- validation reports
- OOF prediction files
- submission files
- `leaderboard_feedback.json`
- `leaderboard_gap_audit.json`

## Workflow

1. Collect candidates:
   - local score
   - fold scores
   - OOF predictions
   - submission path
   - risk level
2. Measure complementarity:
   - pairwise prediction correlation
   - disagreement rate
   - per-class error overlap
3. Choose blend type:
   - mean probability blend for calibrated classifiers
   - rank averaging for score-scale mismatch
   - constrained weighted blend for 2-5 strong candidates
   - stacking only when OOF data is clean and enough rows exist
4. Apply regularization:
   - limit candidate count
   - avoid extreme weights
   - require no fold instability increase
5. Promote only if OOF and validator pass.

## Do Not

- Do not blend models trained on incompatible folds unless OOF alignment is
  documented.
- Do not tune weights only against public leaderboard without Human Gate.
- Do not promote an ensemble that improves mean CV but sharply increases fold
  variance.

## Output Contract

```json
{
  "skill_used": "ensemble_strategy",
  "harness": "regularized_oof_blend_harness",
  "hypothesis": "LightGBM and logistic errors may be complementary because one captures nonlinear interactions and the other preserves a stable linear boundary.",
  "runner_kind": "regularized_blend",
  "expected_gain": "+0.001 to +0.004 accuracy",
  "risk": "blend may overfit OOF folds or amplify public/private gap",
  "validation_plan": "OOF blend with pairwise correlation and fold-level comparison",
  "promotion_gate": {
    "min_cv_gain": 0.001,
    "max_model_correlation": "<= 0.995",
    "validator_must_pass": true,
    "manual_submit_allowed": false
  }
}
```
