---
name: clean_ensemble
description: Use when AutoKaggle has multiple OOF candidates and must filter invalid predictions before blending, stacking, rank averaging, or class-wise blending.
---

# Clean Ensemble

Use this skill after diversity or per-class audits show multiple candidates.
The first job is evidence hygiene: remove invalid OOF candidates before any
blend search. A blend based on empty, misaligned, or label-incompatible OOF is
not evidence.

## Required Context

Read:

- all candidate `validation_report.json`
- all candidate `oof_predictions.csv`
- `oof_diversity_report.json`
- `per_class_oof_report.json`
- `validator_result.json`

## Workflow

1. Filter candidates with missing OOF, empty predictions, label mismatch, row
   mismatch, or failed validator.
2. Rank valid candidates by local score and diversity.
3. Build a constrained blend from a small candidate set.
4. Report skipped candidates and why they were skipped.
5. Promote only when OOF score and validator pass.

## Output Contract

```json
{
  "skill_used": "clean_ensemble",
  "harness": "clean_oof_blend_harness",
  "runner_kind": "clean_oof_blend",
  "hypothesis": "A clean blend of verified LightGBM and diverse XGBoost candidates may improve CV without invalid OOF artifacts.",
  "validation_plan": "Filter invalid OOF, compute blend OOF score, compare per-class metrics",
  "evidence_needed": ["clean_blend_report.json", "skipped_candidates.json", "oof_diversity_report.json"],
  "promotion_gate": {
    "validator_must_pass": true,
    "min_local_score": 0.965,
    "manual_submit_allowed": false
  }
}
```
