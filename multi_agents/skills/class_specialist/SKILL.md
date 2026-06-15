---
name: class_specialist
description: Use when per-class OOF analysis shows a weak class and AutoKaggle should train one-vs-rest, hard-example, or class-wise correction models.
---

# Class Specialist

Use this skill when the overall score is plateauing but per-class metrics show a
specific weak class. The specialist should change a named control variable,
such as class weight, one-vs-rest training, hard-example focus, or class-wise
prediction override.

## Required Context

Read:

- `per_class_oof_report.json`
- champion `validation_report.json`
- champion `oof_predictions.csv`
- `data_manifest.json`
- `oof_diversity_report.json`

## Workflow

1. Select the lowest-recall class with enough support.
2. Train a specialist binary model for that class.
3. Combine it with a stable multiclass base model using OOF-only thresholds.
4. Report recall, precision, coverage, and overall score.
5. Promote only if the weak class improves without unacceptable overall loss.

## Output Contract

```json
{
  "skill_used": "class_specialist",
  "harness": "class_specialist_oof_harness",
  "runner_kind": "star_specialist_lgbm",
  "hypothesis": "A STAR-vs-rest specialist may recover missed STAR examples while preserving overall accuracy.",
  "validation_plan": "5-fold OOF, compare STAR recall and overall accuracy against champion",
  "evidence_needed": ["specialist_report.json", "per_class_oof_report.json", "oof_predictions.csv"],
  "promotion_gate": {
    "validator_must_pass": true,
    "target_class_recall": ">=0.920",
    "min_local_score": 0.964,
    "manual_submit_allowed": false
  }
}
```
