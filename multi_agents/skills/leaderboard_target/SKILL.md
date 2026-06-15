---
name: leaderboard_target
description: Use when AutoKaggle needs to compare local experiments against Kaggle leaderboard objectives, estimate medal targets, or decide whether a baseline is actually competitive.
---

# Leaderboard Target

Use this skill before Brain treats a local baseline as "good". A score is only
meaningful relative to the competition metric, leaderboard distribution, medal
line, and public/private risk.

## Required Context

Read these artifacts when available:

- `data_manifest.json`
- `metric_spec.json`
- `baseline_review.json`
- `leaderboard_target.json`
- `leaderboard_feedback.json`
- `leaderboard_gap_audit.json`
- Kaggle leaderboard CSV or CLI output

## Workflow

1. Identify the official metric and whether higher is better.
2. Load the current best local score from `baseline_review.json`,
   `champion_selection.json`, or latest validation reports.
3. Fetch or ingest leaderboard signals:
   - top score
   - top 10 score
   - top 5% score
   - rank distribution if available
   - current submitted public score/rank if available
4. Estimate medal targets conservatively.
   - If official medal cutoffs are not available, use top percentile proxies.
   - For active competitions, treat target estimates as moving targets.
5. Compute gaps:
   - local-to-top gap
   - local-to-silver-proxy gap
   - public-to-local gap when feedback exists
6. Decide the optimization objective:
   - `baseline_validation`: no leaderboard yet, validate and package first.
   - `gap_closing`: leaderboard target is known and local gap is material.
   - `stability_first`: public score is materially worse than local CV.
   - `submission_ready`: local and public evidence are strong enough for a
     controlled submission workflow.

## Risk Rules

- Do not let a high local score override leaderboard distance.
- Do not chase public leaderboard if CV is unstable or leakage risk is high.
- Do not estimate silver line from a tiny visible leaderboard without marking
  confidence low.
- For active competitions, report targets as snapshots with timestamps.

## Output Contract

Every Brain plan using this skill should include:

```json
{
  "skill_used": "leaderboard_target",
  "leaderboard_snapshot_available": true,
  "metric_name": "accuracy",
  "higher_is_better": true,
  "top_score": 0.98,
  "top_10_score": 0.97,
  "estimated_silver_score": 0.95,
  "current_best_local_score": 0.927,
  "gap_to_top": 0.053,
  "gap_to_silver": 0.023,
  "target_policy": "silver_or_better",
  "confidence": "medium",
  "next_decision": "gap_closing"
}
```

## Harness Selection

- Use `leaderboard_snapshot_harness` to fetch and parse leaderboard snapshots.
- Use `leaderboard_gap_harness` after public score feedback exists.
- Use `submission_feedback_harness` after a human upload or Kaggle submit.
