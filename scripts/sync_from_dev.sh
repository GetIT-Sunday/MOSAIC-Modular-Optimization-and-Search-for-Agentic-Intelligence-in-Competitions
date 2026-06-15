#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="${AUTOKAGGLE_CONFIG:-autokaggle_config.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REMOTE_ALIAS="$("$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("'"$CONFIG_JSON"'"); d=json.loads(p.read_text()) if p.exists() else {}; print(((d.get("remote") or {}).get("ssh_alias")) or "dev")')"
REMOTE_WS="$("$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("'"$CONFIG_JSON"'"); d=json.loads(p.read_text()) if p.exists() else {}; print(((d.get("remote") or {}).get("workspace")) or "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac")')"
REMOTE_PROJECT_SUBDIR="$("$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("'"$CONFIG_JSON"'"); d=json.loads(p.read_text()) if p.exists() else {}; print(((d.get("remote") or {}).get("project_subdir")) or "workspaces/AutoKaggle")')"
REMOTE_PROJECT="$REMOTE_WS/$REMOTE_PROJECT_SUBDIR"

case "$REMOTE_PROJECT" in
  "$REMOTE_WS"/*) ;;
  *)
    echo "Refusing to sync outside hard remote workspace: $REMOTE_PROJECT" >&2
    exit 3
    ;;
esac

MODE="full"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: scripts/sync_from_dev.sh <competition> [--lite]" >&2
  exit 2
fi

COMPETITION="$1"
if [[ $# -eq 2 ]]; then
  case "$2" in
    --lite) MODE="lite" ;;
    *)
      echo "Unknown sync mode: $2" >&2
      exit 2
      ;;
  esac
fi
case "$COMPETITION" in
  *[!A-Za-z0-9_-]*|"")
    echo "Invalid competition name: $COMPETITION" >&2
    exit 3
    ;;
esac

LOCAL_COMPETITION_DIR="multi_agents/competition/$COMPETITION"
REMOTE_COMPETITION_DIR="$REMOTE_PROJECT/multi_agents/competition/$COMPETITION"

mkdir -p "$LOCAL_COMPETITION_DIR"

ssh "$REMOTE_ALIAS" "set -euo pipefail; test -d '$REMOTE_COMPETITION_DIR'; case '$REMOTE_COMPETITION_DIR' in '$REMOTE_WS'/*) ;; *) exit 4 ;; esac"

COMMON_ITEMS=(
  "baseline_review.json" \
  "best_score.json" \
  "brain_review.json" \
  "champion_selection.json" \
  "champion_comparison.json" \
  "champion_submission.csv" \
  "competition_intake.json" \
  "data_manifest.json" \
  "experiment_plan.json" \
  "enhancement_review.json" \
  "experiment_queue.json" \
  "experiment_queue.md" \
  "experiment_roadmap.json" \
  "experiment_roadmap.md" \
  "llm_experiment_plan.json" \
  "llm_experiment_plan.md" \
  "kaggle_env_preflight.json" \
  "kaggle_submit_plan.json" \
  "kaggle_submit_result.json" \
  "leaderboard_feedback.json" \
  "leaderboard_feedback_loop.json" \
  "leaderboard_gap_audit.json" \
  "leaderboard_feedback_template_fill.json" \
  "leaderboard_feedback_input_template.json" \
  "leaderboard_feedback_input_validation.json" \
  "leaderboard_target.json" \
  "leaderboard_target_raw.csv" \
  "manual_submission_package/README.md" \
  "manual_submission_package/leaderboard_feedback_input_template.json" \
  "manual_submission_package/manifest.json" \
  "manual_submission_package/submission.csv" \
  "manual_submission_package_verification.json" \
  "manual_submit_readiness.json" \
  "metric_spec.json" \
  "optimization_loop_summary.json" \
  "post_submit_workflow.json" \
  "post_submit_workflow.md" \
  "post_experiment_pipeline.json" \
  "post_reselection_gate.json" \
  "promotion_gate_review.json" \
  "promotion_gate_review.md" \
  "promoted_submission.csv" \
  "recommended_submission.csv" \
  "remote_brain_reply.md" \
  "stability_first_review.json" \
  "submission_gate.json" \
  "submission_policy.json" \
  "submission_decision_review.json" \
  "submission_decision_review.md" \
  "submit_decision_handoff.json" \
  "submit_decision_handoff.md" \
  "tabular_feature_leakage_audit.json" \
  "task_card.md"
)

FULL_ITEMS=(
  "runs" \
  "experiments"
)

LITE_ITEMS=(
  "experiments/stability_first_features_v1/feature_report.json" \
  "experiments/stability_first_search_v1/risk_audit.json" \
  "experiments/stability_first_search_v1/validation_report.json" \
  "experiments/tabular_feature_leakage_audit_v1/leakage_report.json" \
  "experiments/leakage_safe_search_v1/risk_audit.json" \
  "experiments/leakage_safe_search_v1/validation_report.json" \
  "experiments/safe_engineered_features_v1/run.log" \
  "experiments/safe_engineered_features_v1/validation_report.json" \
  "experiments/safe_engineered_features_v1/validator_result.json" \
  "experiments/cv_stability_audit_v1/cv_stability_audit.json" \
  "experiments/cv_stability_audit_v1/run.log" \
  "experiments/cv_stability_audit_v1/validation_report.json" \
  "experiments/cv_stability_audit_v1/validator_result.json" \
  "experiments/post_pause_cv_stability_audit_v2/cv_stability_audit.json" \
  "experiments/post_pause_cv_stability_audit_v2/run.log" \
  "experiments/post_pause_cv_stability_audit_v2/validation_report.json" \
  "experiments/post_pause_cv_stability_audit_v2/validator_result.json" \
  "experiments/post_pause_regularized_blend_v1/oof_predictions.csv" \
  "experiments/post_pause_regularized_blend_v1/regularized_blend_report.json" \
  "experiments/post_pause_regularized_blend_v1/run.log" \
  "experiments/post_pause_regularized_blend_v1/validation_report.json" \
  "experiments/post_pause_regularized_blend_v1/validator_result.json" \
  "runs/index.html" \
  "runs/ledger.jsonl"
)

ITEMS=("${COMMON_ITEMS[@]}")
if [[ "$MODE" == "lite" ]]; then
  ITEMS+=("${LITE_ITEMS[@]}")
else
  ITEMS+=("${FULL_ITEMS[@]}")
fi

EXISTING_ITEMS="$(
  printf '%s\n' "${ITEMS[@]}" | ssh "$REMOTE_ALIAS" "set -euo pipefail
REMOTE_COMPETITION_DIR='$REMOTE_COMPETITION_DIR'
while IFS= read -r item; do
  if [[ -e \"\$REMOTE_COMPETITION_DIR/\$item\" ]]; then
    printf '%s\n' \"\$item\"
  fi
done"
)"

if [[ "$MODE" == "lite" ]]; then
  DYNAMIC_LITE_ITEMS="$(
    ssh "$REMOTE_ALIAS" "set -euo pipefail
REMOTE_COMPETITION_DIR='$REMOTE_COMPETITION_DIR'
cd \"\$REMOTE_COMPETITION_DIR\"
if [[ -d experiments ]]; then
  find experiments -type f \( \
    -name 'run.py' -o \
    -name 'run.log' -o \
    -name 'oof_predictions.csv' -o \
    -name 'submission.csv' -o \
    -name 'validation_report.json' -o \
    -name 'validator_result.json' -o \
    -name '*_report.json' -o \
    -name '*_audit.json' \
  \) -print
fi"
  )"
  EXISTING_ITEMS="$(
    {
      printf '%s\n' "$EXISTING_ITEMS"
      printf '%s\n' "$DYNAMIC_LITE_ITEMS"
    } | awk 'NF && !seen[$0]++'
  )"
fi

while IFS= read -r item; do
  [[ -n "$item" ]] || continue
  if [[ "$item" == */* ]]; then
    mkdir -p "$LOCAL_COMPETITION_DIR/${item%/*}"
    rsync -az --delete "$REMOTE_ALIAS:$REMOTE_COMPETITION_DIR/$item" "$LOCAL_COMPETITION_DIR/$item"
  else
    rsync -az --delete "$REMOTE_ALIAS:$REMOTE_COMPETITION_DIR/$item" "$LOCAL_COMPETITION_DIR/"
  fi
done <<< "$EXISTING_ITEMS"

printf '%s (%s)\n' "$LOCAL_COMPETITION_DIR" "$MODE"
