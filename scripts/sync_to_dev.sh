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

ssh "$REMOTE_ALIAS" "set -euo pipefail; mkdir -p '$REMOTE_PROJECT'; test \"\${PWD:-}\" != '$REMOTE_PROJECT'"

rsync -az --delete \
  --exclude ".git/" \
  --exclude ".env" \
  --exclude "autokaggle_config.json" \
  --exclude "api_key.txt" \
  --exclude ".kaggle/" \
  --exclude ".venv/" \
  --exclude ".pycache/" \
  --exclude "__pycache__/" \
  --exclude "multi_agents/competition/*/*.zip" \
  --exclude "multi_agents/competition/*/*.csv" \
  --exclude "multi_agents/competition/*/*.parquet" \
  --exclude "multi_agents/competition/*/*.jsonl" \
  --exclude "multi_agents/competition/*/overview.txt" \
  --exclude "multi_agents/competition/*/competition_intake.json" \
  --exclude "multi_agents/competition/*/runs/" \
  --exclude "multi_agents/competition/*/experiments/" \
  --exclude "multi_agents/competition/*/baseline_review.json" \
  --exclude "multi_agents/competition/*/best_score.json" \
  --exclude "multi_agents/competition/*/brain_review.json" \
  --exclude "multi_agents/competition/*/champion_selection.json" \
  --exclude "multi_agents/competition/*/champion_submission.csv" \
  --exclude "multi_agents/competition/*/data_manifest.json" \
  --exclude "multi_agents/competition/*/enhancement_review.json" \
  --exclude "multi_agents/competition/*/experiment_plan.json" \
  --exclude "multi_agents/competition/*/kaggle_env_preflight.json" \
  --exclude "multi_agents/competition/*/kaggle_submit_plan.json" \
  --exclude "multi_agents/competition/*/kaggle_submit_result.json" \
  --exclude "multi_agents/competition/*/leaderboard_feedback.json" \
  --exclude "multi_agents/competition/*/leaderboard_gap_audit.json" \
  --exclude "multi_agents/competition/*/llm_experiment_plan.json" \
  --exclude "multi_agents/competition/*/llm_experiment_plan.md" \
  --exclude "multi_agents/competition/*/metric_spec.json" \
  --exclude "multi_agents/competition/*/optimization_loop_summary.json" \
  --exclude "multi_agents/competition/*/remote_brain_reply.md" \
  --exclude "multi_agents/competition/*/stability_first_review.json" \
  --exclude "multi_agents/competition/*/submission_gate.json" \
  --exclude "multi_agents/competition/*/task_card.md" \
  --exclude ".aris/upstream/.git/" \
  ./ "$REMOTE_ALIAS:$REMOTE_PROJECT/"

ssh "$REMOTE_ALIAS" "set -euo pipefail; cd '$REMOTE_WS'; test -d '$REMOTE_PROJECT'; printf '%s\n' '$REMOTE_PROJECT'"
