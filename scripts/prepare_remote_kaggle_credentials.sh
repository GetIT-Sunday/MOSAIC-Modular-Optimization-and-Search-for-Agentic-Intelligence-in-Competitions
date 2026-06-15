#!/usr/bin/env bash
set -euo pipefail

REMOTE_ALIAS="dev"
REMOTE_WS="/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac"
REMOTE_PROJECT="$REMOTE_WS/workspaces/AutoKaggle"

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/prepare_remote_kaggle_credentials.sh <competition>" >&2
  exit 2
fi

COMPETITION="$1"
case "$COMPETITION" in
  *[!A-Za-z0-9_-]*|"")
    echo "Invalid competition name: $COMPETITION" >&2
    exit 3
    ;;
esac

REMOTE_COMPETITION_DIR="$REMOTE_PROJECT/multi_agents/competition/$COMPETITION"
REMOTE_KAGGLE_DIR="$REMOTE_COMPETITION_DIR/.kaggle"
REMOTE_KAGGLE_JSON="$REMOTE_KAGGLE_DIR/kaggle.json"

ssh "$REMOTE_ALIAS" "set -euo pipefail
REMOTE_WS='$REMOTE_WS'
REMOTE_COMPETITION_DIR='$REMOTE_COMPETITION_DIR'
REMOTE_KAGGLE_DIR='$REMOTE_KAGGLE_DIR'
REMOTE_KAGGLE_JSON='$REMOTE_KAGGLE_JSON'
case \"\$REMOTE_COMPETITION_DIR\" in
  \"\$REMOTE_WS\"/*) ;;
  *) echo 'Refusing credential path outside hard remote workspace.' >&2; exit 4 ;;
esac
test -d \"\$REMOTE_COMPETITION_DIR\"
mkdir -p \"\$REMOTE_KAGGLE_DIR\"
chmod 700 \"\$REMOTE_KAGGLE_DIR\"
if [[ -f \"\$REMOTE_KAGGLE_JSON\" ]]; then
  chmod 600 \"\$REMOTE_KAGGLE_JSON\"
  python - <<'PY'
import json
from pathlib import Path

path = Path('$REMOTE_KAGGLE_JSON')
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    raise SystemExit(f'kaggle.json exists but is not valid JSON: {exc}')
missing = [key for key in ('username', 'key') if not payload.get(key)]
if missing:
    raise SystemExit('kaggle.json is missing required fields: ' + ', '.join(missing))
print('credential_file=present')
print('credential_json=valid')
print('secret_values=not_printed')
PY
else
  printf '%s\n' 'credential_file=missing'
fi
printf 'credential_dir=%s\n' \"\$REMOTE_KAGGLE_DIR\"
printf 'credential_file=%s\n' \"\$REMOTE_KAGGLE_JSON\"
printf '%s\n' 'Paste Kaggle API JSON into that file on the remote host, then rerun this script.'
"
