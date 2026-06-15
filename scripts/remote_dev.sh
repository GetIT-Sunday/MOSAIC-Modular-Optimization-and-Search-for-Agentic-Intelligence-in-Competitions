#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="${AUTOKAGGLE_CONFIG:-autokaggle_config.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REMOTE_ALIAS="$("$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("'"$CONFIG_JSON"'"); d=json.loads(p.read_text()) if p.exists() else {}; print(((d.get("remote") or {}).get("ssh_alias")) or "dev")')"
REMOTE_WS="$("$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("'"$CONFIG_JSON"'"); d=json.loads(p.read_text()) if p.exists() else {}; print(((d.get("remote") or {}).get("workspace")) or "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac")')"
CONDA_ENV_NAME="$("$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("'"$CONFIG_JSON"'"); d=json.loads(p.read_text()) if p.exists() else {}; print(((d.get("remote") or {}).get("conda_env")) or "mac")')"

if [[ $# -eq 0 ]]; then
  echo "usage: scripts/remote_dev.sh '<command to run under the fixed remote workspace>'" >&2
  exit 2
fi

REMOTE_COMMAND="$*"

ssh "$REMOTE_ALIAS" "set -euo pipefail
REMOTE_WS='$REMOTE_WS'
cd \"\$REMOTE_WS\"
export CONDA_ENVS_PATH=\"\$REMOTE_WS/.conda/envs\"
export CONDA_PKGS_DIRS=\"\$REMOTE_WS/.conda/pkgs\"
export XDG_CACHE_HOME=\"\$REMOTE_WS/.cache\"
export PIP_CACHE_DIR=\"\$REMOTE_WS/.cache/pip\"
export HF_HOME=\"\$REMOTE_WS/.cache/huggingface\"
export TRANSFORMERS_CACHE=\"\$REMOTE_WS/.cache/huggingface/transformers\"
export KAGGLE_CONFIG_DIR="\$REMOTE_WS/.kaggle"
export HOME="\$REMOTE_WS"
export AUTOKAGGLE_REMOTE_WORKSPACE=1
conda run -n '$CONDA_ENV_NAME' bash -c '$REMOTE_COMMAND'"
