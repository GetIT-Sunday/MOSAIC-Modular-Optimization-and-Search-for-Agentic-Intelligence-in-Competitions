#!/usr/bin/env bash
set -euo pipefail

REMOTE_ALIAS="dev"
REMOTE_WS="/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac"
CONDA_ENV_NAME="mac"

case "$REMOTE_WS" in
  /home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac) ;;
  *)
    echo "Refusing to use unexpected remote workspace: $REMOTE_WS" >&2
    exit 3
    ;;
esac

ssh "$REMOTE_ALIAS" "set -euo pipefail
REMOTE_WS='$REMOTE_WS'
cd \"\$REMOTE_WS\"
mkdir -p \"\$REMOTE_WS/.cache/pip\" \"\$REMOTE_WS/.conda/envs\" \"\$REMOTE_WS/.conda/pkgs\"
export CONDA_ENVS_PATH=\"\$REMOTE_WS/.conda/envs\"
export CONDA_PKGS_DIRS=\"\$REMOTE_WS/.conda/pkgs\"
export XDG_CACHE_HOME=\"\$REMOTE_WS/.cache\"
export PIP_CACHE_DIR=\"\$REMOTE_WS/.cache/pip\"
export HF_HOME=\"\$REMOTE_WS/.cache/huggingface\"
export TRANSFORMERS_CACHE=\"\$REMOTE_WS/.cache/huggingface/transformers\"
export AUTOKAGGLE_REMOTE_WORKSPACE=1
conda run -n '$CONDA_ENV_NAME' python -m pip install --cache-dir \"\$PIP_CACHE_DIR\" pytest kaggle
conda run -n '$CONDA_ENV_NAME' python - <<'PY'
import importlib.util
import shutil
import sys

missing = []
if importlib.util.find_spec('pytest') is None:
    missing.append('pytest')
if shutil.which('kaggle') is None:
    missing.append('kaggle-cli')
if missing:
    raise SystemExit('Missing after install: ' + ', '.join(missing))
print('python=' + sys.executable)
print('pytest=available')
print('kaggle=' + str(shutil.which('kaggle')))
PY"
