from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .project_config import ProjectConfigAgent


Runner = Callable[[List[str]], subprocess.CompletedProcess]


@dataclass(frozen=True)
class RemoteHealthCheck:
    key: str
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class RemoteHealthResult:
    status: str
    report_path: Path
    checks: List[RemoteHealthCheck]


class RemoteHealthCheckAgent:
    """Read-only remote execution health diagnostics for open-source users."""

    def __init__(self, project_root: Path, runner: Optional[Runner] = None):
        self.project_root = project_root.resolve()
        self.runner = runner or subprocess.run
        self.config = ProjectConfigAgent(self.project_root).load()

    def run(self, output_path: Optional[Path] = None) -> RemoteHealthResult:
        output_path = output_path or (
            self.project_root / "multi_agents" / "competition" / "console" / "remote_health_check.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checks: List[RemoteHealthCheck] = []
        checks.extend(self._run_basic_remote_probe())
        checks.append(self._run_conda_probe())
        overall = self._overall_status(checks)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": overall,
            "checks": [check.__dict__ for check in checks],
            "safe_remote": self._safe_remote_config(),
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return RemoteHealthResult(status=overall, report_path=output_path, checks=checks)

    def _run_basic_remote_probe(self) -> List[RemoteHealthCheck]:
        remote = self.config.get("remote") or {}
        ssh_alias = str(remote.get("ssh_alias") or "dev")
        workspace = str(remote.get("workspace") or "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac")
        project_subdir = str(remote.get("project_subdir") or "workspaces/AutoKaggle")
        project_path = f"{workspace.rstrip('/')}/{project_subdir.lstrip('/')}"
        script = f"""
set +e
REMOTE_WS={shlex.quote(workspace)}
REMOTE_PROJECT={shlex.quote(project_path)}
echo AUTOKAGGLE_HEALTH_BEGIN
if [ -d "$REMOTE_WS" ]; then echo workspace=pass:"$REMOTE_WS"; else echo workspace=fail:"$REMOTE_WS missing"; fi
case "$REMOTE_PROJECT" in "$REMOTE_WS"/*) echo boundary=pass:"$REMOTE_PROJECT" ;; *) echo boundary=fail:"$REMOTE_PROJECT outside workspace" ;; esac
if [ -d "$REMOTE_PROJECT" ]; then echo project=pass:"$REMOTE_PROJECT"; else echo project=warn:"$REMOTE_PROJECT missing"; fi
if command -v conda >/dev/null 2>&1; then echo conda_cli=pass:$(command -v conda); else echo conda_cli=fail:conda not found; fi
DISK_LINE=$(df -Pk "$REMOTE_WS" 2>/dev/null | awk 'NR==2 {{print $4 "KB_available"}}')
if [ -n "$DISK_LINE" ]; then echo disk=pass:"$DISK_LINE"; else echo disk=warn:df unavailable; fi
if command -v nvidia-smi >/dev/null 2>&1; then GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1); echo gpu=pass:"${{GPU_LINE:-nvidia-smi present}}"; else echo gpu=warn:nvidia-smi not found; fi
if [ -d "$REMOTE_WS/.kaggle" ] || [ -f "$REMOTE_WS/.kaggle/kaggle.json" ]; then echo kaggle_config=pass:"$REMOTE_WS/.kaggle"; else echo kaggle_config=warn:"$REMOTE_WS/.kaggle missing"; fi
echo AUTOKAGGLE_HEALTH_END
"""
        completed = self._run(["ssh", ssh_alias, script])
        if completed.returncode != 0:
            return [
                RemoteHealthCheck(
                    "ssh",
                    "SSH 连接",
                    "fail",
                    self._sanitize_output(completed.stderr or completed.stdout or f"returncode={completed.returncode}"),
                )
            ]
        checks = [RemoteHealthCheck("ssh", "SSH 连接", "pass", ssh_alias)]
        checks.extend(self._parse_probe_output(completed.stdout))
        return checks

    def _run_conda_probe(self) -> RemoteHealthCheck:
        remote = self.config.get("remote") or {}
        ssh_alias = str(remote.get("ssh_alias") or "dev")
        workspace = str(remote.get("workspace") or "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac")
        conda_env = str(remote.get("conda_env") or "mac")
        command = (
            f"set -euo pipefail; cd {shlex.quote(workspace)}; "
            f"export CONDA_ENVS_PATH={shlex.quote(workspace)}/.conda/envs; "
            f"export CONDA_PKGS_DIRS={shlex.quote(workspace)}/.conda/pkgs; "
            f"export HOME={shlex.quote(workspace)}; "
            f"conda run -n {shlex.quote(conda_env)} python -c "
            + shlex.quote("import sys; print('python=' + sys.version.split()[0])")
        )
        completed = self._run(["ssh", ssh_alias, command])
        if completed.returncode == 0:
            return RemoteHealthCheck("conda_env", "Conda 环境运行", "pass", self._sanitize_output(completed.stdout))
        status = "blocked" if completed.returncode in {137, 255} else "fail"
        return RemoteHealthCheck(
            "conda_env",
            "Conda 环境运行",
            status,
            self._sanitize_output(completed.stderr or completed.stdout or f"returncode={completed.returncode}"),
        )

    def _parse_probe_output(self, stdout: str) -> List[RemoteHealthCheck]:
        labels = {
            "workspace": "远端 Workspace",
            "boundary": "路径安全边界",
            "project": "远端项目目录",
            "conda_cli": "Conda CLI",
            "disk": "磁盘空间",
            "gpu": "GPU 检测",
            "kaggle_config": "Kaggle 配置目录",
        }
        checks: List[RemoteHealthCheck] = []
        for line in stdout.splitlines():
            if "=" not in line:
                continue
            key, rest = line.split("=", 1)
            if key not in labels or ":" not in rest:
                continue
            status, detail = rest.split(":", 1)
            checks.append(
                RemoteHealthCheck(
                    key=key,
                    label=labels[key],
                    status=status,
                    detail=self._sanitize_output(detail),
                )
            )
        return checks

    def _run(self, command: List[str]) -> subprocess.CompletedProcess:
        try:
            return self.runner(command, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(command, 124, stdout=exc.stdout or "", stderr="timeout")

    def _overall_status(self, checks: List[RemoteHealthCheck]) -> str:
        statuses = {check.status for check in checks}
        if "fail" in statuses:
            return "fail"
        if "blocked" in statuses:
            return "blocked"
        if "warn" in statuses or "needs_review" in statuses:
            return "needs_review"
        return "pass"

    def _safe_remote_config(self) -> Dict[str, Any]:
        remote = self.config.get("remote") or {}
        return {
            "ssh_alias": remote.get("ssh_alias") or "dev",
            "workspace": remote.get("workspace") or "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac",
            "project_subdir": remote.get("project_subdir") or "workspaces/AutoKaggle",
            "conda_env": remote.get("conda_env") or "mac",
        }

    def _sanitize_output(self, text: Any) -> str:
        value = str(text or "").strip().replace("\n", " ")
        for marker in ["KAGGLE_KEY", "AUTOKAGGLE_API_KEY"]:
            value = value.replace(marker, "<secret_env>")
        if len(value) > 500:
            value = value[:500] + "..."
        return value
