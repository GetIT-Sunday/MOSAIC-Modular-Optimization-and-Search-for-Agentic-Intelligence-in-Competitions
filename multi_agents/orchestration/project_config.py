from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ConfigCheck:
    key: str
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class ProjectConfigSnapshot:
    config_path: str
    example_path: str
    using_private_config: bool
    checks: List[ConfigCheck]
    safe_config: Dict[str, Any]


class ProjectConfigAgent:
    """Load and audit shareable AutoKaggle project configuration without exposing secrets."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.example_path = self.project_root / "autokaggle_config.example.json"
        self.private_path = self.project_root / "autokaggle_config.json"

    def load(self) -> Dict[str, Any]:
        config = self._read_json(self.example_path)
        private = self._read_json(self.private_path)
        return self._deep_merge(config, private)

    def snapshot(self) -> ProjectConfigSnapshot:
        config = self.load()
        safe_config = self._redact_config(config)
        checks = self._checks(config)
        return ProjectConfigSnapshot(
            config_path=str(self.private_path),
            example_path=str(self.example_path),
            using_private_config=self.private_path.exists(),
            checks=checks,
            safe_config=safe_config,
        )

    def write_status(self, path: Path) -> Path:
        snapshot = self.snapshot()
        payload = {
            "config_path": snapshot.config_path,
            "example_path": snapshot.example_path,
            "using_private_config": snapshot.using_private_config,
            "checks": [check.__dict__ for check in snapshot.checks],
            "safe_config": snapshot.safe_config,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _checks(self, config: Dict[str, Any]) -> List[ConfigCheck]:
        remote = config.get("remote") or {}
        llm = config.get("llm") or {}
        safety = config.get("safety") or {}
        kaggle = config.get("kaggle") or {}
        api_key_env = str(llm.get("api_key_env") or "")
        api_key_file = self.project_root / str(llm.get("api_key_file") or "")
        api_key_present = bool(api_key_env and os.getenv(api_key_env)) or api_key_file.exists()
        workspace = str(remote.get("workspace") or "")
        allow_paths = [str(item) for item in safety.get("allow_paths") or []]
        checks = [
            ConfigCheck(
                "private_config",
                "私有配置文件",
                "pass" if self.private_path.exists() else "needs_setup",
                "已使用 autokaggle_config.json" if self.private_path.exists() else "复制 autokaggle_config.example.json 为 autokaggle_config.json 后填写个人配置",
            ),
            ConfigCheck(
                "ssh_alias",
                "SSH 服务器",
                "pass" if remote.get("ssh_alias") else "missing",
                str(remote.get("ssh_alias") or "未配置"),
            ),
            ConfigCheck(
                "remote_workspace",
                "远端 Workspace",
                "pass" if workspace and (not allow_paths or workspace in allow_paths) else "blocked",
                workspace or "未配置",
            ),
            ConfigCheck(
                "conda_env",
                "Conda 环境",
                "pass" if remote.get("conda_env") else "missing",
                str(remote.get("conda_env") or "未配置"),
            ),
            ConfigCheck(
                "llm_base_url",
                "LLM Base URL",
                "pass" if llm.get("openai_base_url") else "missing",
                str(llm.get("openai_base_url") or "未配置"),
            ),
            ConfigCheck(
                "llm_models",
                "LLM 模型",
                "pass" if llm.get("planner_model") and llm.get("coding_model") else "missing",
                f"planner={llm.get('planner_model') or 'missing'}, coding={llm.get('coding_model') or 'missing'}",
            ),
            ConfigCheck(
                "llm_api_key",
                "LLM API Key",
                "pass" if api_key_present else "needs_setup",
                f"使用环境变量 {api_key_env} 或本地文件 {llm.get('api_key_file')}",
            ),
            ConfigCheck(
                "kaggle_auth",
                "Kaggle 认证",
                "needs_runtime_check",
                f"远端配置目录：{kaggle.get('remote_config_dir') or '.kaggle'}",
            ),
        ]
        return checks

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _redact_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        def redact(value: Any, key: str = "") -> Any:
            lowered = key.lower()
            if any(marker in lowered for marker in ["key", "secret", "token", "password"]):
                if key in {"api_key_env", "api_key_file"}:
                    return value
                return "<redacted>"
            if isinstance(value, dict):
                return {item_key: redact(item_value, item_key) for item_key, item_value in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        return redact(config)
