from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .human_gate import HumanGate
from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class KaggleSubmitPlanResult:
    status: str
    plan_path: Path


class KaggleSubmitAdapter:
    """Build a dry-run Kaggle submission plan without submitting."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def preflight_environment(self) -> KaggleSubmitPlanResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        credential_state = self._credential_state()
        kaggle_cli = shutil.which("kaggle")
        pytest_available = find_spec("pytest") is not None
        remote_workspace = os.getenv("AUTOKAGGLE_REMOTE_WORKSPACE") == "1"
        hard_workspace = Path("/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac")
        expected_project = hard_workspace / "workspaces" / "AutoKaggle"
        in_expected_remote_project = self._is_relative_to(self.competition_dir, expected_project)
        conda_env = os.getenv("CONDA_DEFAULT_ENV") or ""
        python_path = Path(sys.executable).resolve()
        issues = []
        warnings = []

        if remote_workspace and not in_expected_remote_project:
            issues.append("Remote workspace flag is set, but competition_dir is outside the hard remote project.")
        if remote_workspace and conda_env != "mac":
            warnings.append("Remote conda env is not named mac.")
        if not kaggle_cli:
            warnings.append("Kaggle CLI is not available in PATH.")
        if not pytest_available:
            warnings.append("pytest is not available in the active Python environment.")
        if not credential_state["available"]:
            warnings.append("Kaggle credentials were not detected in allowed locations.")
        if credential_state.get("permission_warnings"):
            warnings.extend(credential_state["permission_warnings"])

        status = "pass" if not issues else "needs_review"
        report = {
            "competition_name": manifest.competition_name,
            "status": status,
            "decision": "environment_ready_or_needs_optional_setup" if status == "pass" else "environment_blocked",
            "remote_workspace_flag": remote_workspace,
            "hard_remote_workspace": str(hard_workspace),
            "expected_remote_project": str(expected_project),
            "competition_dir": str(self.competition_dir),
            "in_expected_remote_project": in_expected_remote_project,
            "python": {
                "executable": str(python_path),
                "version": sys.version.split()[0],
                "conda_default_env": conda_env or None,
            },
            "tools": {
                "kaggle_cli_available": kaggle_cli is not None,
                "kaggle_cli_path": kaggle_cli,
                "pytest_available": pytest_available,
            },
            "credentials": credential_state,
            "issues": issues,
            "warnings": warnings,
            "next_action": self._preflight_next_action(issues, warnings, credential_state, kaggle_cli, pytest_available),
        }
        report_path = self.competition_dir / "kaggle_env_preflight.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="kaggle_env_preflight",
            agent="kaggle_submit_adapter",
            title="Check Kaggle environment preflight",
            status=status,
            input_payload=report,
            prompt="Check the remote Kaggle submission environment without submitting or printing secrets.",
            scorecard={
                "agent": "kaggle_submit_adapter",
                "task_id": "kaggle_env_preflight",
                "status": status,
                "scores": {
                    "remote_workspace": remote_workspace,
                    "in_expected_remote_project": in_expected_remote_project,
                    "conda_env": conda_env or "unknown",
                    "kaggle_cli_available": kaggle_cli is not None,
                    "pytest_available": pytest_available,
                    "credentials_available": credential_state["available"],
                },
                "metric_name": None,
                "local_score": None,
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"kaggle_env_preflight": report_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="kaggle_env_preflight",
                status=status,
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=report["next_action"],
            )
        )
        return KaggleSubmitPlanResult(status=status, plan_path=report_path)

    def plan(
        self,
        *,
        dry_run: bool = True,
        message: str = "AutoKaggle champion submission",
        submission_target: str = "champion",
    ) -> KaggleSubmitPlanResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        gate = self._read_json(self.competition_dir / "submission_gate.json")
        submission_path = self.competition_dir / (
            "recommended_submission.csv" if submission_target == "recommended" else "champion_submission.csv"
        )
        competition_slug = manifest.competition_name
        kaggle_cli = shutil.which("kaggle")
        credential_state = self._credential_state()
        issues = []
        warnings = []

        if not dry_run:
            issues.append("Real Kaggle submission is not enabled; this adapter currently supports dry-run only.")
        if gate.get("status") != "pass":
            issues.append("submission_gate.json is not pass; run --submission-gate first.")
        if gate.get("submission_target", "champion") != submission_target:
            issues.append(f"submission_gate.json target is {gate.get('submission_target', 'champion')}, not {submission_target}.")
        if not submission_path.exists():
            issues.append(f"{submission_path.name} is missing.")
        if competition_slug == "unknown":
            issues.append("Competition slug is unknown.")
        if kaggle_cli is None:
            warnings.append("Kaggle CLI is not available in PATH.")
        if not credential_state["available"]:
            warnings.append("Kaggle credentials were not detected in allowed locations.")

        command = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            competition_slug,
            "-f",
            str(submission_path),
            "-m",
            message,
        ]
        status = "pass" if not issues else "needs_review"
        plan = {
            "competition_name": manifest.competition_name,
            "dry_run": dry_run,
            "submission_target": submission_target,
            "status": status,
            "decision": "ready_for_explicit_submit_command" if status == "pass" else "submit_blocked",
            "submission_gate_status": gate.get("status"),
            "submission_gate_target": gate.get("submission_target", "champion"),
            "champion_submission_path": str(self.competition_dir / "champion_submission.csv"),
            "recommended_submission_path": str(self.competition_dir / "recommended_submission.csv"),
            "submission_path": str(submission_path),
            "candidate": gate.get("candidate") or gate.get("champion"),
            "kaggle_cli_available": kaggle_cli is not None,
            "kaggle_cli_path": kaggle_cli,
            "credentials": credential_state,
            "submit_command_preview": command,
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Review this plan and explicitly enable a real submit command in a later step."
                if status == "pass"
                else "Fix blocking issues, rerun --submission-gate, then rerun this dry-run adapter."
            ),
        }
        plan_path = self.competition_dir / "kaggle_submit_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="kaggle_submit_plan",
            agent="kaggle_submit_adapter",
            title="Build Kaggle submit dry-run plan",
            status=status,
            input_payload=plan,
            prompt="Build a dry-run Kaggle submit plan for the current champion without performing a real submission.",
            scorecard={
                "agent": "kaggle_submit_adapter",
                "task_id": "kaggle_submit_plan",
                "status": status,
                "scores": {
                    "submission_gate": gate.get("status", "missing"),
                    "submission_target": submission_target,
                    "kaggle_cli_available": kaggle_cli is not None,
                    "credentials_available": credential_state["available"],
                    "dry_run": dry_run,
                },
                "metric_name": (gate.get("candidate") or gate.get("champion") or {}).get("metric_name"),
                "local_score": (gate.get("candidate") or gate.get("champion") or {}).get("local_score"),
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"kaggle_submit_plan": plan_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="kaggle_submit_plan",
                status=status,
                metric_name=(gate.get("candidate") or gate.get("champion") or {}).get("metric_name"),
                local_score=(gate.get("candidate") or gate.get("champion") or {}).get("local_score"),
                submission_path=str(submission_path) if submission_path.exists() else None,
                brain_review_path=str(plan_path),
                artifacts=[str(plan_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=plan["next_action"],
            )
        )
        return KaggleSubmitPlanResult(status=status, plan_path=plan_path)

    def confirmed_submit(
        self,
        *,
        confirmed: bool = False,
        message: str = "AutoKaggle champion submission",
        submission_target: str = "champion",
    ) -> KaggleSubmitPlanResult:
        plan_result = self.plan(dry_run=True, message=message, submission_target=submission_target)
        plan = self._read_json(plan_result.plan_path)
        human_gate = self._latest_human_gate("kaggle_submit_plan")
        issues = list(plan.get("issues", []))
        warnings = list(plan.get("warnings", []))
        if not confirmed:
            issues.append("Missing --kaggle-submit-confirmed flag.")
        if plan.get("submission_gate_status") != "pass":
            issues.append("submission_gate.json is not pass.")
        if not plan.get("kaggle_cli_available"):
            issues.append("Kaggle CLI is not available.")
        if not (plan.get("credentials") or {}).get("available"):
            issues.append("Kaggle credentials are not available.")
        if human_gate.decision != "continue":
            issues.append(f"kaggle_submit_plan human gate is {human_gate.decision}.")
        if "approve_real_submit" not in human_gate.notes:
            issues.append("kaggle_submit_plan human gate notes must include approve_real_submit.")

        status = "blocked"
        returncode = None
        stdout = ""
        stderr = ""
        command = plan.get("submit_command_preview") or []
        if not issues:
            completed = subprocess.run(
                command,
                cwd=str(self.competition_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            status = "submitted" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                issues.append("Kaggle CLI submit command failed.")

        result = {
            "competition_name": plan.get("competition_name"),
            "dry_run": False,
            "confirmed": confirmed,
            "status": status,
            "decision": "submitted" if status == "submitted" else "submit_blocked_or_failed",
            "human_gate": human_gate.to_dict(),
            "submit_command_preview": command,
            "returncode": returncode,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Record the public leaderboard score."
                if status == "submitted"
                else "Resolve blocking issues and rerun dry-run before trying a real submit again."
            ),
        }
        result_path = self.competition_dir / "kaggle_submit_result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="kaggle_submit_confirmed",
            agent="kaggle_submit_adapter",
            title="Attempt confirmed Kaggle submission",
            status="pass" if status == "submitted" else "needs_review",
            input_payload=result,
            prompt="Attempt a real Kaggle submission only when all explicit approval gates are satisfied.",
            scorecard={
                "agent": "kaggle_submit_adapter",
                "task_id": "kaggle_submit_confirmed",
                "status": "pass" if status == "submitted" else "needs_review",
                "scores": {
                    "confirmed": confirmed,
                    "human_gate": human_gate.decision,
                    "returncode": returncode if returncode is not None else "n/a",
                },
                "metric_name": None,
                "local_score": None,
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "submitted" else "patch_prompt",
            },
            artifacts={"kaggle_submit_result": result_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=plan.get("competition_name") or self.competition_dir.name,
                profile_name="tabular_classic",
                task_id="kaggle_submit_confirmed",
                status=status,
                brain_review_path=str(result_path),
                artifacts=[str(result_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=result["next_action"],
            )
        )
        return KaggleSubmitPlanResult(status=status, plan_path=result_path)

    def _credential_state(self) -> Dict[str, Any]:
        env_pair = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        candidates = []
        config_dir = os.getenv("KAGGLE_CONFIG_DIR")
        if config_dir:
            candidates.append(Path(config_dir) / "kaggle.json")
        candidates.append(self.competition_dir / ".kaggle" / "kaggle.json")
        found_files = [str(path) for path in candidates if path.exists()]
        permission_warnings = []
        for path in candidates:
            if not path.exists():
                continue
            mode = path.stat().st_mode & 0o777
            if mode & 0o077:
                permission_warnings.append(f"{path} should be chmod 600 before real submission.")
        return {
            "available": env_pair or bool(found_files),
            "env_pair_present": env_pair,
            "credential_files_present": found_files,
            "allowed_locations_checked": [str(path) for path in candidates],
            "permission_warnings": permission_warnings,
        }

    def _latest_human_gate(self, task_id: str):
        matches = sorted((self.competition_dir / "runs").glob(f"*_{task_id}/human_review.md"))
        return HumanGate.parse(matches[-1] if matches else self.competition_dir / "missing_human_review.md")

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _preflight_next_action(
        issues: list[str],
        warnings: list[str],
        credential_state: Dict[str, Any],
        kaggle_cli: Optional[str],
        pytest_available: bool,
    ) -> str:
        if issues:
            return "Fix blocking workspace issues before running submission tooling."
        missing = []
        if not kaggle_cli:
            missing.append("install Kaggle CLI in the remote mac conda environment")
        if not pytest_available:
            missing.append("install pytest in the remote mac conda environment")
        if not credential_state.get("available"):
            missing.append("create Kaggle credentials in the allowed remote competition .kaggle directory or environment")
        if missing:
            return "Optional setup before real submission: " + "; ".join(missing) + "."
        if warnings:
            return "Review warnings, then rerun --kaggle-submit-dry-run before any confirmed submit."
        return "Environment preflight passed. Rerun --kaggle-submit-dry-run before any confirmed submit."

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
