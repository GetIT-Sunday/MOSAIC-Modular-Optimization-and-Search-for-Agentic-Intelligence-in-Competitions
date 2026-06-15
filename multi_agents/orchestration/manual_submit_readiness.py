from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .human_gate import HumanGate
from .kaggle_submit_adapter import KaggleSubmitAdapter
from .memory import CompetitionMemory, ExperimentRecord
from .post_reselection_gate import PostReselectionGate
from .run_ledger import RunLedger


@dataclass(frozen=True)
class ManualSubmitReadinessResult:
    status: str
    report_path: Path


class ManualSubmitReadinessChecker:
    """Summarize final submission readiness without performing any submission."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self, *, refresh: bool = True, submission_target: str = "champion") -> ManualSubmitReadinessResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")
        if refresh:
            KaggleSubmitAdapter(self.competition_dir, memory=self.memory).preflight_environment()
            PostReselectionGate(self.competition_dir, memory=self.memory).run(submission_target=submission_target)

        preflight = self._read_json(self.competition_dir / "kaggle_env_preflight.json")
        post_gate = self._read_json(self.competition_dir / "post_reselection_gate.json")
        submission_gate = self._read_json(self.competition_dir / "submission_gate.json")
        submit_plan = self._read_json(self.competition_dir / "kaggle_submit_plan.json")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        submission_policy = self._read_json(self.competition_dir / "submission_policy.json")
        champion_submission_path = self.competition_dir / "champion_submission.csv"
        recommended_submission_path = self.competition_dir / "recommended_submission.csv"
        submission_path = recommended_submission_path if submission_target == "recommended" else champion_submission_path
        submit_human_gate = self._latest_human_gate("kaggle_submit_plan")

        issues = []
        warnings = []
        warnings.extend(preflight.get("warnings", []))
        warnings.extend(post_gate.get("warnings", []))
        warnings.extend(submission_gate.get("warnings", []))
        warnings.extend(submit_plan.get("warnings", []))

        manual_submission_ready = True
        if post_gate.get("status") != "pass":
            manual_submission_ready = False
            issues.append("post_reselection_gate.json is not pass.")
        if submission_gate.get("status") != "pass":
            manual_submission_ready = False
            issues.append("submission_gate.json is not pass.")
        if submission_gate.get("submission_target", "champion") != submission_target:
            manual_submission_ready = False
            issues.append("submission_gate.json target does not match requested submission target.")
        if submit_plan.get("submission_target", "champion") != submission_target:
            manual_submission_ready = False
            issues.append("kaggle_submit_plan.json target does not match requested submission target.")
        if submission_target == "recommended" and submission_policy.get("decision") != "recommended_submission_selected":
            manual_submission_ready = False
            issues.append("submission_policy.json has not selected a recommended submission.")
        if not submission_path.exists():
            manual_submission_ready = False
            issues.append(f"{submission_path.name} is missing.")

        credentials_available = bool((submit_plan.get("credentials") or {}).get("available"))
        if not submit_plan:
            credentials_available = bool((preflight.get("credentials") or {}).get("available"))
        kaggle_cli_available = bool(
            submit_plan.get("kaggle_cli_available")
            if "kaggle_cli_available" in submit_plan
            else (preflight.get("tools") or {}).get("kaggle_cli_available")
        )
        explicit_submit_approved = (
            submit_human_gate.decision == "continue"
            and "approve_real_submit" in submit_human_gate.notes
        )
        api_submission_review_ready = (
            manual_submission_ready
            and submit_plan.get("status") == "pass"
            and kaggle_cli_available
            and credentials_available
        )
        confirmed_submit_ready = api_submission_review_ready and explicit_submit_approved

        if not kaggle_cli_available:
            warnings.append("Kaggle CLI is not available; manual browser upload may still be possible.")
        if not credentials_available:
            warnings.append("Kaggle credentials are missing; API submission remains blocked.")
        if api_submission_review_ready and not explicit_submit_approved:
            warnings.append("API submission has tooling and credentials, but human review lacks approve_real_submit.")

        if confirmed_submit_ready:
            status = "ready_for_confirmed_api_submit"
            decision = "confirmed_submit_ready"
            next_action = f"Run confirmed submit only if you still intend to send the current {submission_target} submission to Kaggle."
        elif manual_submission_ready:
            status = "manual_submit_ready"
            decision = "manual_submit_or_finish_api_setup"
            next_action = (
                "Manual upload is ready. For API submission, add Kaggle credentials and approve the kaggle_submit_plan human gate."
            )
        else:
            status = "needs_review"
            decision = "submission_not_ready"
            next_action = "Fix blocking submission issues, then rerun --manual-submit-readiness."

        champion = champion_selection.get("champion") or post_gate.get("champion") or {}
        candidate = submission_gate.get("candidate") or (
            submission_policy.get("recommended_submission_candidate") if submission_target == "recommended" else champion
        ) or {}
        report = {
            "competition_name": champion_selection.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": decision,
            "submission_target": submission_target,
            "manual_submission_ready": manual_submission_ready,
            "api_submission_review_ready": api_submission_review_ready,
            "confirmed_submit_ready": confirmed_submit_ready,
            "champion": champion,
            "candidate": candidate,
            "champion_submission_path": str(champion_submission_path),
            "recommended_submission_path": str(recommended_submission_path),
            "submission_path": str(submission_path),
            "post_reselection_gate_status": post_gate.get("status"),
            "submission_gate_status": submission_gate.get("status"),
            "kaggle_submit_plan_status": submit_plan.get("status"),
            "kaggle_cli_available": kaggle_cli_available,
            "credentials_available": credentials_available,
            "human_gate": submit_human_gate.to_dict(),
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "next_action": next_action,
        }

        report_path = self.competition_dir / "manual_submit_readiness.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="manual_submit_readiness",
            agent="manual_submit_readiness",
            title="Review manual and API submit readiness",
            status=status,
            input_payload=report,
            prompt="Summarize final Kaggle submission readiness without performing any submission.",
            scorecard={
                "agent": "manual_submit_readiness",
                "task_id": "manual_submit_readiness",
                "status": status,
                "scores": {
                    "submission_target": submission_target,
                    "manual_submission_ready": manual_submission_ready,
                    "api_submission_review_ready": api_submission_review_ready,
                    "confirmed_submit_ready": confirmed_submit_ready,
                    "kaggle_cli_available": kaggle_cli_available,
                    "credentials_available": credentials_available,
                },
                "metric_name": candidate.get("metric_name"),
                "local_score": candidate.get("local_score"),
                "issues": issues + list(dict.fromkeys(warnings)),
                "recommended_human_action": "continue" if manual_submission_ready else "patch_prompt",
            },
            artifacts={
                "manual_submit_readiness": report_path,
                "post_reselection_gate": self.competition_dir / "post_reselection_gate.json",
                "submission_gate": self.competition_dir / "submission_gate.json",
                "kaggle_submit_plan": self.competition_dir / "kaggle_submit_plan.json",
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="manual_submit_readiness",
                status=status,
                metric_name=candidate.get("metric_name"),
                local_score=candidate.get("local_score"),
                submission_path=str(submission_path) if submission_path.exists() else None,
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=next_action,
            )
        )
        return ManualSubmitReadinessResult(status=status, report_path=report_path)

    def _latest_human_gate(self, task_id: str):
        matches = sorted((self.competition_dir / "runs").glob(f"*_{task_id}/human_review.md"))
        return HumanGate.parse(matches[-1] if matches else self.competition_dir / "missing_human_review.md")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
