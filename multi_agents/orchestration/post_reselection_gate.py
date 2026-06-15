from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .kaggle_submit_adapter import KaggleSubmitAdapter
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .submission_gate import SubmissionGate


@dataclass(frozen=True)
class PostReselectionGateResult:
    status: str
    report_path: Path
    submission_gate_path: Path
    submit_plan_path: Path


class PostReselectionGate:
    """Refresh submission gate and Kaggle dry-run plan after champion reselection."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self, submission_target: str = "champion") -> PostReselectionGateResult:
        selection_before = self._read_json(self.competition_dir / "champion_selection.json")
        champion = selection_before.get("champion") or {}
        submission_gate = SubmissionGate(self.competition_dir, memory=self.memory).run(
            dry_run=True,
            submission_target=submission_target,
        )
        submit_plan = KaggleSubmitAdapter(self.competition_dir, memory=self.memory).plan(
            dry_run=True,
            submission_target=submission_target,
        )
        gate_payload = self._read_json(submission_gate.gate_path)
        plan_payload = self._read_json(submit_plan.plan_path)
        issues = []
        warnings = []

        if not champion:
            issues.append("Champion selection is missing.")
        if gate_payload.get("status") != "pass":
            issues.append("Refreshed submission_gate.json is not pass.")
        if plan_payload.get("status") != "pass":
            issues.append("Kaggle submit dry-run plan is not pass.")
        if submission_target == "champion" and gate_payload.get("champion", {}).get("task_id") != champion.get("task_id"):
            issues.append("Refreshed submission gate does not match current champion task_id.")
        if gate_payload.get("submission_target", "champion") != submission_target:
            issues.append("Refreshed submission gate target does not match requested target.")
        if plan_payload.get("submission_target", "champion") != submission_target:
            issues.append("Kaggle submit dry-run plan target does not match requested target.")
        warnings.extend(plan_payload.get("warnings", []))
        warnings.extend(gate_payload.get("warnings", []))

        status = "pass" if not issues else "needs_review"
        report = {
            "competition_name": selection_before.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": "ready_for_human_submit_review" if status == "pass" else "post_reselection_blocked",
            "submission_target": submission_target,
            "champion": champion,
            "candidate": gate_payload.get("candidate") or champion,
            "submission_gate_status": gate_payload.get("status"),
            "submission_gate_path": str(submission_gate.gate_path),
            "kaggle_submit_plan_status": plan_payload.get("status"),
            "kaggle_submit_plan_path": str(submit_plan.plan_path),
            "kaggle_cli_available": plan_payload.get("kaggle_cli_available"),
            "credentials_available": (plan_payload.get("credentials") or {}).get("available"),
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Review refreshed gate and dry-run plan, then decide whether to manually submit or approve real submit."
                if status == "pass"
                else "Fix post-reselection gate issues before any submission."
            ),
        }
        report_path = self.competition_dir / "post_reselection_gate.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="post_reselection_gate",
            agent="post_reselection_gate",
            title="Refresh post-reselection submission gate",
            status=status,
            input_payload=report,
            prompt="After risk-aware champion reselection, refresh submission gate and Kaggle dry-run plan before any submission.",
            scorecard={
                "agent": "post_reselection_gate",
                "task_id": "post_reselection_gate",
                "status": status,
                "scores": {
                    "submission_gate": gate_payload.get("status", "missing"),
                    "kaggle_submit_plan": plan_payload.get("status", "missing"),
                    "submission_target": submission_target,
                    "kaggle_cli_available": plan_payload.get("kaggle_cli_available", "unknown"),
                    "credentials_available": (plan_payload.get("credentials") or {}).get("available", "unknown"),
                },
                "metric_name": champion.get("metric_name"),
                "local_score": champion.get("local_score"),
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "post_reselection_gate": report_path,
                "submission_gate": submission_gate.gate_path,
                "kaggle_submit_plan": submit_plan.plan_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="post_reselection_gate",
                status=status,
                metric_name=champion.get("metric_name"),
                local_score=champion.get("local_score"),
                submission_path=str(self.competition_dir / "champion_submission.csv"),
                brain_review_path=str(report_path),
                artifacts=[
                    str(report_path),
                    str(submission_gate.gate_path),
                    str(submit_plan.plan_path),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=report["next_action"],
            )
        )
        return PostReselectionGateResult(
            status=status,
            report_path=report_path,
            submission_gate_path=submission_gate.gate_path,
            submit_plan_path=submit_plan.plan_path,
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
