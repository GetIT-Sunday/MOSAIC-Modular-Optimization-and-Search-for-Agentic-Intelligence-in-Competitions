from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .human_gate import HumanGate
from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .validator import SubmissionValidator


@dataclass(frozen=True)
class SubmissionGateResult:
    status: str
    gate_path: Path
    submission_path: Path


class SubmissionGate:
    """Final dry-run gate before a Kaggle submission is allowed."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self, dry_run: bool = True, submission_target: str = "champion") -> SubmissionGateResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        selection_path = self.competition_dir / "champion_selection.json"
        selection = self._read_json(selection_path)
        policy_path = self.competition_dir / "submission_policy.json"
        policy = self._read_json(policy_path)
        if submission_target == "recommended":
            submission_path = self.competition_dir / "recommended_submission.csv"
            candidate = policy.get("recommended_submission_candidate") or {}
            gate_source_path = policy_path
            gate_task_id = "submission_policy"
        else:
            submission_path = self.competition_dir / "champion_submission.csv"
            candidate = selection.get("champion") or {}
            gate_source_path = selection_path
            gate_task_id = "champion_selection"
        validation = SubmissionValidator(manifest).validate(submission_path)
        human_gate = self._latest_human_gate(gate_task_id)
        issues = []
        warnings = list(validation.warnings)

        if not dry_run:
            issues.append("Real Kaggle submission is not enabled by this gate yet; run in dry-run mode.")
        if selection.get("decision") != "champion_selected":
            issues.append("Champion selector has not selected a champion.")
        if submission_target == "recommended" and policy.get("decision") != "recommended_submission_selected":
            issues.append("Submission policy has not selected a recommended submission.")
        if not submission_path.exists():
            issues.append(f"{submission_path.name} is missing.")
        if not validation.ok:
            issues.extend(validation.errors)
        risk_level = candidate.get("risk_level")
        if risk_level == "high":
            issues.append(f"{submission_target} submission risk level is high.")
        elif risk_level not in {"low", "medium"}:
            warnings.append(f"{submission_target} submission risk level is {risk_level or 'unknown'}; prefer running risk audit before real submission.")
        if human_gate.decision != "continue":
            issues.append(f"Human gate for {gate_task_id} is {human_gate.decision}.")
        if human_gate.notes.strip():
            warnings.append("Human gate notes are present; review before real submission.")
        if manifest.competition_name == "unknown":
            issues.append("Competition name is unknown.")

        status = "pass" if not issues else "needs_review"
        gate = {
            "competition_name": manifest.competition_name,
            "dry_run": dry_run,
            "status": status,
            "decision": "ready_for_manual_or_api_submission" if status == "pass" else "submission_blocked",
            "submission_target": submission_target,
            "candidate": candidate,
            "champion": selection.get("champion") or {},
            "recommended_submission_candidate": policy.get("recommended_submission_candidate"),
            "champion_selection_path": str(selection_path),
            "submission_policy_path": str(policy_path),
            "champion_submission_path": str(self.competition_dir / "champion_submission.csv"),
            "recommended_submission_path": str(self.competition_dir / "recommended_submission.csv"),
            "submission_path": str(submission_path),
            "gate_source_path": str(gate_source_path),
            "submission_validation": validation.to_dict(),
            "human_gate": human_gate.to_dict(),
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                f"Enable Kaggle API submission or manually submit {submission_path.name}."
                if status == "pass"
                else "Fix gate issues, rerun champion selection if needed, then rerun submission gate."
            ),
        }
        gate_path = self.competition_dir / "submission_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="submission_gate",
            agent="submission_gate",
            title="Dry-run final submission gate",
            status=status,
            input_payload=gate,
            prompt="Check whether the current champion submission is ready for Kaggle submission.",
            scorecard={
                "agent": "submission_gate",
                "task_id": "submission_gate",
                "status": status,
                "scores": {
                    "submission_valid": 5 if validation.ok else 1,
                    "submission_target": submission_target,
                    "risk_level": candidate.get("risk_level", "unknown"),
                    "dry_run": dry_run,
                    "human_gate": human_gate.decision,
                },
                "metric_name": candidate.get("metric_name"),
                "local_score": candidate.get("local_score"),
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"submission_gate": gate_path, "submission": submission_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="submission_gate",
                status=status,
                metric_name=candidate.get("metric_name"),
                local_score=candidate.get("local_score"),
                submission_path=str(submission_path) if submission_path.exists() else None,
                brain_review_path=str(gate_path),
                artifacts=[str(gate_path), str(submission_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=gate["next_action"],
            )
        )
        return SubmissionGateResult(status=status, gate_path=gate_path, submission_path=submission_path)

    def _latest_human_gate(self, task_id: str):
        matches = sorted((self.competition_dir / "runs").glob(f"*_{task_id}/human_review.md"))
        return HumanGate.parse(matches[-1] if matches else self.competition_dir / "missing_human_review.md")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
