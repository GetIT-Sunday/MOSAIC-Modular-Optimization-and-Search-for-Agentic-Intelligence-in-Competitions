from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .leaderboard_feedback import LeaderboardFeedbackRecorder
from .leaderboard_feedback_loop import LeaderboardFeedbackLoop
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class LeaderboardFeedbackInputResult:
    status: str
    report_path: Path
    feedback_loop_report_path: Optional[Path]
    experiment_roadmap_path: Optional[Path] = None


class LeaderboardFeedbackInputRunner:
    """Validate filled leaderboard feedback JSON and run the feedback loop."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(
        self,
        *,
        input_path: Optional[Path] = None,
        brain_review: bool = True,
        brain_use_llm: bool = True,
    ) -> LeaderboardFeedbackInputResult:
        input_path = input_path or self.competition_dir / "leaderboard_feedback_input_template.json"
        if not input_path.is_absolute():
            input_path = self.competition_dir / input_path
        payload = self._read_json(input_path)
        workflow = self._read_json(self.competition_dir / "post_submit_workflow.json")
        issues = []
        warnings = []

        if not payload:
            issues.append("Leaderboard feedback input JSON is missing or empty.")
        placeholder_fields = set()
        for field in ["public_score", "private_score", "submission_id", "source", "notes"]:
            if LeaderboardFeedbackRecorder._looks_like_placeholder(payload.get(field)):
                placeholder_fields.add(field)
                issues.append(f"{field} still contains an unreplaced placeholder.")

        public_score = None if "public_score" in placeholder_fields else self._optional_float(payload.get("public_score"), "public_score", issues)
        private_score = None if "private_score" in placeholder_fields else self._optional_float(payload.get("private_score"), "private_score", issues)
        leaderboard_rank = self._optional_int(payload.get("leaderboard_rank"), "leaderboard_rank", issues)
        submission_id = self._optional_str(payload.get("submission_id"))
        source = self._optional_str(payload.get("source")) or "manual"
        notes = self._optional_str(payload.get("notes")) or ""
        submission_target = self._optional_str(payload.get("submission_target")) or "champion"
        candidate_task_id = self._optional_str(payload.get("candidate_task_id"))

        if submission_target not in {"champion", "recommended"}:
            issues.append(f"Unsupported submission_target: {submission_target}")
        if public_score is None and private_score is None and leaderboard_rank is None and not submission_id:
            issues.append("At least one leaderboard signal is required: public_score, private_score, leaderboard_rank, or submission_id.")
        if leaderboard_rank is not None and leaderboard_rank <= 0:
            issues.append("leaderboard_rank must be positive when provided.")

        expected_target = workflow.get("submission_target")
        expected_candidate = (workflow.get("candidate") or {}).get("task_id")
        if expected_target and expected_target != submission_target:
            issues.append(f"submission_target {submission_target} does not match post_submit_workflow target {expected_target}.")
        if expected_candidate and candidate_task_id and expected_candidate != candidate_task_id:
            issues.append(f"candidate_task_id {candidate_task_id} does not match post_submit_workflow candidate {expected_candidate}.")
        if expected_candidate and not candidate_task_id:
            warnings.append("candidate_task_id is missing; post_submit_workflow candidate will be used for auditing context.")
        expected_template = workflow.get("record_feedback_template") or {}
        self._validate_expected_binding(payload, expected_template, issues)
        submission_binding = {
            "expected_submission_sha256": payload.get("expected_submission_sha256"),
            "expected_submission_rows": payload.get("expected_submission_rows"),
            "expected_submission_columns": payload.get("expected_submission_columns"),
            "candidate_risk_level": payload.get("candidate_risk_level"),
        }

        loop_result = None
        if not issues:
            loop_result = LeaderboardFeedbackLoop(self.competition_dir, memory=self.memory).run(
                public_score=public_score,
                private_score=private_score,
                leaderboard_rank=leaderboard_rank,
                submission_id=submission_id,
                source=source,
                notes=notes,
                submission_target=submission_target,
                brain_review=brain_review,
                brain_use_llm=brain_use_llm,
                submission_binding=submission_binding,
            )
            if loop_result.status != "pass":
                warnings.append(f"leaderboard_feedback_loop status is {loop_result.status}.")

        status = "needs_review" if issues else (loop_result.status if loop_result else "needs_review")
        report = {
            "competition_name": workflow.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": "feedback_loop_started" if not issues else "feedback_input_blocked",
            "input_path": str(input_path),
            "submission_target": submission_target,
            "candidate_task_id": candidate_task_id,
            "public_score": public_score,
            "private_score": private_score,
            "leaderboard_rank": leaderboard_rank,
            "submission_id": submission_id,
            "expected_submission_sha256": payload.get("expected_submission_sha256"),
            "expected_submission_rows": payload.get("expected_submission_rows"),
            "expected_submission_columns": payload.get("expected_submission_columns"),
            "candidate_risk_level": payload.get("candidate_risk_level"),
            "source": source,
            "notes": notes,
            "feedback_loop_report_path": str(loop_result.report_path) if loop_result else None,
            "experiment_roadmap_path": str(loop_result.experiment_roadmap_path) if loop_result and loop_result.experiment_roadmap_path else None,
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Inspect leaderboard_feedback_loop.json and the refreshed Brain plan."
                if not issues
                else "Fill leaderboard_feedback_input_template.json with real values and rerun --leaderboard-feedback-from-template."
            ),
        }
        report_path = self.competition_dir / "leaderboard_feedback_input_validation.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="leaderboard_feedback_input",
            agent="leaderboard_feedback_input",
            title="Validate leaderboard feedback input",
            status=status,
            input_payload=report,
            prompt="Validate filled leaderboard feedback JSON and run the leaderboard feedback loop when the input is clean.",
            scorecard={
                "agent": "leaderboard_feedback_input",
                "task_id": "leaderboard_feedback_input",
                "status": status,
                "scores": {
                    "submission_target": submission_target,
                    "candidate_task_id": candidate_task_id or "n/a",
                    "public_score": public_score if public_score is not None else "n/a",
                    "leaderboard_rank": leaderboard_rank if leaderboard_rank is not None else "n/a",
                    "feedback_loop_started": loop_result is not None,
                },
                "metric_name": None,
                "local_score": None,
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "leaderboard_feedback_input_validation": report_path,
                "leaderboard_feedback_input": input_path,
                "leaderboard_feedback_loop": loop_result.report_path if loop_result else Path(""),
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="leaderboard_feedback_input",
                status=status,
                public_score=public_score,
                leaderboard_rank=leaderboard_rank,
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=report["next_action"],
            )
        )
        return LeaderboardFeedbackInputResult(
            status=status,
            report_path=report_path,
            feedback_loop_report_path=loop_result.report_path if loop_result else None,
            experiment_roadmap_path=loop_result.experiment_roadmap_path if loop_result else None,
        )

    @staticmethod
    def _validate_expected_binding(
        payload: Dict[str, Any],
        expected_template: Dict[str, Any],
        issues: list[str],
    ) -> None:
        checks = [
            "expected_submission_sha256",
            "expected_submission_rows",
            "expected_submission_columns",
            "candidate_risk_level",
        ]
        for field in checks:
            expected = expected_template.get(field)
            if expected is None:
                continue
            actual = payload.get(field)
            if actual is None:
                issues.append(f"{field} is missing from leaderboard feedback input.")
            elif actual != expected:
                issues.append(f"{field} does not match post_submit_workflow template.")

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    @staticmethod
    def _optional_float(value: Any, field_name: str, issues: list[str]) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            issues.append(f"{field_name} must be numeric when provided.")
            return None

    @staticmethod
    def _optional_int(value: Any, field_name: str, issues: list[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            issues.append(f"{field_name} must be an integer when provided.")
            return None

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
