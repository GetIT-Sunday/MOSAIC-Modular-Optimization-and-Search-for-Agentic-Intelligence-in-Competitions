from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class LeaderboardFeedbackResult:
    status: str
    feedback_path: Path


class LeaderboardFeedbackRecorder:
    """Record public leaderboard feedback after manual or API submission."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def record(
        self,
        *,
        public_score: Optional[float] = None,
        private_score: Optional[float] = None,
        leaderboard_rank: Optional[int] = None,
        submission_id: Optional[str] = None,
        source: str = "manual",
        notes: str = "",
        submission_target: str = "champion",
        submission_binding: Optional[Dict[str, Any]] = None,
    ) -> LeaderboardFeedbackResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        gate = self._read_json(self.competition_dir / "submission_gate.json")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        submission_policy = self._read_json(self.competition_dir / "submission_policy.json")
        submit_result = self._read_json(self.competition_dir / "kaggle_submit_result.json")
        gate_target = gate.get("submission_target", "champion")
        candidate = gate.get("candidate") or {}
        if not candidate:
            candidate = (
                submission_policy.get("recommended_submission_candidate")
                if submission_target == "recommended"
                else champion_selection.get("champion")
            ) or {}
        submission_path = self.competition_dir / (
            "recommended_submission.csv" if submission_target == "recommended" else "champion_submission.csv"
        )
        issues = []
        warnings = []
        submission_binding = self._normalize_submission_binding(submission_binding)

        if public_score is None and private_score is None and leaderboard_rank is None and not submission_id:
            issues.append("At least one leaderboard signal is required: score, rank, or submission_id.")
        for field_name, value in {
            "submission_id": submission_id,
            "source": source,
            "notes": notes,
        }.items():
            if self._looks_like_placeholder(value):
                issues.append(f"{field_name} still contains an unreplaced placeholder.")
        if gate and gate.get("status") != "pass":
            warnings.append("submission_gate.json is present but not pass.")
        if gate and gate_target != submission_target:
            warnings.append(f"submission_gate.json target is {gate_target}, but feedback target is {submission_target}.")
        if leaderboard_rank is not None and leaderboard_rank <= 0:
            issues.append("leaderboard_rank must be positive.")

        feedback = {
            "competition_name": manifest.competition_name,
            "status": "pass" if not issues else "needs_review",
            "source": source,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "submission_target": submission_target,
            "submission_id": submission_id,
            "public_score": public_score,
            "private_score": private_score,
            "leaderboard_rank": leaderboard_rank,
            "local_score": candidate.get("local_score"),
            "metric_name": candidate.get("metric_name"),
            "candidate_task_id": candidate.get("task_id"),
            "candidate": candidate,
            "submission_binding": submission_binding,
            "expected_submission_sha256": submission_binding.get("expected_submission_sha256"),
            "expected_submission_rows": submission_binding.get("expected_submission_rows"),
            "expected_submission_columns": submission_binding.get("expected_submission_columns"),
            "candidate_risk_level": submission_binding.get("candidate_risk_level"),
            "champion_task_id": (champion_selection.get("champion") or {}).get("task_id"),
            "champion_submission_path": str(self.competition_dir / "champion_submission.csv"),
            "recommended_submission_path": str(self.competition_dir / "recommended_submission.csv"),
            "submission_path": str(submission_path),
            "submit_result_status": submit_result.get("status"),
            "issues": issues,
            "warnings": warnings,
            "notes": notes,
            "next_action": (
                "Feed this leaderboard signal into the next Brain review."
                if not issues
                else "Add at least one valid leaderboard signal and record feedback again."
            ),
        }
        feedback_path = self.competition_dir / "leaderboard_feedback.json"
        feedback_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
        status = feedback["status"]
        ledger_entry = self.ledger.create_entry(
            task_id="leaderboard_feedback",
            agent="leaderboard_feedback_recorder",
            title="Record leaderboard feedback",
            status=status,
            input_payload=feedback,
            prompt="Record public leaderboard feedback so future Brain reviews can compare local CV and leaderboard behavior.",
            scorecard={
                "agent": "leaderboard_feedback_recorder",
                "task_id": "leaderboard_feedback",
                "status": status,
                "scores": {
                    "public_score": public_score if public_score is not None else "n/a",
                    "private_score": private_score if private_score is not None else "n/a",
                    "leaderboard_rank": leaderboard_rank if leaderboard_rank is not None else "n/a",
                    "submission_id": submission_id or "n/a",
                    "submission_target": submission_target,
                    "candidate_task_id": candidate.get("task_id") or "n/a",
                    "expected_submission_sha256": submission_binding.get("expected_submission_sha256") or "n/a",
                    "candidate_risk_level": submission_binding.get("candidate_risk_level") or "n/a",
                    "source": source,
                },
                "metric_name": feedback["metric_name"],
                "local_score": feedback["local_score"],
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"leaderboard_feedback": feedback_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="leaderboard_feedback",
                status=status,
                metric_name=feedback["metric_name"],
                local_score=feedback["local_score"],
                public_score=public_score,
                leaderboard_rank=leaderboard_rank,
                submission_path=feedback["submission_path"],
                brain_review_path=str(feedback_path),
                artifacts=[str(feedback_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=feedback["next_action"],
            )
        )
        return LeaderboardFeedbackResult(status=status, feedback_path=feedback_path)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _looks_like_placeholder(value: Optional[str]) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return (
            stripped.startswith("<")
            and stripped.endswith(">")
            or "paste_" in stripped.lower()
            or stripped in {"PUBLIC_SCORE", "SUBMISSION_ID", "SUBMISSION_ID_OR_NULL"}
        )

    @staticmethod
    def _normalize_submission_binding(submission_binding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(submission_binding, dict):
            return {}
        allowed = {
            "expected_submission_sha256",
            "expected_submission_rows",
            "expected_submission_columns",
            "candidate_risk_level",
        }
        return {key: submission_binding.get(key) for key in allowed if submission_binding.get(key) is not None}
