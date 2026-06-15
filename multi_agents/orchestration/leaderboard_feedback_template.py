from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .leaderboard_feedback import LeaderboardFeedbackRecorder
from .leaderboard_feedback_input import LeaderboardFeedbackInputRunner
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class LeaderboardFeedbackTemplateFillResult:
    status: str
    report_path: Path
    filled_template_path: Path
    feedback_loop_report_path: Optional[Path]
    experiment_roadmap_path: Optional[Path] = None


class LeaderboardFeedbackTemplateFiller:
    """Fill the packaged leaderboard feedback template from Kaggle-returned values."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def fill(
        self,
        *,
        template_path: Optional[Path] = None,
        public_score: Optional[float] = None,
        private_score: Optional[float] = None,
        leaderboard_rank: Optional[int] = None,
        submission_id: Optional[str] = None,
        source: str = "manual",
        notes: str = "post_submit_feedback",
        run_feedback_loop: bool = False,
        brain_use_llm: bool = True,
    ) -> LeaderboardFeedbackTemplateFillResult:
        template_path = template_path or self.competition_dir / "manual_submission_package" / "leaderboard_feedback_input_template.json"
        if not template_path.is_absolute():
            template_path = self.competition_dir / template_path
        payload = self._read_json(template_path)
        issues: list[str] = []
        warnings: list[str] = []
        submission_summary = self._packaged_submission_summary(template_path)

        if not payload:
            issues.append("Leaderboard feedback template is missing or empty.")
        else:
            self._validate_packaged_submission_binding(payload, submission_summary, issues, warnings)
        if public_score is None and private_score is None and leaderboard_rank is None and not submission_id:
            issues.append("At least one leaderboard signal is required: public_score, private_score, leaderboard_rank, or submission_id.")
        if leaderboard_rank is not None and leaderboard_rank <= 0:
            issues.append("leaderboard_rank must be positive when provided.")
        if LeaderboardFeedbackRecorder._looks_like_placeholder(source):
            issues.append("source still contains an unreplaced placeholder.")
        if LeaderboardFeedbackRecorder._looks_like_placeholder(notes):
            issues.append("notes still contains an unreplaced placeholder.")
        if submission_id and LeaderboardFeedbackRecorder._looks_like_placeholder(submission_id):
            issues.append("submission_id still contains an unreplaced placeholder.")

        filled = dict(payload)
        if not issues:
            filled.update(
                {
                    "public_score": public_score,
                    "private_score": private_score,
                    "leaderboard_rank": leaderboard_rank,
                    "submission_id": submission_id,
                    "source": source,
                    "notes": notes,
                }
            )
            template_path.write_text(json.dumps(filled, indent=2, ensure_ascii=False), encoding="utf-8")

        loop_result = None
        if run_feedback_loop and not issues:
            loop_result = LeaderboardFeedbackInputRunner(self.competition_dir, memory=self.memory).run(
                input_path=template_path,
                brain_use_llm=brain_use_llm,
            )
            if loop_result.status != "pass":
                warnings.append(f"leaderboard_feedback_input status is {loop_result.status}.")

        status = "needs_review" if issues else (loop_result.status if loop_result else "filled")
        report = {
            "competition_name": self.competition_dir.name,
            "status": status,
            "decision": "feedback_template_filled" if not issues else "feedback_template_fill_blocked",
            "template_path": str(template_path),
            "run_feedback_loop": run_feedback_loop,
            "feedback_loop_report_path": str(loop_result.report_path) if loop_result else None,
            "experiment_roadmap_path": str(loop_result.experiment_roadmap_path) if loop_result and loop_result.experiment_roadmap_path else None,
            "public_score": public_score,
            "private_score": private_score,
            "leaderboard_rank": leaderboard_rank,
            "submission_id": submission_id,
            "submission_target": filled.get("submission_target"),
            "candidate_task_id": filled.get("candidate_task_id"),
            "expected_submission_sha256": filled.get("expected_submission_sha256"),
            "expected_submission_rows": filled.get("expected_submission_rows"),
            "expected_submission_columns": filled.get("expected_submission_columns"),
            "packaged_submission_file": submission_summary,
            "candidate_risk_level": filled.get("candidate_risk_level"),
            "source": source,
            "notes": notes,
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Inspect leaderboard_feedback_loop.json and refreshed roadmap."
                if loop_result
                else "Run --leaderboard-feedback-from-template with this filled template."
                if not issues
                else "Provide at least one real Kaggle leaderboard signal and rerun the fill command."
            ),
        }
        report_path = self.competition_dir / "leaderboard_feedback_template_fill.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="leaderboard_feedback_template_fill",
            agent="leaderboard_feedback_template_filler",
            title="Fill leaderboard feedback template",
            status=status,
            input_payload=report,
            prompt="Fill the packaged leaderboard feedback JSON from Kaggle-returned public score, rank, or submission id.",
            scorecard={
                "agent": "leaderboard_feedback_template_filler",
                "task_id": "leaderboard_feedback_template_fill",
                "status": status,
                "scores": {
                    "public_score": public_score if public_score is not None else "n/a",
                    "leaderboard_rank": leaderboard_rank if leaderboard_rank is not None else "n/a",
                    "submission_id": submission_id or "n/a",
                    "candidate_task_id": filled.get("candidate_task_id") or "n/a",
                    "expected_submission_sha256": filled.get("expected_submission_sha256") or "n/a",
                    "feedback_loop_started": loop_result is not None,
                },
                "metric_name": None,
                "local_score": None,
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status in {"filled", "pass"} else "patch_prompt",
            },
            artifacts={
                "feedback_template_fill": report_path,
                "filled_feedback_template": template_path,
                "leaderboard_feedback_loop": loop_result.report_path if loop_result else Path(""),
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="leaderboard_feedback_template_fill",
                status=status,
                public_score=public_score,
                leaderboard_rank=leaderboard_rank,
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(template_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=report["next_action"],
            )
        )
        return LeaderboardFeedbackTemplateFillResult(
            status=status,
            report_path=report_path,
            filled_template_path=template_path,
            feedback_loop_report_path=loop_result.report_path if loop_result else None,
            experiment_roadmap_path=loop_result.experiment_roadmap_path if loop_result else None,
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def _packaged_submission_summary(cls, template_path: Path) -> Dict[str, Any]:
        submission_path = template_path.parent / "submission.csv"
        return cls._file_summary(submission_path)

    @staticmethod
    def _validate_packaged_submission_binding(
        payload: Dict[str, Any],
        submission_summary: Dict[str, Any],
        issues: list[str],
        warnings: list[str],
    ) -> None:
        if not submission_summary.get("exists"):
            warnings.append("Packaged submission.csv was not found next to the feedback template; file binding could not be rechecked.")
            return
        checks = [
            ("expected_submission_sha256", "sha256", "submission SHA-256"),
            ("expected_submission_rows", "row_count", "submission row count"),
            ("expected_submission_columns", "columns", "submission columns"),
        ]
        for expected_key, actual_key, label in checks:
            expected = payload.get(expected_key)
            actual = submission_summary.get(actual_key)
            if expected is None:
                warnings.append(f"{expected_key} is missing from feedback template; {label} binding could not be rechecked.")
            elif actual != expected:
                issues.append(f"{label} mismatch: template={expected}, actual={actual}.")

    @staticmethod
    def _file_summary(path: Path) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            return summary
        content = path.read_bytes()
        summary.update(
            {
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            header = rows[0] if rows else []
            summary.update(
                {
                    "columns": header,
                    "column_count": len(header),
                    "row_count": max(len(rows) - 1, 0),
                }
            )
        return summary
