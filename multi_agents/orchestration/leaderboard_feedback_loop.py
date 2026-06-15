from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .experiment_queue import ExperimentQueueBuilder
from .leaderboard_feedback import LeaderboardFeedbackRecorder
from .leaderboard_gap_auditor import LeaderboardGapAuditor
from .memory import CompetitionMemory, ExperimentRecord
from .remote_brain import RemoteBrainReviewer
from .run_ledger import RunLedger


@dataclass(frozen=True)
class LeaderboardFeedbackLoopResult:
    status: str
    report_path: Path
    feedback_path: Path
    gap_audit_path: Optional[Path]
    brain_plan_path: Optional[Path]
    experiment_queue_path: Optional[Path] = None
    experiment_roadmap_path: Optional[Path] = None


class LeaderboardFeedbackLoop:
    """Record leaderboard feedback and immediately convert it into next-step evidence."""

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
        public_score: Optional[float] = None,
        private_score: Optional[float] = None,
        leaderboard_rank: Optional[int] = None,
        submission_id: Optional[str] = None,
        source: str = "manual",
        notes: str = "",
        submission_target: str = "champion",
        brain_review: bool = True,
        brain_use_llm: bool = True,
        submission_binding: Optional[Dict[str, Any]] = None,
    ) -> LeaderboardFeedbackLoopResult:
        feedback_result = LeaderboardFeedbackRecorder(self.competition_dir, memory=self.memory).record(
            public_score=public_score,
            private_score=private_score,
            leaderboard_rank=leaderboard_rank,
            submission_id=submission_id,
            source=source,
            notes=notes,
            submission_target=submission_target,
            submission_binding=submission_binding,
        )
        feedback = self._read_json(feedback_result.feedback_path)
        issues = list(feedback.get("issues", []))
        warnings = list(feedback.get("warnings", []))
        gap_result = None
        gap_audit = {}
        brain_result = None
        brain_plan = {}
        queue_result = None
        experiment_queue = {}
        roadmap_result = None

        if feedback.get("status") == "pass":
            gap_result = LeaderboardGapAuditor(self.competition_dir, memory=self.memory).audit()
            gap_audit = self._read_json(gap_result.audit_path)
            if gap_audit.get("status") != "completed":
                issues.append("leaderboard_gap_audit.json did not complete.")
            if brain_review:
                brain_result = RemoteBrainReviewer(
                    self.competition_dir,
                    memory=self.memory,
                    use_llm=brain_use_llm,
                ).review()
                brain_plan = self._read_json(brain_result.json_path)
                if brain_plan.get("recommended_experiments"):
                    queue_result = ExperimentQueueBuilder(self.competition_dir, memory=self.memory).build()
                    experiment_queue = self._read_json(queue_result.queue_path)
        else:
            warnings.append("Skipped gap audit and Brain review because leaderboard feedback did not pass validation.")

        gap_risk = gap_audit.get("risk_level")
        recommended = brain_plan.get("recommended_experiments") or []
        next_runnable = experiment_queue.get("next_runnable") or {}
        if queue_result and not next_runnable:
            warnings.append("Experiment queue was generated, but it has no next_runnable item.")
        if issues:
            status = "needs_review"
            decision = "feedback_loop_blocked"
            next_action = "Fix leaderboard feedback inputs, then rerun --leaderboard-feedback-loop."
        elif queue_result and not next_runnable:
            status = "needs_review"
            decision = "feedback_queue_needs_replan"
            next_action = "Remote Brain produced no runnable queue item; rerun Brain review or patch the plan."
        elif gap_risk == "high":
            status = "needs_review"
            decision = "stability_first_required"
            next_action = (
                f"Run queued audit/experiment: {next_runnable.get('task_id')}"
                if next_runnable
                else "Run stability-first validation before making another leaderboard submission."
            )
        else:
            status = "pass"
            decision = "ready_for_next_experiment"
            next_action = (
                f"Run queued experiment: {next_runnable.get('task_id')}"
                if next_runnable
                else f"Run next recommended experiment: {recommended[0].get('task_id')}"
                if recommended and isinstance(recommended[0], dict)
                else "Ask Remote Brain for the next experiment plan."
            )

        report = {
            "competition_name": feedback.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": decision,
            "submission_target": feedback.get("submission_target", submission_target),
            "candidate_task_id": feedback.get("candidate_task_id"),
            "submission_binding": feedback.get("submission_binding", {}),
            "expected_submission_sha256": feedback.get("expected_submission_sha256"),
            "expected_submission_rows": feedback.get("expected_submission_rows"),
            "expected_submission_columns": feedback.get("expected_submission_columns"),
            "candidate_risk_level": feedback.get("candidate_risk_level"),
            "leaderboard_feedback_status": feedback.get("status"),
            "feedback_path": str(feedback_result.feedback_path),
            "gap_audit_status": gap_audit.get("status"),
            "gap_audit_path": str(gap_result.audit_path) if gap_result else None,
            "gap_risk_level": gap_risk,
            "score_gap": gap_audit.get("score_gap", {}),
            "brain_review_enabled": brain_review,
            "brain_review_used_llm": brain_result.used_llm if brain_result else False,
            "brain_plan_path": str(brain_result.json_path) if brain_result else None,
            "next_recommended_experiments": recommended,
            "experiment_queue_status": experiment_queue.get("status"),
            "experiment_queue_path": str(queue_result.queue_path) if queue_result else None,
            "experiment_roadmap_path": None,
            "next_runnable": next_runnable or None,
            "next_command": next_runnable.get("next_command") if next_runnable else None,
            "issues": issues,
            "warnings": warnings,
            "next_action": next_action,
        }
        report_path = self.competition_dir / "leaderboard_feedback_loop.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        from .experiment_roadmap import ExperimentRoadmapBuilder

        roadmap_result = ExperimentRoadmapBuilder(self.competition_dir, memory=self.memory).build()
        report["experiment_roadmap_path"] = str(roadmap_result.roadmap_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="leaderboard_feedback_loop",
            agent="leaderboard_feedback_loop",
            title="Close leaderboard feedback loop",
            status=status,
            input_payload=report,
            prompt="Record leaderboard feedback, audit public-vs-CV gap, and convert it into the next experiment decision.",
            scorecard={
                "agent": "leaderboard_feedback_loop",
                "task_id": "leaderboard_feedback_loop",
                "status": status,
                "scores": {
                    "public_score": feedback.get("public_score", "n/a"),
                    "leaderboard_rank": feedback.get("leaderboard_rank", "n/a"),
                    "submission_target": feedback.get("submission_target", submission_target),
                    "candidate_task_id": feedback.get("candidate_task_id", "n/a"),
                    "expected_submission_sha256": feedback.get("expected_submission_sha256") or "n/a",
                    "candidate_risk_level": feedback.get("candidate_risk_level") or "n/a",
                    "gap_risk_level": gap_risk or "n/a",
                    "brain_review_used_llm": report["brain_review_used_llm"],
                    "recommended_experiment_count": len(recommended),
                    "experiment_queue_status": experiment_queue.get("status", "n/a"),
                    "next_runnable": next_runnable.get("task_id", "n/a"),
                },
                "metric_name": feedback.get("metric_name"),
                "local_score": feedback.get("local_score"),
                "issues": issues + warnings + gap_audit.get("issues", []),
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "leaderboard_feedback_loop": report_path,
                "leaderboard_feedback": feedback_result.feedback_path,
                "leaderboard_gap_audit": gap_result.audit_path if gap_result else Path(""),
                "brain_plan": brain_result.json_path if brain_result else Path(""),
                "experiment_queue": queue_result.queue_path if queue_result else Path(""),
                "experiment_roadmap": roadmap_result.roadmap_path if roadmap_result else Path(""),
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="leaderboard_feedback_loop",
                status=status,
                metric_name=feedback.get("metric_name"),
                local_score=feedback.get("local_score"),
                public_score=feedback.get("public_score") if isinstance(feedback.get("public_score"), (int, float)) else None,
                leaderboard_rank=feedback.get("leaderboard_rank") if isinstance(feedback.get("leaderboard_rank"), int) else None,
                submission_path=feedback.get("submission_path"),
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=next_action,
            )
        )
        return LeaderboardFeedbackLoopResult(
            status=status,
            report_path=report_path,
            feedback_path=feedback_result.feedback_path,
            gap_audit_path=gap_result.audit_path if gap_result else None,
            brain_plan_path=brain_result.json_path if brain_result else None,
            experiment_queue_path=queue_result.queue_path if queue_result else None,
            experiment_roadmap_path=roadmap_result.roadmap_path if roadmap_result else None,
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
