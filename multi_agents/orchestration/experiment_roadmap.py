from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .leaderboard_feedback_freshness import LeaderboardFeedbackFreshnessAuditor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class ExperimentRoadmapResult:
    status: str
    roadmap_path: Path
    markdown_path: Path


class ExperimentRoadmapBuilder:
    """Build a prioritized, human-readable next-action backlog for the competition loop."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def build(self) -> ExperimentRoadmapResult:
        context = self._context()
        items = self._roadmap_items(context)
        status = self._status(items)
        roadmap = {
            "competition_name": context["competition_name"],
            "status": status,
            "decision": self._decision(items),
            "current_candidate": context.get("current_candidate"),
            "manual_submission_package_verification": context.get("manual_package_verification", {}),
            "leaderboard_feedback_freshness": context.get("leaderboard_feedback_freshness", {}),
            "top_action": items[0] if items else None,
            "items": items,
            "context_paths": context["context_paths"],
            "issues": self._issues(items),
            "next_action": self._next_action(items),
        }
        roadmap_path = self.competition_dir / "experiment_roadmap.json"
        markdown_path = self.competition_dir / "experiment_roadmap.md"
        roadmap_path.write_text(json.dumps(roadmap, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(self._render_markdown(roadmap), encoding="utf-8")

        ledger_entry = self.ledger.create_entry(
            task_id="experiment_roadmap",
            agent="experiment_roadmap",
            title="Prioritize next AutoKaggle loop actions",
            status=status,
            input_payload=roadmap,
            prompt=(
                "Summarize the current competition state into a prioritized backlog that separates "
                "remote runnable work from human leaderboard feedback gates."
            ),
            scorecard={
                "agent": "experiment_roadmap",
                "task_id": "experiment_roadmap",
                "status": status,
                "scores": {
                    "items": len(items),
                    "ready_items": sum(1 for item in items if item["status"] == "ready"),
                    "waiting_for_human": sum(1 for item in items if item["status"] == "waiting_for_human"),
                    "top_action": (items[0] or {}).get("action_id") if items else "n/a",
                },
                "metric_name": (context.get("current_candidate") or {}).get("metric_name"),
                "local_score": (context.get("current_candidate") or {}).get("local_score"),
                "issues": roadmap["issues"],
                "recommended_human_action": "continue" if status != "blocked" else "patch_prompt",
            },
            artifacts={
                "experiment_roadmap": roadmap_path,
                "experiment_roadmap_markdown": markdown_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=context["competition_name"],
                profile_name="tabular_classic",
                task_id="experiment_roadmap",
                status=status,
                metric_name=(context.get("current_candidate") or {}).get("metric_name"),
                local_score=(context.get("current_candidate") or {}).get("local_score"),
                brain_review_path=str(roadmap_path),
                artifacts=[str(roadmap_path), str(markdown_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=roadmap["next_action"],
            )
        )
        return ExperimentRoadmapResult(status=status, roadmap_path=roadmap_path, markdown_path=markdown_path)

    def _context(self) -> Dict[str, Any]:
        manifest = self._read_json(self.competition_dir / "data_manifest.json")
        queue = self._read_json(self.competition_dir / "experiment_queue.json")
        brain_plan = self._read_json(self.competition_dir / "llm_experiment_plan.json")
        promotion = self._read_json(self.competition_dir / "promotion_gate_review.json")
        pipeline = self._read_json(self.competition_dir / "post_experiment_pipeline.json")
        package = self._read_json(self.competition_dir / "manual_submission_package" / "manifest.json")
        package_verification = self._read_json(self.competition_dir / "manual_submission_package_verification.json")
        leaderboard_feedback = self._read_json(self.competition_dir / "leaderboard_feedback.json")
        feedback_loop = self._read_json(self.competition_dir / "leaderboard_feedback_loop.json")
        current_candidate = (
            package.get("candidate")
            or pipeline.get("promoted_candidate")
            or promotion.get("promoted_candidate")
            or {}
        )
        feedback_freshness = LeaderboardFeedbackFreshnessAuditor(self.competition_dir).audit(
            leaderboard_feedback=leaderboard_feedback,
            manual_package=package,
            post_submit_workflow=self._read_json(self.competition_dir / "post_submit_workflow.json"),
        )
        return {
            "competition_name": manifest.get("competition_name", self.competition_dir.name),
            "manifest": manifest,
            "queue": queue,
            "brain_plan": brain_plan,
            "promotion": promotion,
            "pipeline": pipeline,
            "manual_package": package,
            "manual_package_verification": package_verification,
            "leaderboard_feedback": leaderboard_feedback,
            "feedback_loop": feedback_loop,
            "leaderboard_feedback_freshness": feedback_freshness,
            "current_candidate": current_candidate,
            "context_paths": {
                "experiment_queue": str(self.competition_dir / "experiment_queue.json"),
                "brain_plan": str(self.competition_dir / "llm_experiment_plan.json"),
                "promotion_gate": str(self.competition_dir / "promotion_gate_review.json"),
                "post_experiment_pipeline": str(self.competition_dir / "post_experiment_pipeline.json"),
                "manual_submission_package": str(self.competition_dir / "manual_submission_package" / "manifest.json"),
                "manual_submission_package_verification": str(self.competition_dir / "manual_submission_package_verification.json"),
                "leaderboard_feedback": str(self.competition_dir / "leaderboard_feedback.json"),
                "leaderboard_feedback_loop": str(self.competition_dir / "leaderboard_feedback_loop.json"),
            },
        }

    def _roadmap_items(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        package = context["manual_package"]
        leaderboard_feedback = context["leaderboard_feedback"]
        feedback_loop = context["feedback_loop"]
        feedback_freshness = context.get("leaderboard_feedback_freshness") or {}
        package_verification = context.get("manual_package_verification") or {}
        queue = context["queue"]
        top_queue = queue.get("next_runnable") if isinstance(queue.get("next_runnable"), dict) else {}
        package_status = package.get("status")
        current_task_id = (context.get("current_candidate") or {}).get("task_id")
        feedback_task_id = leaderboard_feedback.get("candidate_task_id")
        feedback_matches_candidate = bool(feedback_freshness.get("is_current"))
        package_verified = package_verification.get("status") == "pass"

        if package_status == "ready_for_manual_upload" and not feedback_matches_candidate and not package_verified:
            verification_status = package_verification.get("status") or "missing"
            verification_issues = package_verification.get("issues") or []
            issue_note = f" Current package verification status is {verification_status}."
            if verification_issues:
                issue_note += " Issues: " + "; ".join(str(issue) for issue in verification_issues)
            items.append(
                self._item(
                    priority=105,
                    action_id="verify_manual_submission_package",
                    title="Verify packaged submission before manual upload",
                    owner_agent="submit_gate",
                    action_type="remote_runnable",
                    status="ready",
                    rationale=(
                        "A manual upload package exists, but the package/file/template consistency check "
                        "has not passed yet."
                        + issue_note
                    ),
                    evidence_required=[
                        "manual_submission_package_verification.json",
                        "matching submission SHA-256, rows, columns, and feedback template bindings",
                    ],
                    next_command=f"python framework.py --competition {context['competition_name']} --verify-manual-submission-package",
                )
            )

        if package_status == "ready_for_manual_upload" and not feedback_matches_candidate and package_verified:
            mismatch_note = ""
            freshness_issues = feedback_freshness.get("issues") or []
            if leaderboard_feedback:
                mismatch_note = " Existing feedback is not current: " + "; ".join(str(item) for item in freshness_issues)
                if not freshness_issues and feedback_task_id != current_task_id:
                    mismatch_note = (
                        f" Existing feedback belongs to {feedback_task_id}, not current candidate {current_task_id}."
                    )
            items.append(
                self._item(
                    priority=100,
                    action_id="manual_upload_and_feedback_capture",
                    title="Upload packaged submission and capture public leaderboard feedback",
                    owner_agent="human_gate",
                    action_type="human_leaderboard_feedback",
                    status="waiting_for_human",
                    rationale=(
                        "A promoted candidate has passed local gates and the manual package is ready, "
                        "but no matching real public leaderboard feedback exists yet."
                        + mismatch_note
                    ),
                    evidence_required=[
                        "Kaggle public score, rank, or submission id",
                        "filled manual_submission_package/leaderboard_feedback_input_template.json",
                    ],
                    next_command=package.get("feedback_fill_command") or package.get("feedback_loop_command", ""),
                )
            )

        if feedback_matches_candidate and feedback_loop.get("status") != "pass":
            items.append(
                self._item(
                    priority=95,
                    action_id="run_leaderboard_feedback_loop",
                    title="Run leaderboard gap audit and refresh Brain plan",
                    owner_agent="remote_brain",
                    action_type="remote_runnable",
                    status="ready",
                    rationale="Leaderboard feedback exists, but the feedback loop has not produced a passing audit yet.",
                    evidence_required=[
                        "leaderboard_feedback_loop.json",
                        "leaderboard_gap_audit.json",
                        "refreshed llm_experiment_plan.json",
                    ],
                    next_command=f"python framework.py --competition {context['competition_name']} --leaderboard-feedback-loop",
                )
            )

        if top_queue and not self._queue_item_completed(top_queue):
            item_status = "ready" if top_queue.get("status") == "pending" else str(top_queue.get("status") or "ready")
            items.append(
                self._item(
                    priority=80,
                    action_id=f"execute_queue_{top_queue.get('task_id', 'next')}",
                    title=f"Execute queued Brain task: {top_queue.get('task_id', 'unknown')}",
                    owner_agent="coding_agent",
                    action_type="remote_runnable",
                    status=item_status,
                    rationale="Remote Brain has a next runnable queue item that should produce auditable experiment evidence.",
                    evidence_required=top_queue.get("evidence_needed") or ["validation_report.json", "validator_result.json"],
                    next_command=(top_queue.get("next_command") or "python framework.py --competition {competition} --run-enhancement").format(
                        competition=context["competition_name"]
                    ),
                )
            )

        if not top_queue or self._queue_item_completed(top_queue):
            items.append(
                self._item(
                    priority=70,
                    action_id="refresh_remote_brain_queue",
                    title="Refresh Remote Brain plan and rebuild experiment queue",
                    owner_agent="remote_brain",
                    action_type="remote_runnable",
                    status="ready",
                    rationale="The current experiment queue has no runnable coding experiment, so the Brain should re-plan from current evidence.",
                    evidence_required=["llm_experiment_plan.json", "experiment_queue.json"],
                    next_command=(
                        f"python framework.py --competition {context['competition_name']} "
                        "--remote-brain-review && "
                        f"python framework.py --competition {context['competition_name']} --experiment-queue"
                    ),
                )
            )

        items.append(
            self._item(
                priority=50,
                action_id="cross_competition_tabular_smoke",
                title="Run the same control loop on bank_churn to check tabular generalization",
                owner_agent="remote_brain",
                action_type="project_generalization",
                status="ready",
                rationale=(
                    "Titanic validates the loop, but silver-level AutoKaggle needs evidence that ingestion, "
                    "queueing, gates, and submissions are not Titanic-specific."
                ),
                evidence_required=[
                    "bank_churn data_manifest.json",
                    "bank_churn baseline_review.json",
                    "bank_churn experiment_roadmap.json",
                ],
                next_command=(
                    "python framework.py --competition bank_churn --task-card-mode && "
                    "python framework.py --competition bank_churn --run-baselines"
                ),
            )
        )

        return sorted(items, key=lambda item: item["priority"], reverse=True)

    @staticmethod
    def _item(
        *,
        priority: int,
        action_id: str,
        title: str,
        owner_agent: str,
        action_type: str,
        status: str,
        rationale: str,
        evidence_required: List[str],
        next_command: str,
    ) -> Dict[str, Any]:
        return {
            "priority": priority,
            "action_id": action_id,
            "title": title,
            "owner_agent": owner_agent,
            "action_type": action_type,
            "status": status,
            "rationale": rationale,
            "evidence_required": evidence_required,
            "next_command": next_command,
        }

    def _queue_item_completed(self, item: Dict[str, Any]) -> bool:
        if item.get("status") == "completed" or item.get("report_status") == "completed":
            return True
        report_paths = []
        experiment_dir = item.get("experiment_dir")
        if experiment_dir:
            report_paths.append(Path(str(experiment_dir)) / "validation_report.json")
        task_id = item.get("task_id")
        if task_id:
            report_paths.append(self.competition_dir / "experiments" / str(task_id) / "validation_report.json")
        for report_path in report_paths:
            if report_path.exists() and self._read_json(report_path).get("status") == "completed":
                return True
        return False

    @staticmethod
    def _status(items: List[Dict[str, Any]]) -> str:
        if any(item["status"] == "ready" for item in items):
            return "ready"
        if any(item["status"] == "waiting_for_human" for item in items):
            return "waiting_for_human"
        return "blocked"

    @staticmethod
    def _decision(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "no_actions_available"
        top = items[0]
        if top["status"] == "waiting_for_human":
            return "await_human_leaderboard_feedback"
        return f"run_{top['action_id']}"

    @staticmethod
    def _issues(items: List[Dict[str, Any]]) -> List[str]:
        issues = []
        if not any(item["status"] == "ready" for item in items):
            issues.append("No remote-runnable roadmap item is currently ready.")
        if any(item["status"] == "waiting_for_human" for item in items):
            issues.append("At least one high-priority item needs manual Kaggle feedback.")
        return issues

    @staticmethod
    def _next_action(items: List[Dict[str, Any]]) -> str:
        for item in items:
            if item["status"] == "ready":
                return item["next_command"] or item["title"]
        if items:
            return items[0]["next_command"] or items[0]["title"]
        return "Run Remote Brain review to create a new plan."

    @staticmethod
    def _render_markdown(roadmap: Dict[str, Any]) -> str:
        lines = [
            "# Experiment Roadmap",
            "",
            f"Status: {roadmap.get('status')}",
            f"Decision: {roadmap.get('decision')}",
            f"Next action: `{roadmap.get('next_action')}`",
            "",
            "## Items",
        ]
        for item in roadmap.get("items", []):
            lines.extend(
                [
                    "",
                    f"### P{item['priority']} {item['action_id']}",
                    "",
                    f"- Status: {item['status']}",
                    f"- Owner agent: {item['owner_agent']}",
                    f"- Action type: {item['action_type']}",
                    f"- Evidence required: {', '.join(item.get('evidence_required') or [])}",
                    f"- Next command: `{item.get('next_command') or 'n/a'}`",
                    "",
                    item.get("rationale", ""),
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
