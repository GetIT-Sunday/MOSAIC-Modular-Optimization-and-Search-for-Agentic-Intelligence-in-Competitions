from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class ExperimentQueueResult:
    status: str
    queue_path: Path
    markdown_path: Path


class ExperimentQueueBuilder:
    """Turn Remote Brain recommendations into an auditable execution queue."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def build(self) -> ExperimentQueueResult:
        plan_path = self.competition_dir / "llm_experiment_plan.json"
        plan = self._read_json(plan_path)
        existing_queue = self._read_json(self.competition_dir / "experiment_queue.json")
        recommendations = plan.get("recommended_experiments") or existing_queue.get("queue") or []
        issues = []
        if not recommendations:
            issues.append("llm_experiment_plan.json has no recommended_experiments.")

        items = [
            self._queue_item(item, index)
            for index, item in enumerate(recommendations, start=1)
            if isinstance(item, dict)
        ]
        status = "ready" if items and not issues else "needs_review"
        queue = {
            "competition_name": self.competition_dir.name,
            "status": status,
            "source_plan_path": str(plan_path),
            "next_action": plan.get("next_action"),
            "leaderboard_diagnosis": plan.get("leaderboard_diagnosis", {}),
            "queue": items,
            "next_runnable": self._next_runnable(items),
            "issues": issues,
            "warnings": self._warnings(items),
        }
        queue_path = self.competition_dir / "experiment_queue.json"
        markdown_path = self.competition_dir / "experiment_queue.md"
        queue_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(self._render_markdown(queue), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="experiment_queue",
            agent="experiment_queue",
            title="Build Brain recommendation queue",
            status=status,
            input_payload=queue,
            prompt="Convert Remote Brain recommended experiments into a visible execution queue with status and action type.",
            scorecard={
                "agent": "experiment_queue",
                "task_id": "experiment_queue",
                "status": status,
                "scores": {
                    "queued_items": len(items),
                    "pending_items": sum(1 for item in items if item["status"] == "pending"),
                    "completed_items": sum(1 for item in items if item["status"] == "completed"),
                    "manual_submit_items": sum(1 for item in items if item["action_type"] == "manual_submit"),
                    "next_runnable": (queue["next_runnable"] or {}).get("task_id", "n/a"),
                },
                "metric_name": None,
                "local_score": None,
                "issues": issues + queue["warnings"],
                "recommended_human_action": "continue" if status == "ready" else "patch_prompt",
            },
            artifacts={
                "experiment_queue": queue_path,
                "experiment_queue_markdown": markdown_path,
                "llm_experiment_plan": plan_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=self.competition_dir.name,
                profile_name="tabular_classic",
                task_id="experiment_queue",
                status=status,
                brain_review_path=str(queue_path),
                artifacts=[str(queue_path), str(markdown_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=f"Queued {len(items)} Remote Brain recommendation(s).",
            )
        )
        return ExperimentQueueResult(status=status, queue_path=queue_path, markdown_path=markdown_path)

    def _queue_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        task_id = self._task_id(item, index)
        experiment_dir = self.competition_dir / "experiments" / task_id
        validation_report = experiment_dir / "validation_report.json"
        action_type = self._action_type(item)
        report = self._read_json(validation_report) if validation_report.exists() else {}
        submission_review = self._read_json(self.competition_dir / "submission_decision_review.json")
        report_status = report.get("status")
        status = "pending"
        if report_status == "completed":
            status = "completed"
        elif report_status:
            status = "blocked"
        if action_type == "manual_submit" and status == "pending":
            if submission_review.get("queue_task_id") == task_id and submission_review.get("decision") == "pause_manual_submit":
                status = "blocked"
            else:
                status = "manual_gate"
        return {
            "order": index,
            "task_id": task_id,
            "title": item.get("title") or task_id,
            "status": status,
            "action_type": action_type,
            "skill_used": item.get("skill_used", "unknown"),
            "harness": item.get("harness", "unknown"),
            "hypothesis": item.get("hypothesis", ""),
            "runner_kind": item.get("runner_kind"),
            "expected_gain": item.get("expected_gain", "unknown"),
            "risk": item.get("risk", "unknown"),
            "compute_cost": item.get("compute_cost", "unknown"),
            "validation_plan": item.get("validation_plan", ""),
            "evidence_needed": item.get("evidence_needed") if isinstance(item.get("evidence_needed"), list) else [],
            "promotion_gate": item.get("promotion_gate") if isinstance(item.get("promotion_gate"), dict) else {},
            "coding_agent_task": item.get("coding_agent_task", ""),
            "experiment_dir": str(experiment_dir),
            "report_status": report_status,
            "submission_decision": submission_review.get("decision") if submission_review.get("queue_task_id") == task_id else None,
            "failure_reason": "; ".join(report.get("issues", [])) if isinstance(report.get("issues"), list) else None,
            "validation_report": str(validation_report) if validation_report.exists() else None,
            "next_command": self._next_command(task_id, action_type),
        }

    def _next_command(self, task_id: str, action_type: str) -> str:
        if action_type == "manual_submit":
            return "python framework.py --competition {competition} --post-submit-workflow --submission-target champion"
        return "python framework.py --competition {competition} --run-enhancement"

    def _action_type(self, item: Dict[str, Any]) -> str:
        runner_kind = str(item.get("runner_kind") or "").lower()
        if runner_kind in {
            "cv_stability_audit",
            "distribution_shift_audit",
            "overfitting_audit",
        }:
            return "audit"
        identity = " ".join(str(item.get(key, "")) for key in ["task_id", "title"]).lower()
        full_text = " ".join(str(item.get(key, "")) for key in ["task_id", "title", "coding_agent_task"]).lower()
        if "submit" in identity or ("leaderboard" in identity and "kaggle" in full_text):
            return "manual_submit"
        if "audit" in identity or "stability_audit" in identity or "cv_stability" in identity:
            return "audit"
        return "coding_experiment"

    @staticmethod
    def _next_runnable(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for item in items:
            if item["status"] == "pending" and item["action_type"] != "manual_submit":
                return item
        for item in items:
            if item["status"] == "manual_gate":
                return item
        return None

    @staticmethod
    def _warnings(items: List[Dict[str, Any]]) -> List[str]:
        warnings = []
        if any(item["action_type"] == "manual_submit" for item in items):
            warnings.append("Queue contains manual_submit items; require explicit human submission workflow before treating leaderboard feedback as evidence.")
        if not any(item["status"] == "pending" for item in items):
            warnings.append("No pending coding experiments are available.")
        return warnings

    @staticmethod
    def _task_id(item: Dict[str, Any], index: int) -> str:
        raw = item.get("task_id") or item.get("experiment_id") or f"brain_recommendation_{index}"
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(raw))
        return safe.strip("_") or f"brain_recommendation_{index}"

    @staticmethod
    def _render_markdown(queue: Dict[str, Any]) -> str:
        lines = [
            "# Experiment Queue",
            "",
            f"Status: {queue.get('status')}",
            f"Next action: {queue.get('next_action')}",
            "",
            "## Items",
        ]
        for item in queue.get("queue", []):
            lines.extend(
                [
                    "",
                    f"### {item['order']}. {item['task_id']}",
                    "",
                    f"- Status: {item['status']}",
                    f"- Action type: {item['action_type']}",
                    f"- Skill: {item.get('skill_used') or 'unknown'}",
                    f"- Harness: {item.get('harness') or 'unknown'}",
                    f"- Hypothesis: {item.get('hypothesis') or 'n/a'}",
                    f"- Runner kind: {item.get('runner_kind') or 'unknown'}",
                    f"- Expected gain: {item['expected_gain']}",
                    f"- Risk: {item['risk']}",
                    f"- Compute cost: {item['compute_cost']}",
                    f"- Validation plan: {item.get('validation_plan') or 'n/a'}",
                    f"- Evidence needed: {', '.join(item.get('evidence_needed') or [])}",
                    f"- Promotion gate: `{json.dumps(item.get('promotion_gate') or {}, ensure_ascii=False)}`",
                    f"- Next command: `{item['next_command']}`",
                    "",
                    str(item.get("coding_agent_task", "")),
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
