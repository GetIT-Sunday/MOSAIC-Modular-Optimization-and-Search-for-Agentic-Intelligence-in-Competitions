from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .manual_submit_readiness import ManualSubmitReadinessChecker
from .memory import CompetitionMemory, ExperimentRecord
from .promotion_gate import PromotionGateEvaluator
from .remote_brain import RemoteBrainReviewer
from .run_ledger import RunLedger
from .submission_gate import SubmissionGate
from .submission_policy import SubmissionPolicy
from .submit_decision_handoff import SubmitDecisionHandoff
from .post_submit_workflow import PostSubmitWorkflow


@dataclass(frozen=True)
class PostExperimentPipelineResult:
    status: str
    report_path: Path


class PostExperimentPipeline:
    """Run the standard review chain after an experiment finishes."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self, *, submission_target: str = "recommended") -> PostExperimentPipelineResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")

        plan_refresh = RemoteBrainReviewer(
            self.competition_dir,
            memory=self.memory,
            use_llm=False,
        ).refresh_existing_plan_gates()
        promotion_result = PromotionGateEvaluator(self.competition_dir, memory=self.memory).evaluate()
        promotion = self._read_json(promotion_result.review_path)
        issues = []
        warnings = []
        downstream: Dict[str, Any] = {}

        if promotion.get("decision") != "promote_candidate":
            warnings.append("promotion_gate_review.json did not promote the latest planned candidate; falling back to champion-based submission policy.")

        policy_result = SubmissionPolicy(self.competition_dir, memory=self.memory).run()
        policy = self._read_json(policy_result.policy_path)
        gate_result = SubmissionGate(self.competition_dir, memory=self.memory).run(
            dry_run=True,
            submission_target=submission_target,
        )
        gate = self._read_json(gate_result.gate_path)
        readiness_result = ManualSubmitReadinessChecker(
            self.competition_dir,
            memory=self.memory,
        ).run(refresh=True, submission_target=submission_target)
        readiness = self._read_json(readiness_result.report_path)
        handoff_result = SubmitDecisionHandoff(
            self.competition_dir,
            memory=self.memory,
        ).run(refresh=False, submission_target=submission_target)
        handoff = self._read_json(handoff_result.report_path)
        workflow_result = PostSubmitWorkflow(
            self.competition_dir,
            memory=self.memory,
        ).run(refresh=False, submission_target=submission_target)
        workflow = self._read_json(workflow_result.report_path)
        downstream = {
            "submission_policy_status": policy.get("status"),
            "submission_policy_decision": policy.get("decision"),
            "submission_policy_path": str(policy_result.policy_path),
            "submission_gate_status": gate.get("status"),
            "submission_gate_path": str(gate_result.gate_path),
            "manual_submit_readiness_status": readiness.get("status"),
            "manual_submit_readiness_path": str(readiness_result.report_path),
            "submit_decision_handoff_status": handoff.get("status"),
            "submit_decision_handoff_path": str(handoff_result.report_path),
            "post_submit_workflow_status": workflow.get("status"),
            "post_submit_workflow_path": str(workflow_result.report_path),
            "feedback_input_template_path": str(workflow_result.feedback_input_template_path),
            "recommended_submission_path": policy.get("recommended_submission_path"),
            "recommended_submission_candidate": policy.get("recommended_submission_candidate"),
        }
        for label, payload in [
            ("submission_policy", policy),
            ("submission_gate", gate),
            ("manual_submit_readiness", readiness),
            ("submit_decision_handoff", handoff),
            ("post_submit_workflow", workflow),
        ]:
            if payload.get("status") not in {"pass", "manual_submit_ready", "ready_for_human_submit_decision", "ready_for_manual_submit"}:
                issues.append(f"{label} status is {payload.get('status') or 'missing'}.")
            warnings.extend(payload.get("warnings") or [])
        status = "pass" if not issues else "needs_review"
        decision = "ready_for_human_submit_decision" if status == "pass" else "downstream_gate_blocked"
        next_action = (
            "Review submit_decision_handoff.md, upload the recommended submission if approved, then fill leaderboard_feedback_input_template.json."
            if status == "pass"
            else "Fix downstream gate issues before any leaderboard upload."
        )

        promoted = promotion.get("promoted_candidate") or downstream.get("recommended_submission_candidate") or {}
        report = {
            "competition_name": promotion.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": decision,
            "submission_target": submission_target,
            "promoted_candidate": promoted,
            "promotion_gate_status": promotion.get("status"),
            "promotion_gate_decision": promotion.get("decision"),
            "promotion_gate_path": str(promotion_result.review_path),
            "plan_gate_refresh_path": str(plan_refresh.json_path),
            "downstream": downstream,
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "next_action": next_action,
        }
        report_path = self.competition_dir / "post_experiment_pipeline.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        ledger_entry = self.ledger.create_entry(
            task_id="post_experiment_pipeline",
            agent="post_experiment_pipeline",
            title="Run post-experiment promotion and submit gates",
            status=status,
            input_payload=report,
            prompt=(
                "After the latest experiment, run promotion review, recommended-submission policy, "
                "submission gate, manual readiness, and submit-decision handoff without submitting."
            ),
            scorecard={
                "agent": "post_experiment_pipeline",
                "task_id": "post_experiment_pipeline",
                "status": status,
                "scores": {
                    "submission_target": submission_target,
                    "promotion_gate": promotion.get("decision", "missing"),
                    "submission_policy": downstream.get("submission_policy_status", "skipped"),
                    "submission_gate": downstream.get("submission_gate_status", "skipped"),
                    "readiness": downstream.get("manual_submit_readiness_status", "skipped"),
                    "handoff": downstream.get("submit_decision_handoff_status", "skipped"),
                    "post_submit_workflow": downstream.get("post_submit_workflow_status", "skipped"),
                },
                "metric_name": promoted.get("metric_name"),
                "local_score": promoted.get("local_score"),
                "issues": issues + list(dict.fromkeys(warnings)),
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "post_experiment_pipeline": report_path,
                "plan_gate_refresh": plan_refresh.json_path,
                "promotion_gate_review": promotion_result.review_path,
                "promotion_gate_markdown": promotion_result.markdown_path,
                "promoted_submission": promotion_result.promoted_submission_path or Path(""),
                "submission_policy": self.competition_dir / "submission_policy.json",
                "submission_gate": self.competition_dir / "submission_gate.json",
                "manual_submit_readiness": self.competition_dir / "manual_submit_readiness.json",
                "submit_decision_handoff": self.competition_dir / "submit_decision_handoff.json",
                "post_submit_workflow": self.competition_dir / "post_submit_workflow.json",
                "leaderboard_feedback_input_template": self.competition_dir / "leaderboard_feedback_input_template.json",
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="post_experiment_pipeline",
                status=status,
                metric_name=promoted.get("metric_name"),
                local_score=promoted.get("local_score"),
                submission_path=downstream.get("recommended_submission_path"),
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=next_action,
            )
        )
        return PostExperimentPipelineResult(status=status, report_path=report_path)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
