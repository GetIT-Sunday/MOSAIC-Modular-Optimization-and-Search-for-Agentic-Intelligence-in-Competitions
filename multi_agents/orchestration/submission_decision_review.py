from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class SubmissionDecisionReviewResult:
    status: str
    review_path: Path
    markdown_path: Path


class SubmissionDecisionReviewer:
    """Review whether a manual leaderboard submission queue item should proceed."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def review(
        self,
        *,
        queue_task_id: str = "champion_blend_lb_submit",
        submission_target: str = "champion",
    ) -> SubmissionDecisionReviewResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        submission_policy = self._read_json(self.competition_dir / "submission_policy.json")
        leaderboard_feedback = self._read_json(self.competition_dir / "leaderboard_feedback.json")
        stability_audit = self._read_json(
            self.competition_dir / "experiments" / "cv_stability_audit_v1" / "cv_stability_audit.json"
        )
        queue = self._read_json(self.competition_dir / "experiment_queue.json")

        champion = champion_selection.get("champion") or {}
        recommended = submission_policy.get("recommended_submission_candidate") or {}
        candidate = champion if submission_target == "champion" else recommended
        issues = []
        warnings = []

        if not candidate:
            issues.append(f"No {submission_target} candidate is available.")
        if submission_target == "champion" and leaderboard_feedback.get("submission_target") == "recommended":
            warnings.append("The available public leaderboard feedback is for recommended, not champion.")
        if stability_audit.get("risk_level") in {"medium", "high"}:
            issues.append(f"CV stability audit risk is {stability_audit.get('risk_level')}.")
        if stability_audit.get("public_within_seed_ci") is False:
            issues.append("Public score is outside the seed-level confidence interval.")
        if submission_target == "champion" and recommended and champion:
            champion_score = champion.get("local_score")
            recommended_score = recommended.get("local_score")
            if isinstance(champion_score, (int, float)) and isinstance(recommended_score, (int, float)):
                local_gap = champion_score - recommended_score
                if local_gap > 0:
                    warnings.append(f"Champion local CV is {local_gap:.4f} above recommended, but public feedback currently belongs to recommended.")

        decision = "allow_manual_submit" if not issues else "pause_manual_submit"
        status = "pass" if not issues else "needs_review"
        review = {
            "competition_name": self.competition_dir.name,
            "status": status,
            "decision": decision,
            "queue_task_id": queue_task_id,
            "submission_target": submission_target,
            "candidate": self._public_candidate(candidate),
            "champion": self._public_candidate(champion),
            "recommended": self._public_candidate(recommended),
            "leaderboard_feedback": {
                "submission_target": leaderboard_feedback.get("submission_target"),
                "candidate_task_id": leaderboard_feedback.get("candidate_task_id"),
                "public_score": leaderboard_feedback.get("public_score"),
                "leaderboard_rank": leaderboard_feedback.get("leaderboard_rank"),
            },
            "cv_stability_audit": {
                "risk_level": stability_audit.get("risk_level"),
                "seed_mean": stability_audit.get("seed_mean"),
                "seed_std": stability_audit.get("seed_std"),
                "fold_std": stability_audit.get("fold_std"),
                "public_gap_vs_seed_mean": stability_audit.get("public_gap_vs_seed_mean"),
                "public_within_seed_ci": stability_audit.get("public_within_seed_ci"),
                "issues": stability_audit.get("issues", []),
            },
            "queue_status_before_review": self._queue_item(queue, queue_task_id),
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                f"Run post-submit workflow for {submission_target} only after explicit human approval."
                if status == "pass"
                else "Do not submit this queue item yet; ask Remote Brain to re-plan or add an explicit human override."
            ),
        }
        review_path = self.competition_dir / "submission_decision_review.json"
        markdown_path = self.competition_dir / "submission_decision_review.md"
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(self._render_markdown(review), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="submission_decision_review",
            agent="submission_decision_review",
            title="Review manual submission decision",
            status=status,
            input_payload=review,
            prompt="Review whether the current manual leaderboard submission queue item should proceed after stability and leaderboard evidence.",
            scorecard={
                "agent": "submission_decision_review",
                "task_id": "submission_decision_review",
                "status": status,
                "scores": {
                    "queue_task_id": queue_task_id,
                    "submission_target": submission_target,
                    "decision": decision,
                    "stability_risk": stability_audit.get("risk_level", "unknown"),
                    "public_within_seed_ci": stability_audit.get("public_within_seed_ci", "unknown"),
                },
                "metric_name": candidate.get("metric_name"),
                "local_score": candidate.get("local_score"),
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "submission_decision_review": review_path,
                "submission_decision_review_markdown": markdown_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=self.competition_dir.name,
                profile_name="tabular_classic",
                task_id="submission_decision_review",
                status=status,
                metric_name=candidate.get("metric_name"),
                local_score=candidate.get("local_score"),
                brain_review_path=str(review_path),
                artifacts=[str(review_path), str(markdown_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=review["next_action"],
            )
        )
        return SubmissionDecisionReviewResult(status=status, review_path=review_path, markdown_path=markdown_path)

    @staticmethod
    def _queue_item(queue: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
        for item in queue.get("queue", []):
            if item.get("task_id") == task_id:
                return item
        return None

    @staticmethod
    def _public_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": candidate.get("task_id"),
            "metric_name": candidate.get("metric_name"),
            "local_score": candidate.get("local_score"),
            "risk_level": candidate.get("risk_level"),
            "feature_set": candidate.get("feature_set"),
            "selected_submission": candidate.get("selected_submission"),
        }

    @staticmethod
    def _render_markdown(review: Dict[str, Any]) -> str:
        return f"""# Submission Decision Review

Status: {review.get("status")}
Decision: {review.get("decision")}
Queue task: {review.get("queue_task_id")}
Submission target: {review.get("submission_target")}

## Candidate

```json
{json.dumps(review.get("candidate", {}), indent=2, ensure_ascii=False)}
```

## Stability Evidence

```json
{json.dumps(review.get("cv_stability_audit", {}), indent=2, ensure_ascii=False)}
```

## Issues

{chr(10).join(f"- {issue}" for issue in review.get("issues", [])) or "- None"}

## Next Action

{review.get("next_action")}
"""

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
