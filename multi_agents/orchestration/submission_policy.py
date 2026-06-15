from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class SubmissionPolicyResult:
    status: str
    policy_path: Path
    recommended_submission_path: Optional[Path]


class SubmissionPolicy:
    """Choose a submission recommendation separately from the highest-CV champion."""

    SCORE_TOLERANCE = 0.015

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self) -> SubmissionPolicyResult:
        comparison = self._read_json(self.competition_dir / "champion_comparison.json")
        selection = self._read_json(self.competition_dir / "champion_selection.json")
        promotion_review = self._read_json(self.competition_dir / "promotion_gate_review.json")
        issues = []
        warnings = []
        promoted, promotion_notes = self._promoted_candidate(promotion_review)
        if comparison.get("status") != "completed" and not promoted:
            issues.append("champion_comparison.json is missing or incomplete; run --select-champion first.")
        top_candidates = comparison.get("top_candidates") or []
        if not top_candidates and not promoted:
            issues.append("No eligible top candidates are available.")

        cv_champion = top_candidates[0] if top_candidates else selection.get("champion")
        if promoted:
            recommended = promoted
            policy_notes = promotion_notes
        else:
            recommended, policy_notes = self._choose_recommended(cv_champion, comparison)
        recommended_submission_path = None
        if recommended:
            source_submission = Path(recommended.get("submission_path") or "")
            if source_submission.exists():
                recommended_submission_path = self.competition_dir / "recommended_submission.csv"
                shutil.copyfile(source_submission, recommended_submission_path)
            else:
                issues.append("Recommended candidate submission file is missing.")

        status = "pass" if not issues else "needs_review"
        policy = {
            "competition_name": selection.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": "recommended_submission_selected" if recommended and status == "pass" else "policy_blocked",
            "cv_champion": self._public_candidate(cv_champion),
            "recommended_submission_candidate": self._public_candidate(recommended),
            "recommended_submission_path": str(recommended_submission_path) if recommended_submission_path else None,
            "promotion_gate": {
                "status": promotion_review.get("status"),
                "decision": promotion_review.get("decision"),
                "promoted_task_id": (promotion_review.get("promoted_candidate") or {}).get("task_id"),
                "promoted_submission_path": promotion_review.get("promoted_submission_path"),
                "used_promoted_candidate": bool(promoted),
            },
            "selection_context": comparison.get("selection_context", {}),
            "policy": {
                "score_tolerance": self.SCORE_TOLERANCE,
                "notes": policy_notes,
                "changed_from_cv_champion": bool(
                    cv_champion
                    and recommended
                    and cv_champion.get("source_id") != recommended.get("source_id")
                ),
                "source": "promotion_gate" if promoted else "champion_comparison",
            },
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Review recommended_submission.csv before wiring it into the final submission gate."
                if status == "pass"
                else "Fix policy issues, rerun champion selection, then rerun submission policy."
            ),
        }
        policy_path = self.competition_dir / "submission_policy.json"
        policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="submission_policy",
            agent="submission_policy",
            title="Choose recommended submission policy",
            status=status,
            input_payload=policy,
            prompt=(
                "Choose a recommended submission candidate separately from the highest local-CV champion, "
                "using leaderboard gap, leakage audit, feature-control, and risk context."
            ),
            scorecard={
                "agent": "submission_policy",
                "task_id": "submission_policy",
                "status": status,
                "scores": {
                    "changed_from_cv_champion": policy["policy"]["changed_from_cv_champion"],
                    "cv_champion_score": (cv_champion or {}).get("local_score"),
                    "recommended_score": (recommended or {}).get("local_score"),
                    "recommended_risk": (recommended or {}).get("risk_level"),
                    "recommended_feature_set": (recommended or {}).get("feature_set"),
                    "promotion_gate_used": bool(promoted),
                },
                "metric_name": (recommended or cv_champion or {}).get("metric_name"),
                "local_score": (recommended or cv_champion or {}).get("local_score"),
                "issues": issues + warnings + policy_notes,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "submission_policy": policy_path,
                "recommended_submission": recommended_submission_path or Path(""),
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=policy["competition_name"],
                profile_name="tabular_classic",
                task_id="submission_policy",
                status=status,
                metric_name=(recommended or cv_champion or {}).get("metric_name"),
                local_score=(recommended or cv_champion or {}).get("local_score"),
                submission_path=str(recommended_submission_path) if recommended_submission_path else None,
                brain_review_path=str(policy_path),
                artifacts=[str(policy_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=policy["next_action"],
            )
        )
        return SubmissionPolicyResult(status=status, policy_path=policy_path, recommended_submission_path=recommended_submission_path)

    def _promoted_candidate(self, promotion_review: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], List[str]]:
        if promotion_review.get("decision") != "promote_candidate":
            return None, ["Promotion gate did not promote a candidate."]
        promoted = promotion_review.get("promoted_candidate") or {}
        promoted_submission = Path(promotion_review.get("promoted_submission_path") or "")
        if not promoted_submission.exists():
            fallback_submission = Path(promoted.get("submission_path") or "")
            if fallback_submission.exists():
                promoted_submission = fallback_submission
        if not promoted or not promoted_submission.exists():
            return None, ["Promotion gate promoted a candidate, but promoted submission evidence is missing."]
        candidate = dict(promoted)
        candidate["source_id"] = candidate.get("source_id") or f"promotion:{candidate.get('task_id', 'unknown')}"
        candidate["submission_path"] = str(promoted_submission)
        candidate["risk_level"] = candidate.get("risk_level") or "low"
        candidate["feature_set"] = candidate.get("feature_set") or "promotion_gate"
        candidate["promotion_gate_decision"] = promotion_review.get("decision")
        notes = [
            f"Promotion gate selected candidate: {candidate.get('task_id')}.",
            "Recommended submission is sourced from promoted_submission.csv.",
        ]
        return candidate, notes

    def _choose_recommended(
        self,
        cv_champion: Optional[Dict[str, Any]],
        comparison: Dict[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        if not cv_champion:
            return None, ["No CV champion is available."]
        context = comparison.get("selection_context") or {}
        leakage = context.get("feature_leakage") or {}
        gap = context.get("leaderboard_gap") or {}
        risk_context = leakage.get("risk_level") in {"medium", "high"} or gap.get("risk_level") in {"medium", "high"}
        if not risk_context:
            return cv_champion, ["No medium/high leakage or leaderboard-gap context; keep CV champion."]

        champion_score = cv_champion.get("local_score")
        if not isinstance(champion_score, (int, float)):
            return cv_champion, ["CV champion score is unavailable; keep CV champion."]

        safe_candidates = []
        for candidate in comparison.get("feature_control_candidates", []):
            if candidate.get("risk_level") not in {"low", "medium"}:
                continue
            control = candidate.get("feature_control") or {}
            if not (
                control.get("uses_leakage_recommended_drops")
                or control.get("is_stability_first")
                or control.get("is_leakage_safe")
            ):
                continue
            score = candidate.get("local_score")
            if not isinstance(score, (int, float)):
                continue
            if champion_score - score <= self.SCORE_TOLERANCE:
                safe_candidates.append(candidate)
        if not safe_candidates:
            return cv_champion, ["No safer feature-control candidate is within score tolerance; keep CV champion."]

        safe_candidates.sort(key=lambda item: item.get("local_score", float("-inf")), reverse=True)
        recommended = safe_candidates[0]
        notes = [
            "Medium/high leakage or leaderboard-gap context is present.",
            f"Selected safer candidate within tolerance: {recommended.get('task_id')}.",
        ]
        if cv_champion.get("source_id") != recommended.get("source_id"):
            notes.append("Recommended submission differs from highest-CV champion.")
        return recommended, notes

    @staticmethod
    def _public_candidate(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidate:
            return None
        public = dict(candidate)
        for key in ["submission_path", "validation_report_path", "validator_result_path", "risk_audit_path"]:
            value = public.get(key)
            public[key] = str(value) if value else None
        return public

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
