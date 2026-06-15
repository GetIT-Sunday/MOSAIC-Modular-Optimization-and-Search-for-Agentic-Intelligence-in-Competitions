from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class ChampionSelectionResult:
    status: str
    champion_path: Path
    selection_path: Path


class ExperimentChampionSelector:
    """Select the current best valid submission from experiment and ledger artifacts."""

    RISK_PENALTIES = {
        "low": 0.0,
        "medium": 0.01,
        "high": 0.05,
        "unknown": 0.003,
    }
    LEADERBOARD_GAP_PENALTY = 0.04
    HIGH_GAP_UNKNOWN_RISK_PENALTY = 0.025
    STABILITY_FIRST_BONUS = 0.005

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def select(self) -> ChampionSelectionResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        candidates = self._collect_candidates()
        scored = [candidate for candidate in candidates if candidate["eligible"]]
        lower_is_better = self._lower_is_better(scored)
        context = self._selection_context()
        for candidate in scored:
            penalty = self.RISK_PENALTIES.get(candidate["risk_level"], self.RISK_PENALTIES["unknown"])
            contextual = self._contextual_adjustment(candidate, context)
            candidate["base_risk_penalty"] = penalty
            candidate["contextual_penalty"] = contextual["penalty"]
            candidate["contextual_bonus"] = contextual["bonus"]
            candidate["selection_context_notes"] = contextual["notes"]
            effective_penalty = penalty + contextual["penalty"] - contextual["bonus"]
            candidate["risk_penalty"] = effective_penalty
            if lower_is_better:
                candidate["risk_adjusted_score"] = candidate["local_score"] + effective_penalty
            else:
                candidate["risk_adjusted_score"] = candidate["local_score"] - effective_penalty
        scored.sort(key=lambda item: item["risk_adjusted_score"], reverse=not lower_is_better)
        champion = scored[0] if scored else None

        champion_path = self.competition_dir / "champion_submission.csv"
        if champion:
            shutil.copyfile(champion["submission_path"], champion_path)
        comparison = self._build_comparison_report(
            candidates=candidates,
            scored=scored,
            champion=champion,
            context=context,
            lower_is_better=lower_is_better,
        )
        comparison_path = self.competition_dir / "champion_comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        selection = {
            "competition_name": manifest.competition_name,
            "decision": "champion_selected" if champion else "no_valid_champion",
            "champion": self._public_candidate(champion) if champion else None,
            "comparison_report_path": str(comparison_path),
            "selection_context": context,
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(scored),
            "candidates": [self._public_candidate(candidate) for candidate in candidates],
        }
        selection_path = self.competition_dir / "champion_selection.json"
        selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "pass" if champion else "needs_review"
        ledger_entry = self.ledger.create_entry(
            task_id="champion_selection",
            agent="champion_selector",
            title="Select current champion submission",
            status=status,
            input_payload=selection,
            prompt=(
                "Select the current best valid submission from all scored experiment artifacts, "
                "considering validator status and risk-adjusted score."
            ),
            scorecard={
                "agent": "champion_selector",
                "task_id": "champion_selection",
                "status": status,
                "scores": {
                    "eligible_candidates": len(scored),
                    "selected_score": champion.get("local_score") if champion else "n/a",
                    "risk_adjusted_score": champion.get("risk_adjusted_score") if champion else "n/a",
                    "risk_level": champion.get("risk_level") if champion else "n/a",
                    "leaderboard_gap_risk": context.get("leaderboard_gap", {}).get("risk_level", "n/a"),
                    "contextual_penalty": champion.get("contextual_penalty") if champion else "n/a",
                    "contextual_bonus": champion.get("contextual_bonus") if champion else "n/a",
                },
                "metric_name": champion.get("metric_name") if champion else None,
                "local_score": champion.get("local_score") if champion else None,
                "issues": [] if champion else ["No valid scored submission candidate was found."],
                "recommended_human_action": "continue" if champion else "patch_prompt",
            },
            artifacts={
                "champion_selection": selection_path,
                "champion_comparison": comparison_path,
                "champion_submission": champion_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="champion_selection",
                status=status,
                metric_name=champion.get("metric_name") if champion else None,
                local_score=champion.get("local_score") if champion else None,
                submission_path=str(champion_path) if champion else None,
                brain_review_path=str(selection_path),
                artifacts=[str(selection_path), str(champion_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=f"Selected {champion.get('source_id')}" if champion else "No champion selected.",
            )
        )
        return ChampionSelectionResult(status=status, champion_path=champion_path, selection_path=selection_path)

    def _collect_candidates(self) -> List[Dict[str, Any]]:
        candidates = []
        for run_dir in sorted((self.competition_dir / "runs").glob("*")):
            artifacts = run_dir / "artifacts"
            if not artifacts.exists():
                continue
            candidates.append(self._candidate_from_paths(
                source_id=run_dir.name,
                validation_path=artifacts / "validation_report.json",
                validator_path=artifacts / "validator_result.json",
                submission_path=artifacts / "submission.csv",
                risk_path=self._nearest_following_risk(run_dir),
            ))
        for exp_dir in sorted((self.competition_dir / "experiments").glob("*")):
            candidates.append(self._candidate_from_paths(
                source_id=f"experiment:{exp_dir.name}",
                validation_path=exp_dir / "validation_report.json",
                validator_path=exp_dir / "validator_result.json",
                submission_path=exp_dir / "submission.csv",
                risk_path=exp_dir / "risk_audit.json",
            ))
        return [candidate for candidate in candidates if candidate]

    def _candidate_from_paths(
        self,
        *,
        source_id: str,
        validation_path: Path,
        validator_path: Path,
        submission_path: Path,
        risk_path: Optional[Path],
    ) -> Dict[str, Any]:
        validation = self._read_json(validation_path)
        validator = self._read_json(validator_path)
        risk = self._read_json(risk_path) if risk_path else {}
        score = validation.get("local_score")
        valid = validator.get("ok") is True
        eligible = (
            submission_path.exists()
            and valid
            and isinstance(score, (int, float))
            and validation.get("status") in {"completed", None}
        )
        risk_level = risk.get("risk_level") or "unknown"
        if risk_level == "high":
            eligible = False
        return {
            "source_id": source_id,
            "task_id": validation.get("experiment") or validation.get("baseline") or source_id,
            "eligible": eligible,
            "metric_name": validation.get("metric_name"),
            "local_score": score,
            "submission_valid": valid,
            "submission_path": submission_path,
            "validation_report_path": validation_path,
            "validator_result_path": validator_path,
            "risk_audit_path": risk_path,
            "risk_level": risk_level,
            "risk_issues": risk.get("issues", []),
            "feature_set": validation.get("feature_set"),
            "requested_drop_features": validation.get("requested_drop_features", []),
            "feature_set_source": validation.get("feature_set_source"),
            "selected_submission": validation.get("selected_submission"),
            "ineligible_reason": self._ineligible_reason(submission_path, valid, score, validation, risk_level),
        }

    def _nearest_following_risk(self, run_dir: Path) -> Optional[Path]:
        try:
            run_index = int(run_dir.name.split("_", 1)[0])
        except ValueError:
            return None
        task = run_dir.name.split("_", 1)[1] if "_" in run_dir.name else ""
        risks = []
        for candidate in sorted((self.competition_dir / "runs").glob("*_risk_audit")):
            try:
                risk_index = int(candidate.name.split("_", 1)[0])
            except ValueError:
                continue
            if risk_index > run_index and task in candidate.name:
                risk_file = candidate / "artifacts" / "risk_audit.json"
                if risk_file.exists():
                    risks.append((risk_index, risk_file))
        return risks[0][1] if risks else None

    def _ineligible_reason(
        self,
        submission_path: Path,
        valid: bool,
        score: Any,
        validation: Dict[str, Any],
        risk_level: str,
    ) -> str:
        if not submission_path.exists():
            return "missing_submission"
        if not valid:
            return "validator_failed"
        if not isinstance(score, (int, float)):
            return "missing_score"
        if validation.get("status") not in {"completed", None}:
            return f"status_{validation.get('status')}"
        if risk_level == "high":
            return "high_risk"
        return ""

    def _lower_is_better(self, candidates: List[Dict[str, Any]]) -> bool:
        metric = next((item.get("metric_name") for item in candidates if item.get("metric_name")), "")
        return metric in {"rmse", "rmsle", "mae", "log_loss"}

    def _selection_context(self) -> Dict[str, Any]:
        gap_audit = self._read_json(self.competition_dir / "leaderboard_gap_audit.json")
        stability_review = self._read_json(self.competition_dir / "stability_first_review.json")
        leakage_audit = self._read_json(self.competition_dir / "tabular_feature_leakage_audit.json")
        gap_champion = gap_audit.get("champion") or {}
        score_gap = gap_audit.get("score_gap") or {}
        return {
            "leaderboard_gap": {
                "risk_level": gap_audit.get("risk_level"),
                "materially_worse": bool(score_gap.get("materially_worse")),
                "gap": score_gap.get("gap"),
                "champion_task_id": gap_champion.get("task_id"),
                "champion_source_id": gap_champion.get("source_id"),
            },
            "stability_first": {
                "task_id": stability_review.get("task_id"),
                "status": stability_review.get("status"),
                "risk_level": stability_review.get("risk_level"),
                "decision": stability_review.get("decision"),
                "local_score": stability_review.get("local_score"),
            },
            "feature_leakage": {
                "risk_level": leakage_audit.get("risk_level"),
                "recommended_drop_features": leakage_audit.get("recommended_drop_features", []),
                "issues": leakage_audit.get("issues", []),
                "warnings": leakage_audit.get("warnings", []),
            },
        }

    def _contextual_adjustment(
        self,
        candidate: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        notes = []
        penalty = 0.0
        bonus = 0.0
        gap = context.get("leaderboard_gap") or {}
        stability = context.get("stability_first") or {}
        high_gap = gap.get("risk_level") == "high" and gap.get("materially_worse")
        if high_gap and self._matches_gap_champion(candidate, gap):
            penalty += self.LEADERBOARD_GAP_PENALTY
            notes.append("penalized_original_champion_for_public_gap")
        if high_gap and candidate.get("risk_level") == "unknown":
            penalty += self.HIGH_GAP_UNKNOWN_RISK_PENALTY
            notes.append("penalized_unknown_risk_under_public_gap")
        if (
            high_gap
            and candidate.get("task_id") == stability.get("task_id")
            and stability.get("status") == "validated"
            and stability.get("risk_level") in {"low", "medium"}
        ):
            bonus += self.STABILITY_FIRST_BONUS
            notes.append("stability_first_candidate_bonus")
        return {"penalty": penalty, "bonus": bonus, "notes": notes}

    @staticmethod
    def _matches_gap_champion(candidate: Dict[str, Any], gap: Dict[str, Any]) -> bool:
        champion_task = gap.get("champion_task_id")
        champion_source = gap.get("champion_source_id")
        source = candidate.get("source_id") or ""
        return (
            bool(champion_task and candidate.get("task_id") == champion_task)
            or bool(champion_source and source == champion_source)
            or bool(champion_task and champion_task in source)
        )

    def _public_candidate(self, candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if candidate is None:
            return None
        public = dict(candidate)
        for key in ["submission_path", "validation_report_path", "validator_result_path", "risk_audit_path"]:
            value = public.get(key)
            public[key] = str(value) if value else None
        return public

    def _build_comparison_report(
        self,
        *,
        candidates: List[Dict[str, Any]],
        scored: List[Dict[str, Any]],
        champion: Optional[Dict[str, Any]],
        context: Dict[str, Any],
        lower_is_better: bool,
    ) -> Dict[str, Any]:
        leakage = context.get("feature_leakage") or {}
        recommended_drops = set(leakage.get("recommended_drop_features") or [])
        ranked = []
        for index, candidate in enumerate(scored, start=1):
            public = self._public_candidate(candidate) or {}
            drops = set(public.get("requested_drop_features") or [])
            feature_set = public.get("feature_set")
            task_id = public.get("task_id") or ""
            public["rank"] = index
            public["feature_control"] = {
                "feature_set": feature_set,
                "is_stability_first": task_id == (context.get("stability_first") or {}).get("task_id") or feature_set == "stable",
                "is_leakage_safe": feature_set == "leakage_safe" or "leakage_safe" in task_id,
                "uses_leakage_recommended_drops": bool(recommended_drops and recommended_drops.issubset(drops)),
                "leakage_recommended_drop_overlap": sorted(recommended_drops & drops),
            }
            public["selection_summary"] = self._selection_summary(public, context)
            ranked.append(public)
        return {
            "status": "completed",
            "decision": "champion_selected" if champion else "no_valid_champion",
            "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "champion_task_id": champion.get("task_id") if champion else None,
            "champion_source_id": champion.get("source_id") if champion else None,
            "selection_context": context,
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(scored),
            "top_candidates": ranked[:12],
            "feature_control_candidates": [
                item
                for item in ranked
                if item.get("feature_control", {}).get("is_stability_first")
                or item.get("feature_control", {}).get("is_leakage_safe")
            ],
            "ineligible_summary": self._ineligible_summary(candidates),
            "next_action": (
                "Use the selected champion for submission readiness, or inspect feature_control_candidates before overriding."
                if champion
                else "Run a validated experiment with a submission before selection."
            ),
        }

    @staticmethod
    def _selection_summary(candidate: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        notes = []
        if candidate.get("selection_context_notes"):
            notes.extend(candidate["selection_context_notes"])
        control = candidate.get("feature_control", {})
        if control.get("is_stability_first"):
            notes.append("stability_first_candidate")
        if control.get("is_leakage_safe"):
            notes.append("leakage_safe_candidate")
        if control.get("uses_leakage_recommended_drops"):
            notes.append("uses_leakage_recommended_drops")
        leakage = context.get("feature_leakage") or {}
        if leakage.get("risk_level") in {"medium", "high"} and not control.get("uses_leakage_recommended_drops"):
            notes.append("does_not_apply_leakage_drop_recommendation")
        return notes

    @staticmethod
    def _ineligible_summary(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for candidate in candidates:
            if candidate.get("eligible"):
                continue
            reason = candidate.get("ineligible_reason") or "unknown"
            summary[reason] = summary.get(reason, 0) + 1
        return dict(sorted(summary.items()))

    @staticmethod
    def _read_json(path: Optional[Path]) -> Dict[str, Any]:
        if not path or not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
