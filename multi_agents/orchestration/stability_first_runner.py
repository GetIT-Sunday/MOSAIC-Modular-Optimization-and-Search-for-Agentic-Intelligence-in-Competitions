from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .tabular_risk_auditor import TabularRiskAuditor
from .tabular_search_runner import TabularSearchRunner, TabularSearchResult


@dataclass(frozen=True)
class StabilityFirstRunResult:
    status: str
    feature_report_path: Path
    search_result: TabularSearchResult
    review_path: Path


class StabilityFirstRunner:
    """Run a stability-first tabular search after a leaderboard gap audit."""

    DRIFT_DERIVED_FEATURES = {
        "Name": ["Title"],
        "Ticket": ["TicketPrefix"],
        "Cabin": ["HasCabin", "CabinDeck"],
    }

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
        task_id: str = "stability_first_search_v1",
        cv_seeds: Optional[List[int]] = None,
    ) -> StabilityFirstRunResult:
        seeds = cv_seeds or [42, 123, 777]
        feature_report_path = self.prepare_feature_report()
        search_result = TabularSearchRunner(self.competition_dir, memory=self.memory).run(
            task_id=task_id,
            cv_seeds=seeds,
            feature_set="stable",
        )
        risk_result = TabularRiskAuditor(self.competition_dir, memory=self.memory).audit(task_id=task_id)
        review_path = self._write_review(search_result, risk_result.audit_path, feature_report_path, seeds)
        return StabilityFirstRunResult(
            status=search_result.status,
            feature_report_path=feature_report_path,
            search_result=search_result,
            review_path=review_path,
        )

    def prepare_feature_report(self) -> Path:
        gap_audit = self._read_json(self.competition_dir / "leaderboard_gap_audit.json")
        drift = gap_audit.get("data_drift") or {}
        top_features = drift.get("top_features") or []
        drop_features = []
        reasons = []
        for item in top_features:
            feature = item.get("feature")
            score = item.get("drift_score")
            if not feature or not isinstance(score, (int, float)) or score <= 0.5:
                continue
            for derived in self.DRIFT_DERIVED_FEATURES.get(feature, [feature]):
                if derived not in drop_features:
                    drop_features.append(derived)
            reasons.append(
                {
                    "source_feature": feature,
                    "drift_score": score,
                    "dropped_features": self.DRIFT_DERIVED_FEATURES.get(feature, [feature]),
                }
            )
        feature_report = {
            "status": "completed",
            "source": "leaderboard_gap_audit",
            "risk_level": gap_audit.get("risk_level"),
            "drop_features": drop_features,
            "drop_reasons": reasons,
            "notes": "Drop high-drift raw or derived features before repeated-CV search.",
        }
        report_dir = self.competition_dir / "experiments" / "stability_first_features_v1"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "feature_report.json"
        report_path.write_text(json.dumps(feature_report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_path

    def _write_review(
        self,
        search_result: TabularSearchResult,
        risk_audit_path: Path,
        feature_report_path: Path,
        seeds: List[int],
    ) -> Path:
        validation = self._read_json(search_result.validation_report)
        risk_audit = self._read_json(risk_audit_path)
        gap_audit = self._read_json(self.competition_dir / "leaderboard_gap_audit.json")
        review = {
            "task_id": search_result.task_id,
            "status": search_result.status,
            "metric_name": validation.get("metric_name"),
            "local_score": validation.get("local_score"),
            "feature_set": validation.get("feature_set"),
            "requested_drop_features": validation.get("requested_drop_features", []),
            "cv_seeds": seeds,
            "risk_level": risk_audit.get("risk_level"),
            "leaderboard_gap_risk_level": gap_audit.get("risk_level"),
            "submission_valid": search_result.validator_result.ok,
            "decision": (
                "select_champion_if_risk_acceptable"
                if search_result.validator_result.ok and risk_audit.get("risk_level") in {"low", "medium"}
                else "needs_review"
            ),
        }
        review_path = self.competition_dir / "stability_first_review.json"
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="stability_first_review",
            agent="stability_first_runner",
            title="Review stability-first search",
            status="pass" if review["decision"] != "needs_review" else "needs_review",
            input_payload=review,
            prompt="Review the stability-first repeated-CV search after dropping high-drift features.",
            scorecard={
                "agent": "stability_first_runner",
                "task_id": "stability_first_review",
                "status": "pass" if review["decision"] != "needs_review" else "needs_review",
                "scores": {
                    "submission_valid": search_result.validator_result.ok,
                    "risk_level": risk_audit.get("risk_level", "unknown"),
                    "dropped_feature_count": len(review["requested_drop_features"]),
                    "cv_seed_count": len(seeds),
                },
                "metric_name": validation.get("metric_name"),
                "local_score": validation.get("local_score"),
                "issues": risk_audit.get("issues", []) + search_result.validator_result.errors,
                "recommended_human_action": "continue" if review["decision"] != "needs_review" else "patch_prompt",
            },
            artifacts={
                "stability_first_review": review_path,
                "stability_feature_report": feature_report_path,
                "stability_risk_audit": risk_audit_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=self.competition_dir.name,
                profile_name="tabular_classic",
                task_id="stability_first_review",
                status=review["status"],
                metric_name=validation.get("metric_name"),
                local_score=validation.get("local_score"),
                submission_path=str(search_result.submission_path) if search_result.submission_path else None,
                brain_review_path=str(review_path),
                artifacts=[str(review_path), str(feature_report_path), str(risk_audit_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=review["decision"],
            )
        )
        return review_path

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
