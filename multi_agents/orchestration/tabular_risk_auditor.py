from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class TabularRiskAuditResult:
    task_id: str
    status: str
    audit_path: Path


class TabularRiskAuditor:
    """Audit CV stability and leaderboard-risk signals for a tabular search run."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def audit(self, task_id: str = "tabular_model_search_v1") -> TabularRiskAuditResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        experiment_dir = self.competition_dir / "experiments" / task_id
        validation = self._read_json(experiment_dir / "validation_report.json")
        model_report = self._read_json_list(experiment_dir / "model_report.json")
        ensemble_report = self._read_json_list(experiment_dir / "ensemble_report.json")
        oof_path = experiment_dir / "oof_predictions.csv"

        audit = self._build_audit(
            task_id=task_id,
            validation=validation,
            model_report=model_report,
            ensemble_report=ensemble_report,
            oof_path=oof_path,
        )
        audit_path = experiment_dir / "risk_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "pass" if audit["risk_level"] in {"low", "medium"} else "needs_review"
        ledger_entry = self.ledger.create_entry(
            task_id=f"{task_id}_risk_audit",
            agent="tabular_risk_auditor",
            title="Audit tabular CV and leaderboard risk",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "task_id": task_id,
                "validation_report": validation,
                "model_report_count": len(model_report),
                "ensemble_report_count": len(ensemble_report),
                "oof_predictions_exists": oof_path.exists(),
            },
            prompt=(
                "Audit whether the latest tabular search score improvement is reliable enough "
                "to guide the next optimization step."
            ),
            scorecard={
                "agent": "tabular_risk_auditor",
                "task_id": f"{task_id}_risk_audit",
                "status": "pass" if status == "pass" else "needs_review",
                "scores": {
                    "risk_level": audit["risk_level"],
                    "fold_stability": audit["fold_stability_score"],
                    "ensemble_gain": audit["ensemble_gain"],
                    "max_model_correlation": audit["max_model_correlation"],
                },
                "metric_name": validation.get("metric_name"),
                "local_score": validation.get("local_score"),
                "issues": audit["issues"],
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"risk_audit": audit_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id=f"{task_id}_risk_audit",
                status=status,
                metric_name=validation.get("metric_name"),
                local_score=validation.get("local_score"),
                brain_review_path=str(audit_path),
                artifacts=[str(audit_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=audit["recommendation"],
            )
        )
        return TabularRiskAuditResult(task_id=task_id, status=status, audit_path=audit_path)

    def _build_audit(
        self,
        *,
        task_id: str,
        validation: Dict[str, Any],
        model_report: List[Dict[str, Any]],
        ensemble_report: List[Dict[str, Any]],
        oof_path: Path,
    ) -> Dict[str, Any]:
        completed_models = [item for item in model_report if item.get("status") == "completed"]
        fold_stats = []
        for item in completed_models:
            scores = [float(score) for score in item.get("fold_scores", []) if isinstance(score, (int, float))]
            if scores:
                fold_stats.append(
                    {
                        "model": item.get("model"),
                        "mean": float(mean(scores)),
                        "std": float(pstdev(scores)) if len(scores) > 1 else 0.0,
                        "min": float(min(scores)),
                        "max": float(max(scores)),
                    }
                )
        validation_cv_scores = [
            float(score)
            for score in validation.get("cv_scores", [])
            if isinstance(score, (int, float))
        ]
        if not fold_stats and validation_cv_scores:
            fold_stats.append(
                {
                    "model": validation.get("runner_kind") or validation.get("experiment") or task_id,
                    "mean": float(mean(validation_cv_scores)),
                    "std": float(pstdev(validation_cv_scores)) if len(validation_cv_scores) > 1 else 0.0,
                    "min": float(min(validation_cv_scores)),
                    "max": float(max(validation_cv_scores)),
                }
            )

        selected = validation.get("selected_submission") or {}
        best_model = validation.get("best_model") or {}
        single_model_score = validation.get("local_score") if isinstance(validation.get("local_score"), (int, float)) else None
        ensemble_gain = self._numeric(selected.get("score")) - self._numeric(best_model.get("score"))
        correlations = self._oof_correlations(oof_path, selected.get("models") or [])
        max_corr = max((item["correlation"] for item in correlations), default=None)
        avg_corr = mean([item["correlation"] for item in correlations]) if correlations else None
        max_fold_std = max((item["std"] for item in fold_stats), default=None)
        selected_kind = selected.get("kind") or ("best_single" if single_model_score is not None else "unknown")

        issues: List[str] = []
        risk_points = 0
        if max_fold_std is None:
            risk_points += 2
            issues.append("No fold scores were available; CV stability cannot be assessed.")
        elif max_fold_std > 0.035:
            risk_points += 2
            issues.append(f"High fold variance detected: max fold std is {max_fold_std:.4f}.")
        elif max_fold_std > 0.025:
            risk_points += 1
            issues.append(f"Moderate fold variance detected: max fold std is {max_fold_std:.4f}.")

        if selected_kind != "best_single" and ensemble_gain < 0.002:
            risk_points += 1
            issues.append(f"Selected ensemble gain is small ({ensemble_gain:.4f}); improvement may be noise.")
        if max_corr is not None and max_corr > 0.985 and selected_kind != "best_single":
            risk_points += 1
            issues.append(f"Selected ensemble models are highly correlated (max corr {max_corr:.4f}).")
        if selected_kind != "best_single" and len(completed_models) < 3:
            risk_points += 1
            issues.append("Fewer than three completed models; ensemble evidence is thin.")
        if selected_kind != "best_single" and not oof_path.exists():
            risk_points += 2
            issues.append("OOF predictions are missing; correlation and stacking-risk checks are incomplete.")

        if risk_points >= 4:
            risk_level = "high"
        elif risk_points >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"

        recommendation = {
            "low": "Proceed with the selected submission and use it as the current tabular reference.",
            "medium": "Proceed cautiously; prefer one more seed or fold repeat before leaderboard submission.",
            "high": "Do not trust this improvement yet; rerun with repeated CV or inspect leakage/validation split.",
        }[risk_level]

        return {
            "task_id": task_id,
            "status": "completed",
            "risk_level": risk_level,
            "risk_points": risk_points,
            "metric_name": validation.get("metric_name"),
            "selected_submission": selected,
            "best_model": best_model,
            "ensemble_gain": float(ensemble_gain),
            "fold_stability_score": None if max_fold_std is None else float(max(0.0, 1.0 - max_fold_std * 20.0)),
            "max_fold_std": max_fold_std,
            "max_model_correlation": max_corr,
            "avg_model_correlation": avg_corr,
            "fold_stats": fold_stats,
            "model_correlations": correlations,
            "ensemble_candidates": ensemble_report,
            "issues": issues,
            "recommendation": recommendation,
        }

    def _oof_correlations(self, oof_path: Path, model_names: Iterable[str]) -> List[Dict[str, Any]]:
        if not oof_path.exists():
            return []
        wanted = [f"oof_{name}" for name in model_names]
        with oof_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [column for column in wanted if column in (reader.fieldnames or [])]
            values = {column: [] for column in columns}
            for row in reader:
                for column in columns:
                    try:
                        values[column].append(float(row[column]))
                    except (TypeError, ValueError):
                        pass
        correlations = []
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                corr = self._pearson(values[left], values[right])
                if corr is not None:
                    correlations.append(
                        {
                            "left": left.removeprefix("oof_"),
                            "right": right.removeprefix("oof_"),
                            "correlation": corr,
                        }
                    )
        return correlations

    @staticmethod
    def _pearson(left: List[float], right: List[float]) -> Optional[float]:
        if len(left) != len(right) or len(left) < 2:
            return None
        left_mean = mean(left)
        right_mean = mean(right)
        numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
        left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left))
        right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right))
        if left_den == 0 or right_den == 0:
            return None
        return float(numerator / (left_den * right_den))

    @staticmethod
    def _numeric(value: Any) -> float:
        return float(value) if isinstance(value, (int, float)) else 0.0

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_json_list(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
