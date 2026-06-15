from __future__ import annotations

import json
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class LeaderboardGapAuditResult:
    status: str
    audit_path: Path


class LeaderboardGapAuditor:
    """Audit why public leaderboard feedback differs from local validation."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def audit(self) -> LeaderboardGapAuditResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        leaderboard = self._read_json(self.competition_dir / "leaderboard_feedback.json")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        champion = leaderboard.get("candidate") or champion_selection.get("champion") or {}
        risk_audit = self._load_risk_audit(champion)
        gap = self._score_gap(leaderboard, champion)
        drift = self._data_drift(manifest.to_dict())
        issues = self._issues(gap, risk_audit, drift)
        risk_level = self._risk_level(issues)
        recommendation = self._recommendation(risk_level, issues)

        audit = {
            "competition_name": manifest.competition_name,
            "status": "completed",
            "risk_level": risk_level,
            "leaderboard_feedback": self._public_feedback(leaderboard),
            "candidate": self._public_champion(champion),
            "champion": self._public_champion(champion_selection.get("champion") or {}),
            "score_gap": gap,
            "cv_stability": self._cv_stability(risk_audit),
            "data_drift": drift,
            "issues": issues,
            "recommendation": recommendation,
            "next_action": (
                "Ask Remote Brain for a stability-first experiment plan."
                if risk_level in {"high", "medium"}
                else "Leaderboard gap audit is acceptable; continue controlled optimization."
            ),
        }
        audit_path = self.competition_dir / "leaderboard_gap_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "pass" if risk_level in {"low", "medium"} else "needs_review"
        ledger_entry = self.ledger.create_entry(
            task_id="leaderboard_gap_audit",
            agent="leaderboard_gap_auditor",
            title="Audit leaderboard gap and stability",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "leaderboard_feedback_available": bool(leaderboard),
                "champion_available": bool(champion),
                "risk_audit_available": bool(risk_audit),
            },
            prompt=(
                "Audit public leaderboard gap against local CV using champion selection, "
                "risk audit, OOF stability signals, and train/test drift checks."
            ),
            scorecard={
                "agent": "leaderboard_gap_auditor",
                "task_id": "leaderboard_gap_audit",
                "status": status,
                "scores": {
                    "risk_level": risk_level,
                    "score_gap": gap.get("gap", "n/a"),
                    "data_drift_max": drift.get("max_drift_score", "n/a"),
                    "cv_max_fold_std": audit["cv_stability"].get("max_fold_std", "n/a"),
                },
                "metric_name": gap.get("metric_name"),
                "local_score": gap.get("local_score"),
                "issues": issues,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"leaderboard_gap_audit": audit_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="leaderboard_gap_audit",
                status=status,
                metric_name=gap.get("metric_name"),
                local_score=gap.get("local_score"),
                public_score=gap.get("public_score"),
                leaderboard_rank=leaderboard.get("leaderboard_rank") if isinstance(leaderboard.get("leaderboard_rank"), int) else None,
                brain_review_path=str(audit_path),
                artifacts=[str(audit_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=recommendation,
            )
        )
        return LeaderboardGapAuditResult(status=status, audit_path=audit_path)

    def _score_gap(self, leaderboard: Dict[str, Any], champion: Dict[str, Any]) -> Dict[str, Any]:
        local_score = self._number(leaderboard.get("local_score"))
        if local_score is None:
            local_score = self._number(champion.get("local_score"))
        public_score = self._number(leaderboard.get("public_score"))
        metric_name = leaderboard.get("metric_name") or champion.get("metric_name")
        lower_is_better = metric_name in {"rmse", "rmsle", "mae", "log_loss"}
        gap = None
        materially_worse = False
        if local_score is not None and public_score is not None:
            gap = public_score - local_score
            worse = public_score > local_score if lower_is_better else public_score < local_score
            materially_worse = worse and abs(gap) >= max(0.01, abs(local_score) * 0.03)
        return {
            "metric_name": metric_name,
            "lower_is_better": lower_is_better,
            "local_score": local_score,
            "public_score": public_score,
            "gap": gap,
            "materially_worse": materially_worse,
        }

    def _data_drift(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        tables = manifest.get("tables", {})
        train_rel = (tables.get("train.csv") or {}).get("path", "train.csv")
        test_rel = (tables.get("test.csv") or {}).get("path", "test.csv")
        train_path = self.competition_dir / train_rel
        test_path = self.competition_dir / test_rel
        if not train_path.exists() or not test_path.exists():
            return {"status": "missing_data", "features": [], "max_drift_score": None}
        train = self._read_csv_rows(train_path)
        test = self._read_csv_rows(test_path)
        ignored = {
            manifest.get("target_column"),
            manifest.get("id_column"),
        }
        ignored = {item for item in ignored if item}
        train_columns = list(train[0].keys()) if train else []
        test_columns = set(test[0].keys()) if test else set()
        features = []
        for column in train_columns:
            if column in ignored or column not in test_columns:
                continue
            train_values = [row.get(column, "") for row in train]
            test_values = [row.get(column, "") for row in test]
            if self._mostly_numeric(train_values):
                item = self._numeric_drift(column, train_values, test_values)
            else:
                item = self._categorical_drift(column, train_values, test_values)
            features.append(item)
        features.sort(key=lambda item: item["drift_score"], reverse=True)
        return {
            "status": "completed",
            "feature_count": len(features),
            "max_drift_score": features[0]["drift_score"] if features else None,
            "top_features": features[:10],
        }

    def _numeric_drift(self, column: str, train: List[str], test: List[str]) -> Dict[str, Any]:
        train_num = self._to_numbers(train)
        test_num = self._to_numbers(test)
        train_std = self._std(train_num)
        mean_gap = abs(self._mean(train_num) - self._mean(test_num))
        normalized_gap = mean_gap / max(train_std, 1e-9)
        missing_gap = abs(self._missing_rate(train) - self._missing_rate(test))
        return {
            "feature": column,
            "kind": "numeric",
            "drift_score": float(normalized_gap + missing_gap),
            "mean_gap_in_train_std": float(normalized_gap),
            "missing_rate_gap": float(missing_gap),
        }

    def _categorical_drift(self, column: str, train: List[str], test: List[str]) -> Dict[str, Any]:
        train_str = [self._clean_category(value) for value in train]
        test_str = [self._clean_category(value) for value in test]
        train_values = set(train_str)
        test_values = set(test_str)
        unseen = test_values - train_values
        unseen_rate = sum(value in unseen for value in test_str) / max(len(test_str), 1)
        train_top_value = self._top_value(train_str)
        test_top_value = self._top_value(test_str)
        top_shift = 0.0 if train_top_value == test_top_value else 0.5
        missing_gap = abs(
            train_str.count("__MISSING__") / max(len(train_str), 1)
            - test_str.count("__MISSING__") / max(len(test_str), 1)
        )
        return {
            "feature": column,
            "kind": "categorical",
            "drift_score": float(unseen_rate + top_shift + missing_gap),
            "unseen_category_rate": unseen_rate,
            "top_category_changed": bool(train_top_value != test_top_value),
            "missing_rate_gap": float(missing_gap),
        }

    def _load_risk_audit(self, champion: Dict[str, Any]) -> Dict[str, Any]:
        candidates = []
        risk_path = champion.get("risk_audit_path")
        if risk_path:
            candidates.append(Path(risk_path))
            candidates.append(self.competition_dir / risk_path)
        task_id = champion.get("task_id")
        if task_id:
            candidates.append(self.competition_dir / "experiments" / task_id / "risk_audit.json")
        candidates.extend(sorted((self.competition_dir / "experiments").glob("*/risk_audit.json")))
        candidates.extend(sorted((self.competition_dir / "runs").glob("*/artifacts/risk_audit.json")))
        for path in candidates:
            if path.exists():
                return self._read_json(path)
        return {}

    def _cv_stability(self, risk_audit: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "risk_level": risk_audit.get("risk_level"),
            "max_fold_std": risk_audit.get("max_fold_std"),
            "max_model_correlation": risk_audit.get("max_model_correlation"),
            "ensemble_gain": risk_audit.get("ensemble_gain"),
            "issues": risk_audit.get("issues", []),
            "available": bool(risk_audit),
        }

    def _issues(
        self,
        gap: Dict[str, Any],
        risk_audit: Dict[str, Any],
        drift: Dict[str, Any],
    ) -> List[str]:
        issues = []
        if gap.get("public_score") is None:
            issues.append("No public leaderboard score is available.")
        elif gap.get("materially_worse"):
            issues.append("Public leaderboard score is materially worse than local CV.")
        if risk_audit.get("risk_level") in {"medium", "high"}:
            issues.append(f"Existing tabular risk audit is {risk_audit.get('risk_level')}.")
        max_fold_std = self._number(risk_audit.get("max_fold_std"))
        if max_fold_std is not None and max_fold_std > 0.025:
            issues.append(f"Fold variance is elevated: max_fold_std={max_fold_std:.4f}.")
        max_corr = self._number(risk_audit.get("max_model_correlation"))
        if max_corr is not None and max_corr > 0.985:
            issues.append(f"Ensemble members are highly correlated: max_corr={max_corr:.4f}.")
        max_drift = self._number(drift.get("max_drift_score"))
        if max_drift is not None and max_drift > 0.5:
            issues.append(f"Train/test drift signal is elevated: max_drift_score={max_drift:.4f}.")
        return issues

    @staticmethod
    def _risk_level(issues: List[str]) -> str:
        if any("materially worse" in issue for issue in issues):
            return "high"
        if len(issues) >= 2:
            return "medium"
        if issues:
            return "low"
        return "low"

    @staticmethod
    def _recommendation(risk_level: str, issues: List[str]) -> str:
        if risk_level == "high":
            return "Pause leaderboard submissions; run repeated CV, drift checks, and stable low-variance alternatives."
        if risk_level == "medium":
            return "Proceed cautiously; prefer another seed repeat or validation split audit before submitting again."
        if issues:
            return "Continue optimization, but keep the identified warning in the next Brain review."
        return "Leaderboard gap audit found no strong blocking signal."

    @staticmethod
    def _public_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": feedback.get("status"),
            "source": feedback.get("source"),
            "submission_id": feedback.get("submission_id"),
            "submission_target": feedback.get("submission_target"),
            "candidate_task_id": feedback.get("candidate_task_id"),
            "public_score": feedback.get("public_score"),
            "private_score": feedback.get("private_score"),
            "leaderboard_rank": feedback.get("leaderboard_rank"),
            "local_score": feedback.get("local_score"),
            "metric_name": feedback.get("metric_name"),
            "notes": feedback.get("notes"),
            "submission_path": feedback.get("submission_path"),
        }

    @staticmethod
    def _public_champion(champion: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source_id": champion.get("source_id"),
            "task_id": champion.get("task_id"),
            "metric_name": champion.get("metric_name"),
            "local_score": champion.get("local_score"),
            "risk_level": champion.get("risk_level"),
            "selected_submission": champion.get("selected_submission"),
        }

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _mostly_numeric(values: List[str]) -> bool:
        non_missing = [value for value in values if str(value).strip() != ""]
        if not non_missing:
            return False
        numeric = 0
        for value in non_missing:
            try:
                float(value)
                numeric += 1
            except (TypeError, ValueError):
                pass
        return numeric / len(non_missing) >= 0.9

    @staticmethod
    def _to_numbers(values: List[str]) -> List[float]:
        numbers = []
        for value in values:
            if str(value).strip() == "":
                continue
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                pass
        return numbers

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @classmethod
    def _std(cls, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        avg = cls._mean(values)
        return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))

    @staticmethod
    def _missing_rate(values: List[str]) -> float:
        return sum(str(value).strip() == "" for value in values) / max(len(values), 1)

    @staticmethod
    def _clean_category(value: Any) -> str:
        text = str(value).strip()
        return text if text else "__MISSING__"

    @staticmethod
    def _top_value(values: List[str]) -> Optional[str]:
        if not values:
            return None
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
