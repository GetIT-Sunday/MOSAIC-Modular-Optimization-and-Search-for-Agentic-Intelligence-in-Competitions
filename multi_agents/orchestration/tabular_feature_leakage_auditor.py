from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class TabularFeatureLeakageAuditResult:
    status: str
    audit_path: Path
    experiment_dir: Path


class TabularFeatureLeakageAuditor:
    """Audit tabular feature leakage, transform-scope risks, and train/test drift."""

    DERIVED_FEATURES = {
        "FamilySize": ["SibSp", "Parch"],
        "IsAlone": ["SibSp", "Parch"],
        "Title": ["Name"],
        "HasCabin": ["Cabin"],
        "CabinDeck": ["Cabin"],
        "TicketPrefix": ["Ticket"],
    }
    FIT_SCOPE_RISKS = {
        "Title": "Rare-title bucketing must be fit on train and applied to test; computing rarity separately on test can make validation optimistic.",
    }

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def audit(self, task_id: str = "tabular_feature_leakage_audit_v1") -> TabularFeatureLeakageAuditResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        manifest.write_json(self.competition_dir / "data_manifest.json")
        experiment_dir = self.competition_dir / "experiments" / task_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.competition_dir / "train.csv"
        test_path = self.competition_dir / "test.csv"
        issues = []
        warnings = []
        if not train_path.exists() or not test_path.exists():
            issues.append("train.csv or test.csv is missing.")
            audit = self._empty_audit(manifest.competition_name, issues)
        else:
            train = self._read_csv_rows(train_path)
            test = self._read_csv_rows(test_path)
            train_features = self._add_common_features(train)
            test_features = self._add_common_features(test)
            raw_drift = self._drift_report(train, test, manifest.target_column, manifest.id_column)
            engineered_drift = self._drift_report(
                train_features,
                test_features,
                manifest.target_column,
                manifest.id_column,
                only_columns=[name for name in self.DERIVED_FEATURES if name in train_features[0] and name in test_features[0]] if train_features and test_features else [],
            )
            leakage_checks = self._leakage_checks(train, test, train_features, test_features, manifest.to_dict())
            issues.extend(leakage_checks["issues"])
            warnings.extend(leakage_checks["warnings"])
            issues.extend(self._drift_issues(raw_drift, engineered_drift))
            risk_level = self._risk_level(issues, warnings, raw_drift, engineered_drift)
            recommended_drop_features = self._recommended_drop_features(engineered_drift, leakage_checks)
            audit = {
                "competition_name": manifest.competition_name,
                "status": "completed",
                "task_id": task_id,
                "risk_level": risk_level,
                "manifest_summary": {
                    "task_type": manifest.task_type,
                    "target_column": manifest.target_column,
                    "id_column": manifest.id_column,
                    "metric_candidates": manifest.metric_candidates,
                },
                "raw_train_test_drift": raw_drift,
                "engineered_train_test_drift": engineered_drift,
                "leakage_checks": leakage_checks,
                "recommended_drop_features": recommended_drop_features,
                "issues": issues,
                "warnings": warnings,
                "next_action": self._next_action(risk_level, recommended_drop_features),
            }

        if "risk_level" not in audit:
            audit["risk_level"] = "high" if issues else "low"
        audit_path = experiment_dir / "leakage_report.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        root_audit_path = self.competition_dir / "tabular_feature_leakage_audit.json"
        root_audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "pass" if audit["risk_level"] in {"low", "medium"} else "needs_review"
        ledger_entry = self.ledger.create_entry(
            task_id=task_id,
            agent="tabular_feature_leakage_auditor",
            title="Audit tabular feature leakage and drift",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "task_id": task_id,
                "derived_features_checked": sorted(self.DERIVED_FEATURES),
            },
            prompt=(
                "Audit train/test drift, derived feature transform-scope risks, target leakage indicators, "
                "and feature drop recommendations before further leaderboard-driven optimization."
            ),
            scorecard={
                "agent": "tabular_feature_leakage_auditor",
                "task_id": task_id,
                "status": status,
                "scores": {
                    "risk_level": audit.get("risk_level"),
                    "raw_max_drift": audit.get("raw_train_test_drift", {}).get("max_drift_score"),
                    "engineered_max_drift": audit.get("engineered_train_test_drift", {}).get("max_drift_score"),
                    "recommended_drop_count": len(audit.get("recommended_drop_features", [])),
                    "leakage_issue_count": len(audit.get("leakage_checks", {}).get("issues", [])),
                },
                "metric_name": (manifest.metric_candidates or [None])[0],
                "local_score": None,
                "issues": audit.get("issues", []) + audit.get("warnings", []),
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "leakage_report": audit_path,
                "tabular_feature_leakage_audit": root_audit_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id=task_id,
                status=status,
                metric_name=(manifest.metric_candidates or [None])[0],
                brain_review_path=str(audit_path),
                artifacts=[
                    str(audit_path),
                    str(root_audit_path),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=audit.get("next_action", ""),
            )
        )
        return TabularFeatureLeakageAuditResult(status=status, audit_path=audit_path, experiment_dir=experiment_dir)

    def _leakage_checks(
        self,
        train: List[Dict[str, str]],
        test: List[Dict[str, str]],
        train_features: List[Dict[str, str]],
        test_features: List[Dict[str, str]],
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        target = manifest.get("target_column")
        id_column = manifest.get("id_column")
        train_columns = set(train[0]) if train else set()
        test_columns = set(test[0]) if test else set()
        issues = []
        warnings = []
        if target and target in test_columns:
            issues.append(f"Target column {target} appears in test.csv.")
        if id_column:
            train_ids = [row.get(id_column, "") for row in train if row.get(id_column, "") != ""]
            test_ids = [row.get(id_column, "") for row in test if row.get(id_column, "") != ""]
            duplicate_train_ids = len(train_ids) - len(set(train_ids))
            duplicate_test_ids = len(test_ids) - len(set(test_ids))
            overlap = sorted(set(train_ids) & set(test_ids))[:10]
            if duplicate_train_ids:
                issues.append(f"Train ID column has {duplicate_train_ids} duplicates.")
            if duplicate_test_ids:
                issues.append(f"Test ID column has {duplicate_test_ids} duplicates.")
            if overlap:
                warnings.append(f"Train/test ID overlap detected for {len(overlap)} sample IDs.")
        suspicious_columns = sorted(
            column
            for column in train_columns
            if column != target and re.search(r"(^target$|^label$|survived|outcome|response|^class$|_class$)", column, re.I)
        )
        for column in suspicious_columns:
            warnings.append(f"Column {column} looks target-like; verify it is allowed as a feature.")
        derived_present = [name for name in self.DERIVED_FEATURES if train_features and name in train_features[0]]
        transform_scope_risks = [
            {"feature": name, "risk": self.FIT_SCOPE_RISKS[name]}
            for name in derived_present
            if name in self.FIT_SCOPE_RISKS
        ]
        for item in transform_scope_risks:
            warnings.append(item["risk"])
        return {
            "issues": issues,
            "warnings": warnings,
            "target_in_test": bool(target and target in test_columns),
            "target_like_columns": suspicious_columns,
            "derived_features_present": derived_present,
            "transform_scope_risks": transform_scope_risks,
        }

    def _drift_report(
        self,
        train: List[Dict[str, str]],
        test: List[Dict[str, str]],
        target_column: Optional[str],
        id_column: Optional[str],
        only_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not train or not test:
            return {"status": "missing_data", "feature_count": 0, "max_drift_score": None, "top_features": []}
        ignored = {target_column, id_column, None, ""}
        train_columns = list(train[0].keys())
        test_columns = set(test[0].keys())
        candidates = only_columns if only_columns is not None else train_columns
        features = []
        for column in candidates:
            if column in ignored or column not in test_columns:
                continue
            train_values = [row.get(column, "") for row in train]
            test_values = [row.get(column, "") for row in test]
            if self._mostly_numeric(train_values):
                item = self._numeric_drift(column, train_values, test_values)
            else:
                item = self._categorical_drift(column, train_values, test_values)
            sources = self.DERIVED_FEATURES.get(column)
            if sources:
                item["derived_from"] = sources
            features.append(item)
        features.sort(key=lambda item: item["drift_score"], reverse=True)
        return {
            "status": "completed",
            "feature_count": len(features),
            "max_drift_score": features[0]["drift_score"] if features else None,
            "top_features": features[:20],
        }

    def _recommended_drop_features(
        self,
        engineered_drift: Dict[str, Any],
        leakage_checks: Dict[str, Any],
    ) -> List[str]:
        drops = []
        for item in engineered_drift.get("top_features", []):
            if item.get("drift_score", 0) > 0.5:
                drops.append(item["feature"])
        for item in leakage_checks.get("transform_scope_risks", []):
            feature = item.get("feature")
            if feature and feature not in drops:
                drops.append(feature)
        return drops

    @staticmethod
    def _drift_issues(raw_drift: Dict[str, Any], engineered_drift: Dict[str, Any]) -> List[str]:
        issues = []
        raw_max = raw_drift.get("max_drift_score")
        engineered_max = engineered_drift.get("max_drift_score")
        if isinstance(raw_max, (int, float)) and raw_max > 0.75:
            issues.append(f"Raw train/test drift is high: max_drift_score={raw_max:.4f}.")
        if isinstance(engineered_max, (int, float)) and engineered_max > 0.5:
            issues.append(f"Engineered feature drift is elevated: max_drift_score={engineered_max:.4f}.")
        return issues

    @staticmethod
    def _risk_level(
        issues: List[str],
        warnings: List[str],
        raw_drift: Dict[str, Any],
        engineered_drift: Dict[str, Any],
    ) -> str:
        if any("Target column" in issue for issue in issues):
            return "high"
        engineered_max = engineered_drift.get("max_drift_score")
        raw_max = raw_drift.get("max_drift_score")
        if isinstance(engineered_max, (int, float)) and engineered_max > 0.75:
            return "high"
        if len(issues) >= 2:
            return "high"
        if issues or warnings or (isinstance(raw_max, (int, float)) and raw_max > 0.5):
            return "medium"
        return "low"

    @staticmethod
    def _next_action(risk_level: str, drops: List[str]) -> str:
        if risk_level == "high":
            return "Patch feature engineering before another leaderboard submission."
        if drops:
            return "Run a stable/pruned tabular search after dropping risky engineered features: " + ", ".join(drops) + "."
        return "No strong leakage signal found; continue controlled optimization."

    @staticmethod
    def _empty_audit(competition_name: str, issues: List[str]) -> Dict[str, Any]:
        return {
            "competition_name": competition_name,
            "status": "missing_data",
            "risk_level": "high",
            "raw_train_test_drift": {},
            "engineered_train_test_drift": {},
            "leakage_checks": {"issues": issues, "warnings": []},
            "recommended_drop_features": [],
            "issues": issues,
            "warnings": [],
            "next_action": "Provide train.csv and test.csv before leakage audit.",
        }

    @staticmethod
    def _add_common_features(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out = []
        titles = []
        for row in rows:
            title = "Unknown"
            name = row.get("Name", "")
            match = re.search(r",\s*([^\.]+)\.", str(name))
            if match:
                title = match.group(1)
            titles.append(title)
        title_counts = {title: titles.count(title) for title in set(titles)}
        for index, row in enumerate(rows):
            item = dict(row)
            sibsp = TabularFeatureLeakageAuditor._to_float(row.get("SibSp"))
            parch = TabularFeatureLeakageAuditor._to_float(row.get("Parch"))
            if sibsp is not None and parch is not None:
                family_size = sibsp + parch + 1
                item["FamilySize"] = str(family_size)
                item["IsAlone"] = "1" if family_size == 1 else "0"
            if "Name" in row:
                title = titles[index]
                item["Title"] = title if title_counts.get(title, 0) >= 10 else "Rare"
            if "Cabin" in row:
                cabin = str(row.get("Cabin", "")).strip()
                item["HasCabin"] = "1" if cabin else "0"
                item["CabinDeck"] = cabin[0] if cabin else "Unknown"
            if "Ticket" in row:
                prefix = re.sub(r"[0-9./]", "", str(row.get("Ticket", ""))).strip()
                item["TicketPrefix"] = prefix if prefix else "NONE"
            out.append(item)
        return out

    @staticmethod
    def _numeric_drift(column: str, train: List[str], test: List[str]) -> Dict[str, Any]:
        train_num = TabularFeatureLeakageAuditor._to_numbers(train)
        test_num = TabularFeatureLeakageAuditor._to_numbers(test)
        train_std = TabularFeatureLeakageAuditor._std(train_num)
        mean_gap = abs(TabularFeatureLeakageAuditor._mean(train_num) - TabularFeatureLeakageAuditor._mean(test_num))
        normalized_gap = mean_gap / max(train_std, 1e-9)
        missing_gap = abs(
            TabularFeatureLeakageAuditor._missing_rate(train)
            - TabularFeatureLeakageAuditor._missing_rate(test)
        )
        return {
            "feature": column,
            "kind": "numeric",
            "drift_score": float(normalized_gap + missing_gap),
            "mean_gap_in_train_std": float(normalized_gap),
            "missing_rate_gap": float(missing_gap),
        }

    @staticmethod
    def _categorical_drift(column: str, train: List[str], test: List[str]) -> Dict[str, Any]:
        train_str = [TabularFeatureLeakageAuditor._clean_category(value) for value in train]
        test_str = [TabularFeatureLeakageAuditor._clean_category(value) for value in test]
        train_values = set(train_str)
        test_values = set(test_str)
        unseen = test_values - train_values
        unseen_rate = sum(value in unseen for value in test_str) / max(len(test_str), 1)
        train_top_value = TabularFeatureLeakageAuditor._top_value(train_str)
        test_top_value = TabularFeatureLeakageAuditor._top_value(test_str)
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

    @staticmethod
    def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _mostly_numeric(values: List[str]) -> bool:
        non_missing = [value for value in values if str(value).strip() != ""]
        if not non_missing:
            return False
        numeric = sum(TabularFeatureLeakageAuditor._to_float(value) is not None for value in non_missing)
        return numeric / len(non_missing) >= 0.9

    @staticmethod
    def _to_numbers(values: List[str]) -> List[float]:
        numbers = [TabularFeatureLeakageAuditor._to_float(value) for value in values]
        return [value for value in numbers if value is not None]

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if str(value).strip() == "":
                return None
            parsed = float(value)
            if math.isnan(parsed):
                return None
            return parsed
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = TabularFeatureLeakageAuditor._mean(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))

    @staticmethod
    def _missing_rate(values: List[str]) -> float:
        return sum(str(value).strip() == "" for value in values) / max(len(values), 1)

    @staticmethod
    def _clean_category(value: str) -> str:
        value = str(value).strip()
        return value if value else "__MISSING__"

    @staticmethod
    def _top_value(values: List[str]) -> str:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda item: item[1])[0]
