from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .ingestion import DataManifest


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


class SubmissionValidator:
    def __init__(self, manifest: DataManifest):
        self.manifest = manifest

    def validate(self, submission_path: Path) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        if not submission_path.exists():
            return ValidationResult(False, [f"submission does not exist: {submission_path}"], [])

        header, rows = self._read_csv(submission_path)
        expected_columns = self.manifest.submission_columns
        if expected_columns and header != expected_columns:
            errors.append(f"columns mismatch: expected {expected_columns}, got {header}")

        expected_rows = self.manifest.tables["test.csv"].row_count
        if expected_rows and len(rows) != expected_rows:
            errors.append(f"row count mismatch: expected {expected_rows}, got {len(rows)}")

        id_column = self.manifest.id_column
        if id_column != "unknown" and id_column in header:
            ids = [row.get(id_column, "") for row in rows]
            if any(value == "" for value in ids):
                errors.append(f"missing values in id column: {id_column}")
            if len(set(ids)) != len(ids):
                errors.append(f"duplicate values in id column: {id_column}")
            test_ids = self._test_ids(id_column)
            if test_ids is not None and ids != test_ids:
                errors.append(f"id order mismatch for column: {id_column}")
        elif id_column != "unknown":
            errors.append(f"id column missing from submission: {id_column}")

        prediction_columns = [column for column in header if column != id_column]
        for column in prediction_columns:
            values = [row.get(column, "") for row in rows]
            if any(value == "" for value in values):
                errors.append(f"missing predictions in column: {column}")
                continue
            numeric_values = [self._to_float(value) for value in values]
            if self.manifest.task_type == "classification":
                allowed = self._allowed_class_labels()
                if allowed and all(value in allowed for value in values):
                    continue
                if all(value is not None for value in numeric_values):
                    if any(value < 0 or value > 1 for value in numeric_values if value is not None):
                        warnings.append(f"numeric classification predictions outside [0, 1]: {column}")
                elif allowed:
                    errors.append(f"unknown class labels in column {column}; allowed={sorted(allowed)}")
            elif self.manifest.task_type == "regression":
                if any(value is None for value in numeric_values):
                    errors.append(f"non-numeric regression predictions in column: {column}")

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

    def _test_ids(self, id_column: str) -> Optional[List[str]]:
        table = self.manifest.tables["test.csv"]
        if id_column not in table.columns:
            return None
        test_path = Path(self.manifest.competition_dir) / "test.csv"
        _, rows = self._read_csv(test_path)
        return [row.get(id_column, "") for row in rows]

    def _allowed_class_labels(self) -> set[str]:
        target = self.manifest.target_column
        if target == "unknown":
            return set()
        train_path = Path(self.manifest.competition_dir) / "train.csv"
        if not train_path.exists():
            return set()
        _, rows = self._read_csv(train_path)
        labels = {row.get(target, "") for row in rows if row.get(target, "") != ""}
        return labels if 0 < len(labels) <= 50 else set()

    @staticmethod
    def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            return list(reader.fieldnames or []), rows

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except ValueError:
            return None
