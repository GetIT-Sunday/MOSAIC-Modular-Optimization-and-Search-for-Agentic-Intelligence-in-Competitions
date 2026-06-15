from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CsvTableInfo:
    path: str
    exists: bool
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    sample_rows: List[Dict[str, str]] = field(default_factory=list)
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    unique_sample_by_column: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class DataManifest:
    competition_name: str
    competition_dir: str
    files: List[str]
    overview_path: Optional[str]
    tables: Dict[str, CsvTableInfo]
    id_column: str
    target_column: str
    submission_columns: List[str]
    task_type: str
    metric_candidates: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


class CompetitionIngestor:
    def __init__(self, competition_dir: Path):
        self.competition_dir = competition_dir.resolve()

    def build_manifest(self) -> DataManifest:
        files = sorted(
            str(path.relative_to(self.competition_dir))
            for path in self.competition_dir.rglob("*")
            if path.is_file() and self._should_include_file(path)
        )
        overview_path = "overview.txt" if (self.competition_dir / "overview.txt").exists() else None
        overview_text = self._read_text(self.competition_dir / "overview.txt")
        tables = {
            name: self._read_csv_info(name)
            for name in ["train.csv", "test.csv", "sample_submission.csv"]
        }
        id_column = self._infer_id_column(tables)
        target_column = self._infer_target_column(tables)
        task_type = self._infer_task_type(tables, target_column)
        metric_candidates = self._infer_metric_candidates(overview_text, task_type)
        notes = self._build_notes(tables, id_column, target_column, metric_candidates)
        return DataManifest(
            competition_name=self.competition_dir.name,
            competition_dir=str(self.competition_dir),
            files=files,
            overview_path=overview_path,
            tables=tables,
            id_column=id_column,
            target_column=target_column,
            submission_columns=tables["sample_submission.csv"].columns,
            task_type=task_type,
            metric_candidates=metric_candidates,
            notes=notes,
        )

    def _read_csv_info(self, relative_path: str) -> CsvTableInfo:
        path = self.competition_dir / relative_path
        if not path.exists():
            return CsvTableInfo(path=relative_path, exists=False)

        row_count = 0
        sample_rows: List[Dict[str, str]] = []
        missing_by_column: Dict[str, int] = {}
        unique_sample_by_column: Dict[str, set[str]] = {}
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            missing_by_column = {column: 0 for column in columns}
            unique_sample_by_column = {column: set() for column in columns}
            for row in reader:
                row_count += 1
                if len(sample_rows) < 5:
                    sample_rows.append({column: row.get(column, "") for column in columns})
                for column in columns:
                    value = row.get(column, "")
                    if value == "":
                        missing_by_column[column] += 1
                    elif len(unique_sample_by_column[column]) < 50:
                        unique_sample_by_column[column].add(value)

        return CsvTableInfo(
            path=relative_path,
            exists=True,
            row_count=row_count,
            columns=columns,
            sample_rows=sample_rows,
            missing_by_column=missing_by_column,
            unique_sample_by_column={
                column: sorted(values) for column, values in unique_sample_by_column.items()
            },
        )

    def _infer_id_column(self, tables: Dict[str, CsvTableInfo]) -> str:
        submission_columns = tables["sample_submission.csv"].columns
        if submission_columns:
            return submission_columns[0]
        common = set(tables["train.csv"].columns) & set(tables["test.csv"].columns)
        for column in tables["train.csv"].columns:
            if column.lower() in {"id", "passengerid", "row_id"} and column in common:
                return column
        return "unknown"

    def _infer_target_column(self, tables: Dict[str, CsvTableInfo]) -> str:
        train_columns = set(tables["train.csv"].columns)
        test_columns = set(tables["test.csv"].columns)
        candidates = [column for column in tables["train.csv"].columns if column not in test_columns]
        if len(candidates) == 1:
            return candidates[0]
        submission_columns = tables["sample_submission.csv"].columns
        if len(submission_columns) >= 2 and submission_columns[-1] in train_columns:
            return submission_columns[-1]
        return "unknown"

    def _infer_task_type(self, tables: Dict[str, CsvTableInfo], target_column: str) -> str:
        if target_column == "unknown":
            return "unknown"
        values = tables["train.csv"].unique_sample_by_column.get(target_column, [])
        if not values:
            return "unknown"
        unique_values = set(values)
        if all(self._is_float(value) for value in values):
            return "classification" if len(unique_values) <= 20 else "regression"
        return "classification"

    def _infer_metric_candidates(self, overview_text: str, task_type: str) -> List[str]:
        text = overview_text.lower()
        patterns = [
            ("roc_auc", r"\b(auc|area under receiver|roc)\b"),
            ("accuracy", r"\baccuracy\b"),
            ("rmse", r"\b(root mean squared error|rmse)\b"),
            ("rmsle", r"\b(root mean squared logarithmic error|rmsle)\b"),
            ("log_loss", r"\b(log loss|cross entropy)\b"),
            ("f1", r"\bf1\b"),
        ]
        candidates = [name for name, pattern in patterns if re.search(pattern, text)]
        if candidates:
            return candidates
        if task_type == "classification":
            return ["accuracy"]
        if task_type == "regression":
            return ["rmse"]
        return ["unknown"]

    def _build_notes(
        self,
        tables: Dict[str, CsvTableInfo],
        id_column: str,
        target_column: str,
        metric_candidates: List[str],
    ) -> List[str]:
        notes = []
        for name, table in tables.items():
            if not table.exists:
                notes.append(f"missing {name}")
        if id_column == "unknown":
            notes.append("id_column unknown")
        if target_column == "unknown":
            notes.append("target_column unknown")
        if metric_candidates == ["unknown"]:
            notes.append("metric unknown")
        return notes

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    def _should_include_file(self, path: Path) -> bool:
        relative = path.relative_to(self.competition_dir)
        ignored_parts = {
            ".ipynb_checkpoints",
            ".pycache",
            "experiments",
            "manual_submission_package",
            "runs",
        }
        if any(part in ignored_parts for part in relative.parts):
            return False
        if path.suffix in {".pyc", ".html"}:
            return False
        return True

    @staticmethod
    def _is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False
