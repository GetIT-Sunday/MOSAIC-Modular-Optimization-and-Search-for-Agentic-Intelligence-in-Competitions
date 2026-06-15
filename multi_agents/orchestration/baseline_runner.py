from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .ingestion import CompetitionIngestor, DataManifest
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .validator import SubmissionValidator, ValidationResult


@dataclass(frozen=True)
class BaselineRunResult:
    task_id: str
    status: str
    experiment_dir: Path
    run_script: Path
    run_log: Path
    validation_report: Path
    submission_path: Optional[Path]
    validator_result: ValidationResult


class TabularBaselineRunner:
    BASELINES = [
        "sample_submission_baseline",
        "target_frequency_or_mean_baseline",
        "sklearn_linear_baseline",
    ]

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run_all(self, baselines: Optional[List[str]] = None) -> List[BaselineRunResult]:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        self._ensure_manifest(manifest)
        selected = baselines or self.BASELINES
        results = []
        for baseline in selected:
            results.append(self.run_baseline(baseline, manifest))
        self._write_best_baseline_review(results, manifest)
        return results

    def run_baseline(
        self,
        baseline: str,
        manifest: Optional[DataManifest] = None,
    ) -> BaselineRunResult:
        manifest = manifest or CompetitionIngestor(self.competition_dir).build_manifest()
        task_id = f"baseline_{baseline}"
        experiment_dir = self.competition_dir / "experiments" / baseline
        experiment_dir.mkdir(parents=True, exist_ok=True)
        run_script = experiment_dir / "run.py"
        run_log = experiment_dir / "run.log"
        validation_report = experiment_dir / "validation_report.json"
        submission_path = experiment_dir / "submission.csv"
        validator_result_path = experiment_dir / "validator_result.json"

        run_script.write_text(self._script_for(baseline), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(run_script)],
            cwd=str(self.competition_dir),
            capture_output=True,
            text=True,
            timeout=900,
        )
        run_log.write_text(
            "\n".join(
                [
                    f"returncode={completed.returncode}",
                    "",
                    "STDOUT:",
                    completed.stdout,
                    "",
                    "STDERR:",
                    completed.stderr,
                ]
            ),
            encoding="utf-8",
        )

        if submission_path.exists():
            validator_result = SubmissionValidator(manifest).validate(submission_path)
        else:
            validator_result = ValidationResult(False, ["submission.csv was not created"], [])
        validator_result.write_json(validator_result_path)

        report = self._read_json(validation_report)
        status = self._status(completed.returncode, report, validator_result)
        ledger_entry = self.ledger.create_entry(
            task_id=task_id,
            agent="baseline_runner",
            title=f"Run {baseline}",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "baseline": baseline,
                "manifest": manifest.to_dict(),
            },
            prompt=self._prompt_for(baseline, manifest),
            scorecard=self._scorecard_for(baseline, status, report, validator_result),
            artifacts={
                "run": run_script,
                "run_log": run_log,
                "validation_report": validation_report,
                "submission": submission_path,
                "validator_result": validator_result_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id=task_id,
                status=status,
                metric_name=report.get("metric_name"),
                local_score=report.get("local_score"),
                script_path=str(run_script),
                submission_path=str(submission_path) if submission_path.exists() else None,
                failure_reason="; ".join(validator_result.errors),
                artifacts=[
                    str(run_script),
                    str(run_log),
                    str(validation_report),
                    str(validator_result_path),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=report.get("notes", ""),
            )
        )
        return BaselineRunResult(
            task_id=task_id,
            status=status,
            experiment_dir=experiment_dir,
            run_script=run_script,
            run_log=run_log,
            validation_report=validation_report,
            submission_path=submission_path if submission_path.exists() else None,
            validator_result=validator_result,
        )

    def _write_best_baseline_review(
        self,
        results: List[BaselineRunResult],
        manifest: DataManifest,
    ) -> None:
        summaries = []
        for result in results:
            report = self._read_json(result.validation_report)
            summaries.append(
                {
                    "task_id": result.task_id,
                    "status": result.status,
                    "metric_name": report.get("metric_name"),
                    "local_score": report.get("local_score"),
                    "submission_valid": result.validator_result.ok,
                }
            )
        valid = [
            item
            for item in summaries
            if item["submission_valid"] and isinstance(item.get("local_score"), (int, float))
        ]
        reverse = manifest.metric_candidates[0] not in {"rmse", "rmsle", "mae"}
        best = sorted(valid, key=lambda item: item["local_score"], reverse=reverse)[0] if valid else None
        review = {
            "competition_name": manifest.competition_name,
            "decision": "continue_to_llm_enhancement" if best else "needs_human_review",
            "best_baseline": best,
            "baselines": summaries,
        }
        path = self.competition_dir / "baseline_review.json"
        path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        self.ledger.create_entry(
            task_id="baseline_selection",
            agent="brain",
            title="Select best baseline for next optimization round",
            status="pass" if best else "needs_review",
            input_payload=review,
            prompt="Review baseline reports, select the best valid baseline, and decide whether to continue to LLM enhancement.",
            scorecard={
                "agent": "brain",
                "task_id": "baseline_selection",
                "status": "pass" if best else "needs_review",
                "scores": {
                    "valid_baseline_count": len(valid),
                    "total_baseline_count": len(summaries),
                    "selection_ready": 5 if best else 1,
                },
                "issues": [] if best else ["No valid scored baseline was produced."],
                "recommended_human_action": "continue" if best else "patch_prompt",
            },
            artifacts={"baseline_review": path},
        )

    def _ensure_manifest(self, manifest: DataManifest) -> None:
        manifest.write_json(self.competition_dir / "data_manifest.json")

    def _status(
        self,
        returncode: int,
        report: Dict,
        validator_result: ValidationResult,
    ) -> str:
        if returncode != 0:
            return "failed"
        if report.get("status") == "skipped":
            return "skipped"
        if not validator_result.ok:
            return "validation_failed"
        return "validated"

    def _scorecard_for(
        self,
        baseline: str,
        status: str,
        report: Dict,
        validator_result: ValidationResult,
    ) -> Dict:
        return {
            "agent": "baseline_runner",
            "task_id": f"baseline_{baseline}",
            "status": "pass" if status == "validated" else "needs_review",
            "scores": {
                "script_runs": 5 if status != "failed" else 1,
                "submission_valid": 5 if validator_result.ok else 1,
                "local_score_available": 5 if report.get("local_score") is not None else 1,
                "reproducibility": 4,
            },
            "metric_name": report.get("metric_name"),
            "local_score": report.get("local_score"),
            "issues": validator_result.errors + validator_result.warnings + report.get("issues", []),
            "recommended_human_action": "continue" if status == "validated" else "patch_prompt",
        }

    def _prompt_for(self, baseline: str, manifest: DataManifest) -> str:
        return (
            f"Implement and run the deterministic tabular baseline `{baseline}`.\n\n"
            f"Competition: {manifest.competition_name}\n"
            f"Task type: {manifest.task_type}\n"
            f"Metric candidates: {', '.join(manifest.metric_candidates)}\n"
            f"ID column: {manifest.id_column}\n"
            f"Target column: {manifest.target_column}\n"
            "Required outputs: run.py, run.log, validation_report.json, submission.csv, validator_result.json.\n"
        )

    def _script_for(self, baseline: str) -> str:
        if baseline == "sample_submission_baseline":
            return SAMPLE_SUBMISSION_SCRIPT
        if baseline == "target_frequency_or_mean_baseline":
            return TARGET_FREQUENCY_SCRIPT
        if baseline == "sklearn_linear_baseline":
            return SKLEARN_LINEAR_SCRIPT
        raise ValueError(f"Unknown baseline: {baseline}")

    @staticmethod
    def _read_json(path: Path) -> Dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


SAMPLE_SUBMISSION_SCRIPT = r'''
from pathlib import Path
import csv
import json
import shutil

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
source = root / "sample_submission.csv"
target = experiment_dir / "submission.csv"
shutil.copyfile(source, target)
rows = sum(1 for _ in csv.DictReader(target.open("r", encoding="utf-8", newline="")))
report = {
    "baseline": "sample_submission_baseline",
    "status": "completed",
    "metric_name": manifest["metric_candidates"][0],
    "local_score": None,
    "row_count": rows,
    "issues": [],
    "notes": "Copied sample_submission.csv as a schema baseline; no local score is computed.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


TARGET_FREQUENCY_SCRIPT = r'''
from pathlib import Path
import csv
import json

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
id_column = manifest["id_column"]
target_column = manifest["target_column"]
metric_name = manifest["metric_candidates"][0]
task_type = manifest["task_type"]
submission_columns = manifest["submission_columns"]

with (root / "train.csv").open("r", encoding="utf-8", errors="ignore", newline="") as handle:
    train_rows = list(csv.DictReader(handle))
with (root / "test.csv").open("r", encoding="utf-8", errors="ignore", newline="") as handle:
    test_rows = list(csv.DictReader(handle))

values = [row[target_column] for row in train_rows if row.get(target_column, "") != ""]
numeric_values = []
for value in values:
    try:
        numeric_values.append(float(value))
    except ValueError:
        pass

issues = []
if task_type == "regression":
    prediction = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
    local_score = None
elif metric_name in {"roc_auc", "log_loss"} and numeric_values:
    prediction = sum(numeric_values) / len(numeric_values)
    local_score = None
else:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    prediction = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else "0"
    local_score = max(counts.values()) / len(values) if counts and values else None

prediction_column = [column for column in submission_columns if column != id_column][0]
with (experiment_dir / "submission.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=submission_columns)
    writer.writeheader()
    for row in test_rows:
        writer.writerow({id_column: row[id_column], prediction_column: prediction})

report = {
    "baseline": "target_frequency_or_mean_baseline",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "prediction": prediction,
    "issues": issues,
    "notes": "Predicts majority class, target frequency, or target mean depending on task and metric.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


SKLEARN_LINEAR_SCRIPT = r'''
from pathlib import Path
import csv
import json
import subprocess
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:
    report = {
        "baseline": "sklearn_linear_baseline",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"sklearn/pandas unavailable: {exc}"],
        "notes": "Install pandas and scikit-learn to enable this baseline.",
    }
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

id_column = manifest["id_column"]
target_column = manifest["target_column"]
metric_name = manifest["metric_candidates"][0]
task_type = manifest["task_type"]
submission_columns = manifest["submission_columns"]
prediction_column = [column for column in submission_columns if column != id_column][0]

train = pd.read_csv(root / "train.csv")
test = pd.read_csv(root / "test.csv")
y = train[target_column]
drop_cols = [target_column]
if id_column in train.columns:
    drop_cols.append(id_column)
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=[id_column], errors="ignore")
categorical = [column for column in X.columns if X[column].dtype == "object"]
high_cardinality = [column for column in categorical if X[column].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality, errors="ignore")
categorical = [column for column in X.columns if X[column].dtype == "object"]
numeric = [column for column in X.columns if column not in categorical]

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
    ]
)

stratify = y if task_type == "classification" and y.nunique() <= 20 else None
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify,
)

if task_type == "regression":
    model = Ridge(alpha=1.0)
else:
    model = LogisticRegression(max_iter=1000, n_jobs=1)

pipe = Pipeline([("preprocess", preprocess), ("model", model)])
pipe.fit(X_train, y_train)

if task_type == "regression":
    valid_pred = pipe.predict(X_valid)
    local_score = float(mean_squared_error(y_valid, valid_pred) ** 0.5)
    test_pred = pipe.predict(X_test)
elif metric_name == "roc_auc" and len(set(y)) == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
    valid_pred = pipe.predict_proba(X_valid)[:, 1]
    local_score = float(roc_auc_score(y_valid, valid_pred))
    test_pred = pipe.predict_proba(X_test)[:, 1]
else:
    valid_pred = pipe.predict(X_valid)
    local_score = float(accuracy_score(y_valid, valid_pred))
    test_pred = pipe.predict(X_test)

submission = pd.DataFrame({id_column: test[id_column], prediction_column: test_pred})
submission.to_csv(experiment_dir / "submission.csv", index=False)
report = {
    "baseline": "sklearn_linear_baseline",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "feature_count": int(X.shape[1]),
    "dropped_high_cardinality_columns": high_cardinality,
    "issues": [],
    "notes": "Linear sklearn baseline with median/mode imputation, scaling, and one-hot encoding.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''
