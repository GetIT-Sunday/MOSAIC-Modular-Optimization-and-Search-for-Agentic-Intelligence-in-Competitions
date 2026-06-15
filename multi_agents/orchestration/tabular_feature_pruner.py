from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .validator import SubmissionValidator, ValidationResult


@dataclass(frozen=True)
class TabularFeaturePruneResult:
    task_id: str
    status: str
    experiment_dir: Path
    run_script: Path
    run_log: Path
    validation_report: Path
    submission_path: Optional[Path]
    validator_result: ValidationResult


class TabularFeaturePruner:
    """Estimate feature value and compare all-features vs pruned tabular runs."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self, task_id: str = "tabular_feature_prune_v1") -> TabularFeaturePruneResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        manifest.write_json(self.competition_dir / "data_manifest.json")

        experiment_dir = self.competition_dir / "experiments" / task_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        run_script = experiment_dir / "run.py"
        run_log = experiment_dir / "run.log"
        validation_report = experiment_dir / "validation_report.json"
        submission_path = experiment_dir / "submission.csv"
        validator_result_path = experiment_dir / "validator_result.json"

        run_script.write_text(FEATURE_PRUNER_SCRIPT.replace("__TASK_ID__", task_id), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(run_script)],
            cwd=str(self.competition_dir),
            capture_output=True,
            text=True,
            timeout=1800,
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
            agent="tabular_feature_pruner",
            title="Run tabular feature importance and pruning check",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "manifest": manifest.to_dict(),
                "method": "permutation_importance_then_cv_prune_compare",
            },
            prompt=(
                "Estimate original-column feature importance, compare all-feature and pruned-feature CV, "
                "write a selected submission, and record pruning risk."
            ),
            scorecard=self._scorecard_for(task_id, status, report, validator_result),
            artifacts={
                "run": run_script,
                "run_log": run_log,
                "validation_report": validation_report,
                "submission": submission_path,
                "validator_result": validator_result_path,
                "feature_report": experiment_dir / "feature_report.json",
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
                    str(experiment_dir / "feature_report.json"),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=report.get("notes", ""),
            )
        )
        return TabularFeaturePruneResult(
            task_id=task_id,
            status=status,
            experiment_dir=experiment_dir,
            run_script=run_script,
            run_log=run_log,
            validation_report=validation_report,
            submission_path=submission_path if submission_path.exists() else None,
            validator_result=validator_result,
        )

    def _status(
        self,
        returncode: int,
        report: Dict[str, Any],
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
        task_id: str,
        status: str,
        report: Dict[str, Any],
        validator_result: ValidationResult,
    ) -> Dict[str, Any]:
        return {
            "agent": "tabular_feature_pruner",
            "task_id": task_id,
            "status": "pass" if status == "validated" else "needs_review",
            "scores": {
                "script_runs": 5 if status != "failed" else 1,
                "submission_valid": 5 if validator_result.ok else 1,
                "feature_count": report.get("feature_count"),
                "kept_feature_count": report.get("kept_feature_count"),
                "pruned_feature_count": report.get("pruned_feature_count"),
            },
            "metric_name": report.get("metric_name"),
            "local_score": report.get("local_score"),
            "issues": validator_result.errors + validator_result.warnings + report.get("issues", []),
            "recommended_human_action": "continue" if status == "validated" else "patch_prompt",
        }

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


FEATURE_PRUNER_SCRIPT = r'''
from pathlib import Path
import json
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required feature pruning dependencies unavailable: {exc}"],
        "notes": "Install pandas and scikit-learn to enable feature pruning.",
    }
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (experiment_dir / "feature_report.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

id_column = manifest["id_column"]
target_column = manifest["target_column"]
metric_name = manifest["metric_candidates"][0]
task_type = manifest["task_type"]
submission_columns = manifest["submission_columns"]
prediction_column = [column for column in submission_columns if column != id_column][0]
lower_is_better = metric_name in {"rmse", "rmsle", "mae"}

train = pd.read_csv(root / "train.csv")
test = pd.read_csv(root / "test.csv")

def add_features(df):
    out = df.copy()
    if "SibSp" in out.columns and "Parch" in out.columns:
        out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    if "Name" in out.columns:
        out["Title"] = out["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Unknown")
        rare_titles = out["Title"].value_counts()
        rare_titles = set(rare_titles[rare_titles < 10].index)
        out["Title"] = out["Title"].where(~out["Title"].isin(rare_titles), "Rare")
    if "Cabin" in out.columns:
        out["HasCabin"] = out["Cabin"].notna().astype(int)
        out["CabinDeck"] = out["Cabin"].astype(str).str[0].replace("n", "Unknown")
    if "Ticket" in out.columns:
        out["TicketPrefix"] = (
            out["Ticket"].astype(str).str.replace(r"[0-9./]", "", regex=True).str.strip().replace("", "NONE")
        )
    return out

train = add_features(train)
test = add_features(test)
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

def make_preprocess(columns):
    cat = [column for column in columns if X[column].dtype == "object"]
    num = [column for column in columns if column not in cat]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), cat),
        ]
    )

def make_model():
    if task_type == "regression":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

def make_pipe(columns):
    return Pipeline([("preprocess", make_preprocess(columns)), ("model", make_model())])

def scoring_name():
    if task_type == "regression":
        return "neg_root_mean_squared_error"
    if metric_name == "roc_auc" and y.nunique() == 2:
        return "roc_auc"
    return "accuracy"

def score_predictions(y_true, pred):
    if task_type == "regression":
        return float(mean_squared_error(y_true, pred) ** 0.5)
    if metric_name == "roc_auc" and y.nunique() == 2:
        return float(roc_auc_score(y_true, pred))
    return float(accuracy_score(y_true, pred))

def predict_for_metric(pipe, data):
    if task_type == "regression":
        return pipe.predict(data)
    if metric_name == "roc_auc" and y.nunique() == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(data)[:, 1]
    return pipe.predict(data)

def cv_score(columns):
    if task_type == "classification" and y.nunique() <= 20:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(make_pipe(columns), X[columns], y, cv=cv, scoring=scoring_name(), n_jobs=1)
    if task_type == "regression":
        values = [-float(score) for score in scores]
        return float(np.mean(values)), values
    return float(np.mean(scores)), [float(score) for score in scores]

all_columns = list(X.columns)
stratify = y if task_type == "classification" and y.nunique() <= 20 else None
X_train, X_valid, y_train, y_valid = train_test_split(
    X[all_columns],
    y,
    test_size=0.25,
    random_state=42,
    stratify=stratify,
)
importance_pipe = make_pipe(all_columns)
importance_pipe.fit(X_train, y_train)
perm = permutation_importance(
    importance_pipe,
    X_valid,
    y_valid,
    scoring=scoring_name(),
    n_repeats=3,
    random_state=42,
    n_jobs=1,
)
feature_rows = []
for column, mean_importance, std_importance in zip(all_columns, perm.importances_mean, perm.importances_std):
    feature_rows.append({
        "feature": column,
        "importance_mean": float(mean_importance),
        "importance_std": float(std_importance),
    })
feature_rows = sorted(feature_rows, key=lambda item: item["importance_mean"], reverse=True)

positive = [item["feature"] for item in feature_rows if item["importance_mean"] > 0]
minimum_keep = min(len(all_columns), max(3, len(all_columns) // 2))
if len(positive) < minimum_keep:
    kept_columns = [item["feature"] for item in feature_rows[:minimum_keep]]
else:
    kept_columns = positive
pruned_columns = [column for column in all_columns if column not in kept_columns]

all_score, all_fold_scores = cv_score(all_columns)
pruned_score, pruned_fold_scores = cv_score(kept_columns)
use_pruned = pruned_score < all_score if lower_is_better else pruned_score > all_score
selected_columns = kept_columns if use_pruned else all_columns
selected_score = pruned_score if use_pruned else all_score

final_pipe = make_pipe(selected_columns)
final_pipe.fit(X[selected_columns], y)
test_pred = predict_for_metric(final_pipe, X_test[selected_columns])
submission = pd.DataFrame({id_column: test[id_column], prediction_column: test_pred})
submission.to_csv(experiment_dir / "submission.csv", index=False)

feature_report = {
    "feature_importance": feature_rows,
    "kept_features": kept_columns,
    "pruned_features": pruned_columns,
    "all_features": {
        "score": all_score,
        "fold_scores": all_fold_scores,
    },
    "pruned_features_result": {
        "score": pruned_score,
        "fold_scores": pruned_fold_scores,
    },
    "selected_feature_set": "pruned" if use_pruned else "all",
}
(experiment_dir / "feature_report.json").write_text(json.dumps(feature_report, indent=2), encoding="utf-8")

report = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": selected_score,
    "selected_feature_set": "pruned" if use_pruned else "all",
    "feature_count": len(all_columns),
    "kept_feature_count": len(kept_columns),
    "pruned_feature_count": len(pruned_columns),
    "top_features": feature_rows[:10],
    "dropped_high_cardinality_columns": high_cardinality,
    "issues": [],
    "notes": "Permutation-importance feature audit with all-vs-pruned 5-fold CV comparison.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''
