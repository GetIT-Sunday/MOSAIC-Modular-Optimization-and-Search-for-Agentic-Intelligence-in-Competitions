from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .validator import SubmissionValidator, ValidationResult


@dataclass(frozen=True)
class TabularSearchResult:
    task_id: str
    status: str
    experiment_dir: Path
    run_script: Path
    run_log: Path
    validation_report: Path
    submission_path: Optional[Path]
    validator_result: ValidationResult


class TabularSearchRunner:
    """Run a compact multi-model tabular search and blend."""

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
        task_id: str = "tabular_model_search_v1",
        cv_seeds: Optional[Iterable[int]] = None,
        feature_set: str = "all",
    ) -> TabularSearchResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        manifest.write_json(self.competition_dir / "data_manifest.json")
        seeds = list(cv_seeds or [42])
        if feature_set not in {"all", "pruned", "stable", "leakage_safe"}:
            raise ValueError(f"Unsupported feature_set: {feature_set}")

        experiment_dir = self.competition_dir / "experiments" / task_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        run_script = experiment_dir / "run.py"
        run_log = experiment_dir / "run.log"
        validation_report = experiment_dir / "validation_report.json"
        submission_path = experiment_dir / "submission.csv"
        validator_result_path = experiment_dir / "validator_result.json"

        run_script.write_text(
            TABULAR_SEARCH_SCRIPT
            .replace("__TASK_ID__", task_id)
            .replace("__CV_SEEDS__", json.dumps(seeds))
            .replace("__FEATURE_SET__", json.dumps(feature_set)),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, str(run_script)],
            cwd=str(self.competition_dir),
            capture_output=True,
            text=True,
            timeout=2400,
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
            agent="tabular_search_runner",
            title="Run tabular multi-model search and blend",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "manifest": manifest.to_dict(),
                "cv_seeds": seeds,
                "feature_set": feature_set,
                "search_space": ["logistic_or_ridge", "random_forest", "extra_trees", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost"],
            },
            prompt=self._prompt_for(task_id),
            scorecard=self._scorecard_for(task_id, status, report, validator_result),
            artifacts={
                "run": run_script,
                "run_log": run_log,
                "validation_report": validation_report,
                "submission": submission_path,
                "validator_result": validator_result_path,
                "model_report": experiment_dir / "model_report.json",
                "ensemble_report": experiment_dir / "ensemble_report.json",
                "oof_predictions": experiment_dir / "oof_predictions.csv",
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
                    str(experiment_dir / "model_report.json"),
                    str(experiment_dir / "ensemble_report.json"),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=report.get("notes", ""),
            )
        )
        return TabularSearchResult(
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
            "agent": "tabular_search_runner",
            "task_id": task_id,
            "status": "pass" if status == "validated" else "needs_review",
            "scores": {
                "script_runs": 5 if status != "failed" else 1,
                "submission_valid": 5 if validator_result.ok else 1,
                "models_completed": len(report.get("models", [])),
                "ensemble_candidates": len(report.get("ensemble_candidates", [])),
                "cv_seeds": len(report.get("cv_seeds", [])),
                "feature_set": report.get("feature_set", "unknown"),
                "blend_score_available": 5 if report.get("local_score") is not None else 1,
            },
            "metric_name": report.get("metric_name"),
            "local_score": report.get("local_score"),
            "issues": validator_result.errors + validator_result.warnings + report.get("issues", []),
            "recommended_human_action": "continue" if status == "validated" else "patch_prompt",
        }

    def _prompt_for(self, task_id: str) -> str:
        return (
            f"Run `{task_id}` as a controlled tabular search on remote Linux.\n\n"
            "Train several strong tabular models with shared preprocessing, record fold scores, "
            "write repeated-CV OOF predictions, choose the best/blend/stacking submission, validate it, and expose all artifacts in the Run Ledger.\n"
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


TABULAR_SEARCH_SCRIPT = r'''
from pathlib import Path
import json
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
requested_feature_set = __FEATURE_SET__

try:
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "models": [],
        "feature_set": requested_feature_set,
        "issues": [f"required tabular search dependencies unavailable: {exc}"],
        "notes": "Install pandas, numpy, and scikit-learn to enable tabular search.",
    }
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (experiment_dir / "model_report.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    (experiment_dir / "ensemble_report.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

id_column = manifest["id_column"]
target_column = manifest["target_column"]
metric_name = manifest["metric_candidates"][0]
task_type = manifest["task_type"]
submission_columns = manifest["submission_columns"]
prediction_column = [column for column in submission_columns if column != id_column][0]
lower_is_better = metric_name in {"rmse", "rmsle", "mae"}
cv_seeds = __CV_SEEDS__

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
feature_report_path = root / "experiments" / "tabular_feature_prune_v1" / "feature_report.json"
stable_feature_report_path = root / "experiments" / "stability_first_features_v1" / "feature_report.json"
leakage_audit_path = root / "tabular_feature_leakage_audit.json"
feature_set_source = None
feature_set_issues = []
requested_kept_features = []
requested_drop_features = []
if requested_feature_set == "pruned":
    if feature_report_path.exists():
        feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
        requested_kept_features = [
            column for column in feature_report.get("kept_features", []) if column in X.columns
        ]
        if requested_kept_features:
            X = X[requested_kept_features]
            X_test = X_test[requested_kept_features]
            feature_set_source = str(feature_report_path)
        else:
            feature_set_issues.append("Requested pruned feature set, but feature_report.json had no usable kept_features; falling back to all features.")
            requested_feature_set = "all"
    else:
        feature_set_issues.append("Requested pruned feature set, but feature_report.json was missing; falling back to all features.")
        requested_feature_set = "all"
elif requested_feature_set == "stable":
    if stable_feature_report_path.exists():
        feature_report = json.loads(stable_feature_report_path.read_text(encoding="utf-8"))
        requested_drop_features = [
            column for column in feature_report.get("drop_features", []) if column in X.columns
        ]
        if requested_drop_features:
            X = X.drop(columns=requested_drop_features)
            X_test = X_test.drop(columns=requested_drop_features, errors="ignore")
            feature_set_source = str(stable_feature_report_path)
        else:
            feature_set_issues.append("Requested stable feature set, but feature_report.json had no usable drop_features; using all features.")
            requested_feature_set = "all"
    else:
        feature_set_issues.append("Requested stable feature set, but stability_first feature_report.json was missing; falling back to all features.")
        requested_feature_set = "all"
elif requested_feature_set == "leakage_safe":
    if leakage_audit_path.exists():
        leakage_audit = json.loads(leakage_audit_path.read_text(encoding="utf-8"))
        requested_drop_features = [
            column for column in leakage_audit.get("recommended_drop_features", []) if column in X.columns
        ]
        if requested_drop_features:
            X = X.drop(columns=requested_drop_features)
            X_test = X_test.drop(columns=requested_drop_features, errors="ignore")
            feature_set_source = str(leakage_audit_path)
        else:
            feature_set_issues.append("Requested leakage_safe feature set, but tabular_feature_leakage_audit.json had no usable recommended_drop_features; using all features.")
            requested_feature_set = "all"
    else:
        feature_set_issues.append("Requested leakage_safe feature set, but tabular_feature_leakage_audit.json was missing; falling back to all features.")
        requested_feature_set = "all"
categorical = [column for column in X.columns if X[column].dtype == "object"]
numeric = [column for column in X.columns if column not in categorical]

def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocess(scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_encoder())]), categorical),
        ]
    )

def optional_model_specs():
    specs = []
    if task_type == "regression":
        specs.extend([
            ("ridge", Pipeline([("preprocess", preprocess(scale_numeric=True)), ("model", Ridge(alpha=1.0))])),
            ("random_forest", Pipeline([("preprocess", preprocess()), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1))])),
            ("extra_trees", Pipeline([("preprocess", preprocess()), ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1))])),
            ("hist_gradient_boosting", Pipeline([("preprocess", preprocess()), ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, random_state=42))])),
        ])
    else:
        specs.extend([
            ("logistic", Pipeline([("preprocess", preprocess(scale_numeric=True)), ("model", LogisticRegression(max_iter=1500, n_jobs=1))])),
            ("random_forest", Pipeline([("preprocess", preprocess()), ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])),
            ("extra_trees", Pipeline([("preprocess", preprocess()), ("model", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))])),
            ("hist_gradient_boosting", Pipeline([("preprocess", preprocess()), ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, random_state=42))])),
        ])
    try:
        if task_type == "regression":
            from lightgbm import LGBMRegressor
            specs.append(("lightgbm", Pipeline([("preprocess", preprocess()), ("model", LGBMRegressor(n_estimators=500, learning_rate=0.025, random_state=42, n_jobs=-1, verbose=-1))])))
        else:
            from lightgbm import LGBMClassifier
            specs.append(("lightgbm", Pipeline([("preprocess", preprocess()), ("model", LGBMClassifier(n_estimators=500, learning_rate=0.025, random_state=42, n_jobs=-1, verbose=-1))])))
    except Exception:
        pass
    try:
        if task_type == "regression":
            from xgboost import XGBRegressor
            specs.append(("xgboost", Pipeline([("preprocess", preprocess()), ("model", XGBRegressor(n_estimators=500, learning_rate=0.025, max_depth=3, random_state=42, n_jobs=-1, eval_metric="rmse"))])))
        else:
            from xgboost import XGBClassifier
            specs.append(("xgboost", Pipeline([("preprocess", preprocess()), ("model", XGBClassifier(n_estimators=500, learning_rate=0.025, max_depth=3, random_state=42, n_jobs=-1, eval_metric="logloss"))])))
    except Exception:
        pass
    try:
        if task_type == "regression":
            from catboost import CatBoostRegressor
            specs.append(("catboost", Pipeline([("preprocess", preprocess()), ("model", CatBoostRegressor(iterations=500, learning_rate=0.025, random_seed=42, verbose=False))])))
        else:
            from catboost import CatBoostClassifier
            specs.append(("catboost", Pipeline([("preprocess", preprocess()), ("model", CatBoostClassifier(iterations=500, learning_rate=0.025, random_seed=42, verbose=False))])))
    except Exception:
        pass
    return specs

def score_predictions(y_true, pred):
    if task_type == "regression":
        return float(mean_squared_error(y_true, pred) ** 0.5)
    if metric_name == "roc_auc" and y.nunique() == 2:
        return float(roc_auc_score(y_true, pred))
    return float(accuracy_score(y_true, pred))

def model_prediction(pipe, data, proba_for_binary):
    if task_type == "regression":
        return pipe.predict(data)
    if proba_for_binary and hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(data)[:, 1]
    return pipe.predict(data)

def splits_for_seed(seed):
    if task_type == "classification" and y.nunique() <= 20:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(cv.split(X, y))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(X))

model_reports = []
oof_frame = pd.DataFrame({id_column: train[id_column] if id_column in train.columns else np.arange(len(train)), target_column: y})
test_outputs = {}
proba_for_binary = task_type == "classification" and y.nunique() == 2 and metric_name in {"roc_auc", "log_loss"}
label_threshold_mode = task_type == "classification" and y.nunique() == 2 and metric_name not in {"roc_auc", "log_loss"}

for name, base_pipe in optional_model_specs():
    try:
        fold_scores = []
        seed_scores = []
        oof_accumulator = np.zeros(len(X), dtype=float if (task_type == "regression" or y.nunique() == 2) else object)
        test_fold_preds = []
        for seed in cv_seeds:
            seed_oof = np.zeros(len(X), dtype=float if (task_type == "regression" or y.nunique() == 2) else object)
            seed_fold_scores = []
            for fold, (train_idx, valid_idx) in enumerate(splits_for_seed(seed)):
                pipe = clone(base_pipe)
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                if label_threshold_mode and hasattr(pipe.named_steps["model"], "predict_proba"):
                    valid_raw = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
                    valid_pred = (valid_raw >= 0.5).astype(int)
                    test_pred = pipe.predict_proba(X_test)[:, 1]
                    seed_oof[valid_idx] = valid_raw
                else:
                    valid_pred = model_prediction(pipe, X.iloc[valid_idx], proba_for_binary)
                    test_pred = model_prediction(pipe, X_test, proba_for_binary or label_threshold_mode)
                    seed_oof[valid_idx] = valid_pred
                score = score_predictions(y.iloc[valid_idx], valid_pred)
                fold_scores.append(score)
                seed_fold_scores.append(score)
                test_fold_preds.append(np.asarray(test_pred))
            if label_threshold_mode:
                seed_score = score_predictions(y, (seed_oof >= 0.5).astype(int))
            else:
                seed_score = score_predictions(y, seed_oof)
            seed_scores.append({"seed": int(seed), "score": float(seed_score), "fold_scores": [float(item) for item in seed_fold_scores]})
            if np.issubdtype(seed_oof.dtype, np.number):
                oof_accumulator = oof_accumulator.astype(float) + seed_oof.astype(float)
            else:
                oof_accumulator = seed_oof
        oof_raw = oof_accumulator / len(cv_seeds) if np.issubdtype(oof_accumulator.dtype, np.number) else oof_accumulator
        if label_threshold_mode:
            score = score_predictions(y, (oof_raw >= 0.5).astype(int))
        else:
            score = score_predictions(y, oof_raw)
        test_pred_mean = np.mean(np.vstack(test_fold_preds), axis=0)
        model_reports.append({
            "model": name,
            "score": float(score),
            "fold_scores": [float(item) for item in fold_scores],
            "seed_scores": seed_scores,
            "status": "completed",
        })
        oof_frame[f"oof_{name}"] = oof_raw
        test_outputs[name] = test_pred_mean
    except Exception as exc:
        model_reports.append({"model": name, "status": "failed", "error": str(exc)})

completed = [item for item in model_reports if item.get("status") == "completed"]
if not completed:
    report = {
        "experiment": "__TASK_ID__",
        "status": "skipped",
        "metric_name": metric_name,
        "local_score": None,
        "models": model_reports,
        "issues": ["No model completed successfully."],
        "notes": "All candidate models failed.",
    }
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (experiment_dir / "model_report.json").write_text(json.dumps(model_reports, indent=2), encoding="utf-8")
    (experiment_dir / "ensemble_report.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

completed_sorted = sorted(completed, key=lambda item: item["score"], reverse=not lower_is_better)
best = completed_sorted[0]
labels = sorted(y.dropna().unique()) if task_type == "classification" else []

def final_prediction(raw):
    if task_type == "classification" and y.nunique() == 2 and metric_name not in {"roc_auc", "log_loss"}:
        return np.where(np.asarray(raw, dtype=float) >= 0.5, labels[-1], labels[0])
    return raw

def oof_for_model(name):
    return oof_frame[f"oof_{name}"].to_numpy()

def add_candidate(candidates, name, kind, model_names, oof_raw, test_raw, score=None):
    if score is None:
        score = score_predictions(y, final_prediction(oof_raw))
    candidates.append({
        "name": name,
        "kind": kind,
        "models": model_names,
        "score": float(score),
        "test_prediction": np.asarray(test_raw),
    })

ensemble_candidates = []
add_candidate(
    ensemble_candidates,
    name=f"best_single_{best['model']}",
    kind="best_single",
    model_names=[best["model"]],
    oof_raw=oof_for_model(best["model"]),
    test_raw=test_outputs[best["model"]],
    score=best["score"],
)

numeric_model_names = [
    item["model"]
    for item in completed_sorted
    if np.issubdtype(np.asarray(test_outputs[item["model"]]).dtype, np.number)
]
blend_names = numeric_model_names[: min(3, len(numeric_model_names))]
if len(blend_names) >= 2:
    blend_oof = np.mean(np.vstack([oof_for_model(name).astype(float) for name in blend_names]), axis=0)
    blend_test = np.mean(np.vstack([test_outputs[name].astype(float) for name in blend_names]), axis=0)
    add_candidate(
        ensemble_candidates,
        name="top_model_mean_blend",
        kind="mean_blend",
        model_names=blend_names,
        oof_raw=blend_oof,
        test_raw=blend_test,
    )

stack_names = numeric_model_names[: min(5, len(numeric_model_names))]
if len(stack_names) >= 2 and (task_type == "regression" or y.nunique() == 2):
    try:
        stack_X = np.column_stack([oof_for_model(name).astype(float) for name in stack_names])
        stack_test = np.column_stack([test_outputs[name].astype(float) for name in stack_names])
        stack_oof = np.zeros(len(y), dtype=float)
        if task_type == "regression":
            meta_cv = KFold(n_splits=5, shuffle=True, random_state=123)
            for train_idx, valid_idx in meta_cv.split(stack_X):
                meta = Ridge(alpha=1.0)
                meta.fit(stack_X[train_idx], y.iloc[train_idx])
                stack_oof[valid_idx] = meta.predict(stack_X[valid_idx])
            final_meta = Ridge(alpha=1.0)
            final_meta.fit(stack_X, y)
            stack_test_pred = final_meta.predict(stack_test)
        else:
            meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
            for train_idx, valid_idx in meta_cv.split(stack_X, y):
                meta = LogisticRegression(max_iter=1000, n_jobs=1)
                meta.fit(stack_X[train_idx], y.iloc[train_idx])
                if metric_name in {"roc_auc", "log_loss"}:
                    stack_oof[valid_idx] = meta.predict_proba(stack_X[valid_idx])[:, 1]
                else:
                    stack_oof[valid_idx] = meta.predict(stack_X[valid_idx])
            final_meta = LogisticRegression(max_iter=1000, n_jobs=1)
            final_meta.fit(stack_X, y)
            if metric_name in {"roc_auc", "log_loss"}:
                stack_test_pred = final_meta.predict_proba(stack_test)[:, 1]
            else:
                stack_test_pred = final_meta.predict(stack_test)
        add_candidate(
            ensemble_candidates,
            name="logistic_or_ridge_stacking",
            kind="stacking",
            model_names=stack_names,
            oof_raw=stack_oof,
            test_raw=stack_test_pred,
        )
    except Exception as exc:
        ensemble_candidates.append({
            "name": "logistic_or_ridge_stacking",
            "kind": "stacking",
            "models": stack_names,
            "status": "failed",
            "error": str(exc),
            "score": None,
            "test_prediction": np.asarray([]),
        })

valid_candidates = [item for item in ensemble_candidates if isinstance(item.get("score"), (int, float))]
selected = sorted(valid_candidates, key=lambda item: item["score"], reverse=not lower_is_better)[0]
selected_pred = final_prediction(selected["test_prediction"])

submission = pd.DataFrame({id_column: test[id_column], prediction_column: selected_pred})
submission.to_csv(experiment_dir / "submission.csv", index=False)
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)
(experiment_dir / "model_report.json").write_text(json.dumps(model_reports, indent=2), encoding="utf-8")
ensemble_report = [
    {
        "name": item["name"],
        "kind": item["kind"],
        "models": item["models"],
        "score": item.get("score"),
        "status": item.get("status", "completed"),
        "error": item.get("error"),
    }
    for item in ensemble_candidates
]
(experiment_dir / "ensemble_report.json").write_text(json.dumps(ensemble_report, indent=2), encoding="utf-8")

report = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": float(selected["score"]),
    "feature_set": requested_feature_set,
    "feature_set_source": feature_set_source,
    "feature_count": int(X.shape[1]),
    "requested_kept_features": requested_kept_features,
    "requested_drop_features": requested_drop_features,
    "cv_seeds": [int(seed) for seed in cv_seeds],
    "repeated_cv": {
        "seed_count": len(cv_seeds),
        "folds_per_seed": 5,
        "total_folds_per_model": len(cv_seeds) * 5,
    },
    "best_model": best,
    "selected_submission": {
        "name": selected["name"],
        "kind": selected["kind"],
        "models": selected["models"],
        "score": float(selected["score"]),
    },
    "ensemble_candidates": ensemble_report,
    "blend_models": blend_names,
    "models": completed,
    "failed_models": [item for item in model_reports if item.get("status") != "completed"],
    "dropped_high_cardinality_columns": high_cardinality,
    "issues": feature_set_issues,
    "notes": "Multi-model tabular search with repeated-CV OOF tracking; submission is selected from best single, mean blend, and stacking candidates.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''
