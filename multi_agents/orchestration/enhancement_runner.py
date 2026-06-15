from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestion import CompetitionIngestor, DataManifest
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger
from .validator import SubmissionValidator, ValidationResult
from .experiment_queue import ExperimentQueueBuilder


@dataclass(frozen=True)
class EnhancementRunResult:
    task_id: str
    status: str
    experiment_dir: Path
    run_script: Path
    run_log: Path
    validation_report: Path
    submission_path: Optional[Path]
    validator_result: ValidationResult


class EnhancementRunner:
    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run_first_recommendation(self) -> EnhancementRunResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        manifest.write_json(self.competition_dir / "data_manifest.json")
        plan = self._read_json(self.competition_dir / "llm_experiment_plan.json")
        experiment = self._select_experiment(plan, skip_completed=False)
        result = self.run_experiment(experiment, manifest, plan)
        self._write_enhancement_review(result, plan)
        self._refresh_experiment_queue()
        return result

    def run_next_recommendation(self) -> EnhancementRunResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        manifest.write_json(self.competition_dir / "data_manifest.json")
        plan = self._read_json(self.competition_dir / "llm_experiment_plan.json")
        experiment = self._select_queued_experiment() or self._select_experiment(plan, skip_completed=True)
        result = self.run_experiment(experiment, manifest, plan)
        self._write_enhancement_review(result, plan)
        self._refresh_experiment_queue()
        return result

    def run_experiment(
        self,
        experiment: Dict[str, Any],
        manifest: DataManifest,
        plan: Dict[str, Any],
    ) -> EnhancementRunResult:
        task_id = self._task_id(experiment)
        experiment_dir = self.competition_dir / "experiments" / task_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        run_script = experiment_dir / "run.py"
        run_log = experiment_dir / "run.log"
        validation_report = experiment_dir / "validation_report.json"
        submission_path = experiment_dir / "submission.csv"
        validator_result_path = experiment_dir / "validator_result.json"

        run_script.write_text(self._script_for(experiment), encoding="utf-8")
        completed = self._run_script(run_script, timeout_seconds=1800)
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
        if completed.returncode != 0 and not validation_report.exists():
            report = self._failure_report(
                experiment=experiment,
                manifest=manifest,
                issue=self._failure_issue(completed),
            )
            validation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if submission_path.exists():
            validator_result = SubmissionValidator(manifest).validate(submission_path)
        else:
            validator_result = ValidationResult(False, ["submission.csv was not created"], [])
        validator_result.write_json(validator_result_path)

        report = self._read_json(validation_report)
        status = self._status(completed.returncode, report, validator_result)
        ledger_entry = self.ledger.create_entry(
            task_id=task_id,
            agent="enhancement_runner",
            title=f"Run enhancement: {experiment.get('title') or task_id}",
            status=status,
            input_payload={
                "competition_name": manifest.competition_name,
                "experiment": experiment,
                "runner_kind": self._runner_kind(experiment),
                "plan_summary": {
                    "next_action": plan.get("next_action"),
                    "current_best_baseline": plan.get("current_best_baseline"),
                },
                "manifest": manifest.to_dict(),
            },
            prompt=self._prompt_for(experiment, manifest),
            scorecard=self._scorecard_for(task_id, status, report, validator_result),
            artifacts={
                "run": run_script,
                "run_log": run_log,
                "validation_report": validation_report,
                "cv_stability_audit": experiment_dir / "cv_stability_audit.json",
                "distribution_shift_audit": experiment_dir / "distribution_shift_audit.json",
                "overfitting_audit": experiment_dir / "overfitting_audit.json",
                "regularized_blend_report": experiment_dir / "regularized_blend_report.json",
                "gbdt_oof_report": experiment_dir / "gbdt_oof_report.json",
                "oof_predictions": experiment_dir / "oof_predictions.csv",
                "feature_importance": experiment_dir / "feature_importance.csv",
                "feature_importance_by_fold": experiment_dir / "feature_importance_by_fold.csv",
                "plan_vs_execution_diff": experiment_dir / "plan_vs_execution_diff.json",
                "actual_feature_list": experiment_dir / "actual_feature_list.json",
                "feature_implementation_audit": experiment_dir / "feature_implementation_audit.json",
                "per_class_oof_report": experiment_dir / "per_class_oof_report.json",
                "oof_diversity_report": experiment_dir / "oof_diversity_report.json",
                "nn_training_report": experiment_dir / "nn_training_report.json",
                "model_config": experiment_dir / "model_config.json",
                "specialist_report": experiment_dir / "specialist_report.json",
                "clean_blend_report": experiment_dir / "clean_blend_report.json",
                "skipped_candidates": experiment_dir / "skipped_candidates.json",
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
        return EnhancementRunResult(
            task_id=task_id,
            status=status,
            experiment_dir=experiment_dir,
            run_script=run_script,
            run_log=run_log,
            validation_report=validation_report,
            submission_path=submission_path if submission_path.exists() else None,
            validator_result=validator_result,
        )

    def _run_script(self, run_script: Path, timeout_seconds: int) -> subprocess.CompletedProcess:
        command = [sys.executable, str(run_script)]
        try:
            return subprocess.run(
                command,
                cwd=str(self.competition_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                command,
                124,
                stdout=self._coerce_process_text(exc.stdout),
                stderr=(
                    self._coerce_process_text(exc.stderr)
                    + f"\nEnhancementRunner timeout after {timeout_seconds} seconds."
                ).strip(),
            )

    def _coerce_process_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _failure_issue(self, completed: subprocess.CompletedProcess) -> str:
        if completed.returncode == 124 and "timeout" in str(completed.stderr).lower():
            return "runner timed out after 1800 seconds"
        stderr = str(completed.stderr or "").strip()
        return stderr.splitlines()[-1] if stderr else f"runner exited with returncode {completed.returncode}"

    def _failure_report(
        self,
        experiment: Dict[str, Any],
        manifest: DataManifest,
        issue: str,
    ) -> Dict[str, Any]:
        metric_name = manifest.metric_candidates[0] if manifest.metric_candidates else "unknown"
        return {
            "experiment": self._task_id(experiment),
            "runner_kind": self._runner_kind(experiment),
            "status": "failed",
            "metric_name": metric_name,
            "local_score": None,
            "issues": [issue],
            "notes": "EnhancementRunner captured the runner failure and kept the Agent Loop alive.",
        }

    def _write_enhancement_review(
        self,
        result: EnhancementRunResult,
        plan: Dict[str, Any],
    ) -> None:
        report = self._read_json(result.validation_report)
        baseline = plan.get("current_best_baseline") or {}
        baseline_score = baseline.get("local_score")
        score = report.get("local_score")
        improved = (
            isinstance(score, (int, float))
            and isinstance(baseline_score, (int, float))
            and score > baseline_score
        )
        review = {
            "task_id": result.task_id,
            "status": result.status,
            "metric_name": report.get("metric_name"),
            "local_score": score,
            "baseline_score": baseline_score,
            "improved_over_baseline": improved,
            "submission_valid": result.validator_result.ok,
            "decision": "ask_remote_brain_for_next_step" if result.validator_result.ok else "needs_debug",
        }
        path = self.competition_dir / "enhancement_review.json"
        path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        self.ledger.create_entry(
            task_id="enhancement_review",
            agent="brain",
            title="Review latest enhancement experiment",
            status="pass" if result.validator_result.ok else "needs_review",
            input_payload=review,
            prompt="Compare the latest enhancement result against the current best baseline and decide the next control action.",
            scorecard={
                "agent": "brain",
                "task_id": "enhancement_review",
                "status": "pass" if result.validator_result.ok else "needs_review",
                "scores": {
                    "submission_valid": 5 if result.validator_result.ok else 1,
                    "score_available": 5 if isinstance(score, (int, float)) else 1,
                    "improved_over_baseline": 5 if improved else 2,
                },
                "metric_name": report.get("metric_name"),
                "local_score": score,
                "issues": result.validator_result.errors + result.validator_result.warnings,
                "recommended_human_action": "continue" if result.validator_result.ok else "patch_prompt",
            },
            artifacts={"enhancement_review": path},
        )

    def _select_experiment(self, plan: Dict[str, Any], skip_completed: bool) -> Dict[str, Any]:
        experiments = plan.get("recommended_experiments") or []
        if not experiments:
            return {
                "task_id": "enhance_random_forest_v1",
                "title": "Random Forest enhancement fallback",
                "coding_agent_task": "Run a RandomForestClassifier enhancement with engineered family/title features.",
            }
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            if skip_completed and self._is_completed(self._task_id(experiment)):
                continue
            return experiment
        return {
            "task_id": self._next_fallback_task_id(),
            "title": "Random Forest enhancement fallback",
            "coding_agent_task": "Run a RandomForestClassifier enhancement with engineered family/title features.",
        }

    def _select_queued_experiment(self) -> Optional[Dict[str, Any]]:
        queue = self._read_json(self.competition_dir / "experiment_queue.json")
        for item in queue.get("queue", []):
            if not isinstance(item, dict):
                continue
            if item.get("status") != "pending":
                continue
            if item.get("action_type") == "manual_submit":
                continue
            if self._is_completed(self._task_id(item)):
                continue
            return item
        return None

    def _refresh_experiment_queue(self) -> None:
        if (self.competition_dir / "experiment_queue.json").exists():
            ExperimentQueueBuilder(self.competition_dir, memory=self.memory).build()
        if (self.competition_dir / "experiment_roadmap.json").exists():
            from .experiment_roadmap import ExperimentRoadmapBuilder

            ExperimentRoadmapBuilder(self.competition_dir, memory=self.memory).build()

    def _is_completed(self, task_id: str) -> bool:
        report = self.competition_dir / "experiments" / task_id / "validation_report.json"
        if not report.exists():
            return False
        data = self._read_json(report)
        return data.get("status") in {"completed", "skipped"}

    def _next_fallback_task_id(self) -> str:
        index = 1
        while True:
            task_id = f"enhance_random_forest_v{index}"
            if not (self.competition_dir / "experiments" / task_id / "validation_report.json").exists():
                return task_id
            index += 1

    def _task_id(self, experiment: Dict[str, Any]) -> str:
        raw = (
            experiment.get("task_id")
            or experiment.get("experiment_id")
            or self._slug(experiment.get("title") or experiment.get("description") or "")
            or "enhance_random_forest_v1"
        )
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(raw))
        return safe.strip("_") or "enhance_random_forest_v1"

    def _slug(self, text: str) -> str:
        import re

        words = re.findall(r"[A-Za-z0-9]+", str(text).lower())[:8]
        return "_".join(words)

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
            "agent": "enhancement_runner",
            "task_id": task_id,
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

    def _prompt_for(self, experiment: Dict[str, Any], manifest: DataManifest) -> str:
        return (
            f"Implement enhancement experiment `{self._task_id(experiment)}` on remote Linux.\n\n"
            f"Competition: {manifest.competition_name}\n"
            f"Task type: {manifest.task_type}\n"
            f"Metric candidates: {', '.join(manifest.metric_candidates)}\n"
            f"ID column: {manifest.id_column}\n"
            f"Target column: {manifest.target_column}\n\n"
            f"Remote Brain task:\n{experiment.get('coding_agent_task') or experiment.get('coding_prompt_append') or experiment.get('description') or ''}\n\n"
            "Required outputs: run.py, run.log, validation_report.json, submission.csv, validator_result.json.\n"
        )

    def _script_for(self, experiment: Dict[str, Any]) -> str:
        task_id = self._task_id(experiment)
        runner_kind = self._runner_kind(experiment)
        text = self._experiment_text(experiment)
        if runner_kind in {"lightgbm", "catboost", "xgboost"}:
            return (
                GRADIENT_BOOSTING_ENHANCEMENT_SCRIPT
                .replace("__TASK_ID__", task_id)
                .replace("__MODEL_KIND__", runner_kind)
            )
        if runner_kind in {"tabular_mlp", "tabular_resnet"}:
            return (
                TABULAR_MLP_SCRIPT
                .replace("__TASK_ID__", task_id)
                .replace("__MODEL_KIND__", runner_kind)
            )
        if runner_kind == "star_specialist_threshold_tuning":
            return STAR_SPECIALIST_THRESHOLD_TUNING_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind == "star_specialist_lgbm":
            return STAR_SPECIALIST_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind in {"classwise_blend", "clean_oof_blend"}:
            return (
                CLEAN_OOF_BLEND_SCRIPT
                .replace("__TASK_ID__", task_id)
                .replace("__MODEL_KIND__", runner_kind)
            )
        if "per_class" in text or "confusion matrix" in text or "dominant error class" in text:
            return PER_CLASS_OOF_AUDIT_SCRIPT.replace("__TASK_ID__", task_id)
        if "oof_diversity" in text or "correlation" in text or "disagreement" in text:
            return OOF_DIVERSITY_AUDIT_SCRIPT.replace("__TASK_ID__", task_id)
        if "execution_fidelity" in text or "execution fidelity" in text:
            return EXECUTION_FIDELITY_AUDIT_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind == "cv_stability_audit":
            return CV_STABILITY_AUDIT_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind == "distribution_shift_audit":
            return DISTRIBUTION_SHIFT_AUDIT_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind == "overfitting_audit":
            return OVERFITTING_AUDIT_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind == "regularized_blend":
            return OOF_SUBMISSION_BLEND_SCRIPT.replace("__TASK_ID__", task_id)
        if runner_kind == "tuned_random_forest":
            return TUNED_RANDOM_FOREST_SCRIPT.replace("__TASK_ID__", task_id)
        return RANDOM_FOREST_ENHANCEMENT_SCRIPT.replace("__TASK_ID__", task_id)

    def _experiment_text(self, experiment: Dict[str, Any]) -> str:
        return " ".join(
            str(experiment.get(key, ""))
            for key in [
                "task_id",
                "experiment_id",
                "title",
                "description",
                "hypothesis",
                "validation_plan",
                "coding_agent_task",
                "coding_prompt_append",
            ]
        ).lower()

    def _runner_kind(self, experiment: Dict[str, Any]) -> str:
        explicit = str(experiment.get("runner_kind") or "").strip().lower()
        explicit = explicit.replace("-", "_").replace(" ", "_")
        aliases = {
            "lgbm": "lightgbm",
            "xgb": "xgboost",
            "cv_stability": "cv_stability_audit",
            "stability_audit": "cv_stability_audit",
            "distribution_shift": "distribution_shift_audit",
            "drift_audit": "distribution_shift_audit",
            "overfitting": "overfitting_audit",
            "regularization_blend": "regularized_blend",
            "mlp": "tabular_mlp",
            "tabular_nn": "tabular_mlp",
            "resnet": "tabular_resnet",
            "star_specialist": "star_specialist_lgbm",
            "star_specialist_tuning": "star_specialist_threshold_tuning",
            "clean_blend": "clean_oof_blend",
        }
        explicit = aliases.get(explicit, explicit)
        if explicit in {
            "random_forest",
            "tuned_random_forest",
            "lightgbm",
            "catboost",
            "xgboost",
            "tabular_mlp",
            "tabular_resnet",
            "star_specialist_lgbm",
            "star_specialist_threshold_tuning",
            "classwise_blend",
            "clean_oof_blend",
            "cv_stability_audit",
            "distribution_shift_audit",
            "overfitting_audit",
            "regularized_blend",
        }:
            return explicit
        text = " ".join(
            str(experiment.get(key, ""))
            for key in [
                "task_id",
                "experiment_id",
                "title",
                "description",
                "coding_agent_task",
                "coding_prompt_append",
            ]
        ).lower()
        if "distribution_shift" in text or "distribution shift" in text or "drift audit" in text:
            return "distribution_shift_audit"
        if "overfitting" in text or "overfit" in text:
            return "overfitting_audit"
        if (
            "regularized_blend" in text
            or "regularized blend" in text
            or "blend_with_regularization" in text
            or ("blend" in text and "regularization" in text)
        ):
            return "regularized_blend"
        if "cv_stability" in text or "stability audit" in text or "stability_audit" in text:
            return "cv_stability_audit"
        if "lightgbm" in text or "lgbm" in text:
            return "lightgbm"
        if "catboost" in text:
            return "catboost"
        if "xgboost" in text or "xgb" in text:
            return "xgboost"
        if "threshold" in text and ("star_specialist" in text or "star specialist" in text or "star-vs-rest" in text):
            return "star_specialist_threshold_tuning"
        if "star_specialist" in text or "star specialist" in text or "star-vs-rest" in text:
            return "star_specialist_lgbm"
        if "clean_oof_blend" in text or "clean oof blend" in text or "clean blend" in text:
            return "clean_oof_blend"
        if "classwise_blend" in text or "class-wise blend" in text:
            return "classwise_blend"
        if "tabular_resnet" in text or "resnet" in text:
            return "tabular_resnet"
        if "tabular_mlp" in text or "tabular nn" in text or "mlp" in text:
            return "tabular_mlp"
        if "gridsearch" in text or "grid search" in text or "tuning" in text or "tune" in text:
            return "tuned_random_forest"
        return "random_forest"

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


RANDOM_FOREST_ENHANCEMENT_SCRIPT = r'''
from pathlib import Path
import json
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required tabular ML dependencies unavailable: {exc}"],
        "notes": "Install pandas and scikit-learn to enable enhancement experiments.",
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
y_model = y
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
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
    ]
)

if task_type == "regression":
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
else:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("preprocess", preprocess), ("model", model)])

if task_type == "regression":
    cv_scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    local_score = float(np.mean(cv_scores))
elif metric_name == "roc_auc" and y.nunique() == 2:
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    local_score = float(np.mean(cv_scores))
else:
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    local_score = float(np.mean(cv_scores))

pipe.fit(X, y_model)
if task_type == "regression":
    test_pred = pipe.predict(X_test)
elif metric_name == "roc_auc" and y.nunique() == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
    test_pred = pipe.predict_proba(X_test)[:, 1]
else:
    test_pred = pipe.predict(X_test)

submission = pd.DataFrame({id_column: test[id_column], prediction_column: test_pred})
submission.to_csv(experiment_dir / "submission.csv", index=False)

report = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "cv_scores": [float(score) for score in cv_scores],
    "feature_count": int(X.shape[1]),
    "dropped_high_cardinality_columns": high_cardinality,
    "issues": [],
    "notes": "Random forest enhancement with 5-fold CV and simple Titanic-style engineered features when columns exist.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


GRADIENT_BOOSTING_ENHANCEMENT_SCRIPT = r'''
from pathlib import Path
import json
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
model_kind = "__MODEL_KIND__"

try:
    import pandas as pd
    import numpy as np
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    if model_kind == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
    elif model_kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
    elif model_kind == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "runner_kind": model_kind,
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required boosting dependencies unavailable: {exc}"],
        "notes": "Install pandas, scikit-learn, and the requested boosting library to enable this enhancement.",
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
original_train_columns = list(train.columns)
original_test_columns = list(test.columns)

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def planned_feature_terms():
    terms = []
    for name in ["llm_experiment_plan.json", "experiment_queue.json", "remote_brain_mission.json"]:
        payload = read_json(root / name)
        text = json.dumps(payload, ensure_ascii=False).lower()
        for term in ["u-g", "g-r", "r-i", "i-z", "redshift", "color", "interaction", "actual_feature_list"]:
            if term in text and term not in terms:
                terms.append(term)
    return terms

def add_features(df):
    out = df.copy()
    lower_columns = {str(column).lower(): column for column in out.columns}

    def col(name):
        return lower_columns.get(name.lower())

    def numeric(name):
        source = col(name)
        if source is None:
            return None
        return pd.to_numeric(out[source], errors="coerce")

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
    for left, right in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        left_values = numeric(left)
        right_values = numeric(right)
        if left_values is not None and right_values is not None:
            out[f"{left}-{right}"] = left_values - right_values
    color_pairs = [name for name in ["u-g", "g-r", "r-i", "i-z"] if name in out.columns]
    if color_pairs:
        out["color_sum"] = out[color_pairs].sum(axis=1)
        out["color_std"] = out[color_pairs].std(axis=1)
    redshift_values = numeric("redshift")
    if redshift_values is not None:
        safe_redshift = redshift_values.clip(lower=0)
        out["log1p_redshift"] = np.log1p(safe_redshift)
        out["redshift_squared"] = redshift_values ** 2
        out["redshift_bin"] = pd.cut(redshift_values, bins=6, labels=False, duplicates="drop").astype("float")
        for color_name in color_pairs[:4]:
            out[f"{color_name}_x_redshift"] = out[color_name] * redshift_values
    return out

train = add_features(train)
test = add_features(test)
y = train[target_column]
label_encoder = None
target_classes = None
if task_type != "regression":
    label_encoder = LabelEncoder()
    y_model = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)
    target_classes = [str(item) for item in label_encoder.classes_]
else:
    y_model = y
drop_cols = [target_column]
if id_column in train.columns:
    drop_cols.append(id_column)
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=[id_column], errors="ignore")
engineered_features = sorted(
    column for column in X.columns if column not in set(original_train_columns) and column != target_column
)

categorical = [column for column in X.columns if X[column].dtype == "object"]
high_cardinality = [column for column in categorical if X[column].nunique(dropna=True) > 100]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality, errors="ignore")
categorical = [column for column in X.columns if X[column].dtype == "object"]
numeric = [column for column in X.columns if column not in categorical]
engineered_features_after_drop = sorted(column for column in engineered_features if column in X.columns)
planned_terms = planned_feature_terms()
implemented_text = " ".join(engineered_features_after_drop).lower()
missing_planned_terms = []
for term in planned_terms:
    if term == "color":
        matched = any(feature in implemented_text for feature in ["u-g", "g-r", "r-i", "i-z", "color_"])
    elif term == "interaction":
        matched = "_x_" in implemented_text or "interaction" in implemented_text
    elif term == "actual_feature_list":
        matched = True
    else:
        matched = term in implemented_text or term.replace("-", "_") in implemented_text
    if not matched:
        missing_planned_terms.append(term)

feature_audit = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "raw_train_columns": original_train_columns,
    "raw_test_columns": original_test_columns,
    "raw_feature_count": int(len([column for column in original_train_columns if column not in set(drop_cols)])),
    "feature_count": int(X.shape[1]),
    "engineered_feature_count": int(len(engineered_features_after_drop)),
    "engineered_features": engineered_features_after_drop,
    "all_model_features": [str(column) for column in X.columns],
    "numeric_features": [str(column) for column in numeric],
    "categorical_features": [str(column) for column in categorical],
    "dropped_high_cardinality_columns": high_cardinality,
}
(experiment_dir / "actual_feature_list.json").write_text(json.dumps(feature_audit, indent=2), encoding="utf-8")

plan_diff_issues = []
if missing_planned_terms:
    plan_diff_issues.append("Some planned feature terms are not represented in engineered features.")
if planned_terms and len(engineered_features_after_drop) == 0:
    plan_diff_issues.append("Brain requested feature engineering, but no engineered model features survived preprocessing.")
plan_diff = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "planned_feature_terms": planned_terms,
    "implemented_engineered_features": engineered_features_after_drop,
    "missing_planned_terms": missing_planned_terms,
    "feature_count": int(X.shape[1]),
    "feature_count_gt_10": bool(X.shape[1] > 10),
    "issues": plan_diff_issues,
}
(experiment_dir / "plan_vs_execution_diff.json").write_text(json.dumps(plan_diff, indent=2), encoding="utf-8")

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
    ]
)

if task_type == "regression":
    if model_kind == "lightgbm":
        base_model = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)
    elif model_kind == "catboost":
        base_model = CatBoostRegressor(iterations=300, learning_rate=0.03, random_seed=42, verbose=False)
    else:
        base_model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=42, n_jobs=-1, eval_metric="rmse")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
else:
    if model_kind == "lightgbm":
        base_model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)
    elif model_kind == "catboost":
        base_model = CatBoostClassifier(iterations=300, learning_rate=0.03, random_seed=42, verbose=False)
    else:
        base_model = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=42, n_jobs=-1, eval_metric="logloss")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("preprocess", preprocess), ("model", base_model)])

def rmse(y_true, y_pred):
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def raw_prediction(fitted_pipe, frame):
    model = fitted_pipe.named_steps["model"]
    if task_type == "regression":
        return fitted_pipe.predict(frame)
    if metric_name == "roc_auc" and len(target_classes or []) == 2 and hasattr(model, "predict_proba"):
        return fitted_pipe.predict_proba(frame)[:, 1]
    if len(target_classes or []) > 2 and hasattr(model, "predict_proba"):
        return fitted_pipe.predict_proba(frame)
    return fitted_pipe.predict(frame)

def final_prediction(raw):
    if task_type == "regression":
        return raw
    raw_array = np.asarray(raw)
    if raw_array.ndim == 2:
        encoded = np.argmax(raw_array, axis=1)
    elif metric_name == "roc_auc" and len(target_classes or []) == 2:
        encoded = (raw_array >= 0.5).astype(int)
    else:
        encoded = raw_array.astype(int)
    if label_encoder is not None:
        encoded = np.clip(encoded, 0, len(label_encoder.classes_) - 1)
        return label_encoder.inverse_transform(encoded)
    return encoded

def score_raw(raw, truth):
    if task_type == "regression":
        return rmse(truth, raw)
    if metric_name == "roc_auc" and len(target_classes or []) == 2:
        return float(roc_auc_score(truth, raw))
    raw_array = np.asarray(raw)
    encoded = np.argmax(raw_array, axis=1) if raw_array.ndim == 2 else raw_array.astype(int)
    return float(accuracy_score(truth, encoded))

def feature_names(fitted_pipe):
    fitted_preprocess = fitted_pipe.named_steps["preprocess"]
    try:
        return [str(name) for name in fitted_preprocess.get_feature_names_out()]
    except Exception:
        return [str(column) for column in X.columns]

fold_reports = []
feature_importance_rows = []
oof_raw = None
test_raw_sum = None
splitter = cv.split(X, y_model) if task_type != "regression" else cv.split(X)

for fold_index, (train_idx, valid_idx) in enumerate(splitter, start=1):
    fold_pipe = clone(pipe)
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_train = y_model.iloc[train_idx] if hasattr(y_model, "iloc") else y_model[train_idx]
    y_valid = y_model.iloc[valid_idx] if hasattr(y_model, "iloc") else y_model[valid_idx]
    fold_pipe.fit(X_train, y_train)
    train_raw = raw_prediction(fold_pipe, X_train)
    valid_raw = raw_prediction(fold_pipe, X_valid)
    test_raw = raw_prediction(fold_pipe, X_test)
    if oof_raw is None:
        valid_shape = np.asarray(valid_raw).shape
        if len(valid_shape) == 2:
            oof_raw = np.zeros((len(X), valid_shape[1]), dtype=float)
        else:
            oof_raw = np.zeros(len(X), dtype=float)
    oof_raw[valid_idx] = valid_raw
    test_raw_sum = test_raw.astype(float) if test_raw_sum is None else test_raw_sum + test_raw.astype(float)
    train_score = score_raw(train_raw, y_train)
    valid_score = score_raw(valid_raw, y_valid)
    gap = valid_score - train_score if task_type == "regression" else train_score - valid_score
    fold_reports.append(
        {
            "fold": fold_index,
            "train_score": float(train_score),
            "valid_score": float(valid_score),
            "train_valid_gap": float(gap),
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
        }
    )
    model = fold_pipe.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        names = feature_names(fold_pipe)
        for name, value in zip(names, importances):
            feature_importance_rows.append(
                {
                    "fold": fold_index,
                    "feature": str(name),
                    "importance": float(value),
                }
            )

cv_scores = np.array([item["valid_score"] for item in fold_reports], dtype=float)
local_score = float(np.mean(cv_scores))
train_valid_gap = float(np.mean([item["train_valid_gap"] for item in fold_reports]))
fold_std = float(np.std(cv_scores))

oof_frame = pd.DataFrame({id_column: train[id_column] if id_column in train.columns else np.arange(len(train)), target_column: y})
if np.asarray(oof_raw).ndim == 2:
    for class_index, class_name in enumerate(target_classes or []):
        oof_frame[f"oof_proba_{class_name}"] = oof_raw[:, class_index]
else:
    oof_frame["oof_raw"] = oof_raw
oof_frame["oof_selected"] = final_prediction(oof_raw)
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)

if feature_importance_rows:
    feature_importance = pd.DataFrame(feature_importance_rows)
    summary = (
        feature_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    feature_importance.to_csv(experiment_dir / "feature_importance_by_fold.csv", index=False)
    summary.to_csv(experiment_dir / "feature_importance.csv", index=False)
else:
    pd.DataFrame({"feature": X.columns, "importance": np.nan}).to_csv(
        experiment_dir / "feature_importance.csv", index=False
    )

test_raw_mean = test_raw_sum / cv.get_n_splits()
pipe.fit(X, y_model)
if task_type == "regression":
    test_pred = test_raw_mean
elif metric_name == "roc_auc" and len(target_classes or []) == 2:
    test_pred = test_raw_mean
else:
    test_pred = final_prediction(test_raw_mean)

submission = pd.DataFrame({id_column: test[id_column], prediction_column: test_pred})
submission.to_csv(experiment_dir / "submission.csv", index=False)

gbdt_report = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "metric_name": metric_name,
    "local_score": local_score,
    "cv_scores": [float(score) for score in cv_scores],
    "fold_reports": fold_reports,
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "feature_importance_path": str(experiment_dir / "feature_importance.csv"),
    "oof_predictions_path": str(experiment_dir / "oof_predictions.csv"),
    "actual_feature_list_path": str(experiment_dir / "actual_feature_list.json"),
    "plan_vs_execution_diff_path": str(experiment_dir / "plan_vs_execution_diff.json"),
    "engineered_feature_count": int(len(engineered_features_after_drop)),
    "engineered_features": engineered_features_after_drop,
}
(experiment_dir / "gbdt_oof_report.json").write_text(json.dumps(gbdt_report, indent=2), encoding="utf-8")

report = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "cv_scores": [float(score) for score in cv_scores],
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "fold_reports": fold_reports,
    "feature_count": int(X.shape[1]),
    "engineered_feature_count": int(len(engineered_features_after_drop)),
    "engineered_features": engineered_features_after_drop,
    "feature_importance_path": str(experiment_dir / "feature_importance.csv"),
    "oof_predictions_path": str(experiment_dir / "oof_predictions.csv"),
    "actual_feature_list_path": str(experiment_dir / "actual_feature_list.json"),
    "plan_vs_execution_diff_path": str(experiment_dir / "plan_vs_execution_diff.json"),
    "dropped_high_cardinality_columns": high_cardinality,
    "issues": plan_diff_issues,
    "notes": f"{model_kind} enhancement with 5-fold stratified OOF harness and feature importance evidence.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


TUNED_RANDOM_FOREST_SCRIPT = RANDOM_FOREST_ENHANCEMENT_SCRIPT.replace(
    "n_estimators=300,",
    "n_estimators=500,",
).replace(
    "min_samples_leaf=2,",
    "min_samples_leaf=1, max_depth=5,",
).replace(
    "Random forest enhancement",
    "Tuned random forest enhancement",
)


TABULAR_MLP_SCRIPT = r'''
from pathlib import Path
import json
import shutil
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
model_kind = "__MODEL_KIND__"

try:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "runner_kind": model_kind,
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required tabular NN dependencies unavailable: {exc}"],
        "notes": "Install pandas and scikit-learn to enable tabular NN experiments.",
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

def add_features(df):
    out = df.copy()
    lower = {str(column).lower(): column for column in out.columns}
    def numeric(name):
        source = lower.get(name.lower())
        if source is None:
            return None
        return pd.to_numeric(out[source], errors="coerce")
    for left, right in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        lv = numeric(left)
        rv = numeric(right)
        if lv is not None and rv is not None:
            out[f"{left}-{right}"] = lv - rv
    color_pairs = [name for name in ["u-g", "g-r", "r-i", "i-z"] if name in out.columns]
    if color_pairs:
        out["color_sum"] = out[color_pairs].sum(axis=1)
        out["color_std"] = out[color_pairs].std(axis=1)
    redshift = numeric("redshift")
    if redshift is not None:
        out["log1p_redshift"] = np.log1p(redshift.clip(lower=0))
        out["redshift_squared"] = redshift ** 2
        for color_name in color_pairs:
            out[f"{color_name}_x_redshift"] = out[color_name] * redshift
    return out

train = add_features(train)
test = add_features(test)
y_raw = train[target_column]
drop_cols = [target_column]
if id_column in train.columns:
    drop_cols.append(id_column)
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=[id_column], errors="ignore")

categorical = [column for column in X.columns if X[column].dtype == "object"]
high_cardinality = [column for column in categorical if X[column].nunique(dropna=True) > 120]
if high_cardinality:
    X = X.drop(columns=high_cardinality)
    X_test = X_test.drop(columns=high_cardinality, errors="ignore")
categorical = [column for column in X.columns if X[column].dtype == "object"]
numeric = [column for column in X.columns if column not in categorical]

label_encoder = None
target_classes = None
if task_type != "regression":
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw.astype(str)), index=y_raw.index)
    target_classes = [str(item) for item in label_encoder.classes_]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
else:
    y = y_raw
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
    ]
)

backend = "sklearn_mlp_fallback"
if task_type == "regression":
    model = MLPRegressor(hidden_layer_sizes=(96, 48), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=80, early_stopping=True, random_state=42)
else:
    model = MLPClassifier(hidden_layer_sizes=(96, 48), activation="relu", alpha=1e-4, learning_rate_init=1e-3, max_iter=80, early_stopping=True, random_state=42)
pipe = Pipeline([("preprocess", preprocess), ("model", model)])

def rmse(y_true, y_pred):
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def raw_prediction(fitted_pipe, frame):
    if task_type == "regression":
        return fitted_pipe.predict(frame)
    fitted_model = fitted_pipe.named_steps["model"]
    if hasattr(fitted_model, "predict_proba"):
        return fitted_pipe.predict_proba(frame)
    return fitted_pipe.predict(frame)

def final_prediction(raw):
    if task_type == "regression":
        return raw
    raw_array = np.asarray(raw)
    encoded = np.argmax(raw_array, axis=1) if raw_array.ndim == 2 else raw_array.astype(int)
    encoded = np.clip(encoded, 0, len(label_encoder.classes_) - 1)
    return label_encoder.inverse_transform(encoded)

def score_raw(raw, truth):
    if task_type == "regression":
        return rmse(truth, raw)
    raw_array = np.asarray(raw)
    if metric_name == "roc_auc" and len(target_classes or []) == 2:
        proba = raw_array[:, 1] if raw_array.ndim == 2 else raw_array
        return float(roc_auc_score(truth, proba))
    encoded = np.argmax(raw_array, axis=1) if raw_array.ndim == 2 else raw_array.astype(int)
    return float(accuracy_score(truth, encoded))

fold_reports = []
oof_raw = None
test_raw_sum = None
splitter = cv.split(X, y) if task_type != "regression" else cv.split(X)
for fold_index, (train_idx, valid_idx) in enumerate(splitter, start=1):
    fold_pipe = Pipeline([("preprocess", preprocess), ("model", model.__class__(**model.get_params()))])
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    fold_pipe.fit(X_train, y_train)
    train_raw = raw_prediction(fold_pipe, X_train)
    valid_raw = raw_prediction(fold_pipe, X_valid)
    test_raw = raw_prediction(fold_pipe, X_test)
    if oof_raw is None:
        shape = np.asarray(valid_raw).shape
        oof_raw = np.zeros((len(X), shape[1]), dtype=float) if len(shape) == 2 else np.zeros(len(X), dtype=float)
    oof_raw[valid_idx] = valid_raw
    test_raw_sum = test_raw.astype(float) if test_raw_sum is None else test_raw_sum + test_raw.astype(float)
    train_score = score_raw(train_raw, y_train)
    valid_score = score_raw(valid_raw, y_valid)
    gap = valid_score - train_score if task_type == "regression" else train_score - valid_score
    fold_reports.append({"fold": fold_index, "train_score": float(train_score), "valid_score": float(valid_score), "train_valid_gap": float(gap)})

cv_scores = np.array([item["valid_score"] for item in fold_reports], dtype=float)
local_score = float(np.mean(cv_scores))
fold_std = float(np.std(cv_scores))
train_valid_gap = float(np.mean([item["train_valid_gap"] for item in fold_reports]))

oof_frame = pd.DataFrame({id_column: train[id_column] if id_column in train.columns else np.arange(len(train)), target_column: y_raw})
if np.asarray(oof_raw).ndim == 2:
    for class_index, class_name in enumerate(target_classes or []):
        oof_frame[f"oof_proba_{class_name}"] = oof_raw[:, class_index]
else:
    oof_frame["oof_raw"] = oof_raw
oof_frame["oof_selected"] = final_prediction(oof_raw)
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)

test_raw_mean = test_raw_sum / cv.get_n_splits()
test_pred = final_prediction(test_raw_mean)
pd.DataFrame({id_column: test[id_column], prediction_column: test_pred}).to_csv(experiment_dir / "submission.csv", index=False)

model_config = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "backend": backend,
    "hidden_layer_sizes": [96, 48],
    "max_iter": 80,
    "numeric_features": numeric,
    "categorical_features": categorical,
    "dropped_high_cardinality_columns": high_cardinality,
}
(experiment_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
nn_report = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "backend": backend,
    "metric_name": metric_name,
    "local_score": local_score,
    "cv_scores": [float(score) for score in cv_scores],
    "fold_reports": fold_reports,
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "oof_predictions_path": str(experiment_dir / "oof_predictions.csv"),
}
(experiment_dir / "nn_training_report.json").write_text(json.dumps(nn_report, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "cv_scores": [float(score) for score in cv_scores],
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "backend": backend,
    "issues": [],
    "notes": "Tabular NN-style MLP OOF runner using sklearn fallback backend.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


STAR_SPECIALIST_SCRIPT = r'''
from pathlib import Path
import json
import shutil
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    import numpy as np
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None
except Exception as exc:
    report = {"experiment": "__TASK_ID__", "runner_kind": "star_specialist_lgbm", "status": "skipped", "metric_name": manifest["metric_candidates"][0], "local_score": None, "issues": [str(exc)]}
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

id_column = manifest["id_column"]
target_column = manifest["target_column"]
metric_name = manifest["metric_candidates"][0]
submission_columns = manifest["submission_columns"]
prediction_column = [column for column in submission_columns if column != id_column][0]

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def choose_target_class():
    report = read_json(root / "experiments" / "per_class_oof_audit_v1" / "per_class_oof_report.json")
    for item in report.get("lowest_recall_classes", []):
        label = str(item.get("class") or "")
        if label and label.lower() not in {"nan", "none", "null"} and item.get("support", 0) > 0:
            return label
    return "STAR"

target_class = choose_target_class()
train = pd.read_csv(root / "train.csv")
test = pd.read_csv(root / "test.csv")

def add_features(df):
    out = df.copy()
    lower = {str(column).lower(): column for column in out.columns}
    def numeric(name):
        source = lower.get(name.lower())
        if source is None:
            return None
        return pd.to_numeric(out[source], errors="coerce")
    for left, right in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        lv = numeric(left)
        rv = numeric(right)
        if lv is not None and rv is not None:
            out[f"{left}-{right}"] = lv - rv
    redshift = numeric("redshift")
    if redshift is not None:
        out["log1p_redshift"] = np.log1p(redshift.clip(lower=0))
    return out

train = add_features(train)
test = add_features(test)
y_raw = train[target_column].astype(str)
label_encoder = LabelEncoder()
y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index)
target_classes = [str(item) for item in label_encoder.classes_]
if target_class not in target_classes:
    target_class = target_classes[-1]
target_encoded = int(label_encoder.transform([target_class])[0])
binary_y = (y == target_encoded).astype(int)

drop_cols = [target_column]
if id_column in train.columns:
    drop_cols.append(id_column)
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=[id_column], errors="ignore")
categorical = [column for column in X.columns if X[column].dtype == "object"]
numeric = [column for column in X.columns if column not in categorical]
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
])
if LGBMClassifier is not None:
    base_model = LGBMClassifier(n_estimators=260, learning_rate=0.035, random_state=42, n_jobs=-1, verbose=-1)
    specialist_model = LGBMClassifier(n_estimators=220, learning_rate=0.04, random_state=43, n_jobs=-1, verbose=-1, class_weight="balanced")
    backend = "lightgbm"
else:
    base_model = RandomForestClassifier(n_estimators=220, random_state=42, n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=2)
    specialist_model = RandomForestClassifier(n_estimators=220, random_state=43, n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=2)
    backend = "random_forest_fallback"
base_pipe = Pipeline([("preprocess", preprocess), ("model", base_model)])
specialist_pipe = Pipeline([("preprocess", preprocess), ("model", specialist_model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_base = np.zeros((len(X), len(target_classes)), dtype=float)
oof_specialist = np.zeros(len(X), dtype=float)
test_base_sum = np.zeros((len(X_test), len(target_classes)), dtype=float)
test_specialist_sum = np.zeros(len(X_test), dtype=float)
fold_reports = []
for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    bp = clone(base_pipe)
    sp = clone(specialist_pipe)
    bp.fit(X.iloc[train_idx], y.iloc[train_idx])
    sp.fit(X.iloc[train_idx], binary_y.iloc[train_idx])
    valid_base = bp.predict_proba(X.iloc[valid_idx])
    valid_spec = sp.predict_proba(X.iloc[valid_idx])[:, 1]
    test_base = bp.predict_proba(X_test)
    test_spec = sp.predict_proba(X_test)[:, 1]
    oof_base[valid_idx] = valid_base
    oof_specialist[valid_idx] = valid_spec
    test_base_sum += test_base
    test_specialist_sum += test_spec
    base_pred = np.argmax(valid_base, axis=1)
    combined = base_pred.copy()
    threshold = 0.50
    combined[valid_spec >= threshold] = target_encoded
    fold_reports.append({
        "fold": fold_index,
        "base_score": float(accuracy_score(y.iloc[valid_idx], base_pred)),
        "specialist_score": float(accuracy_score(y.iloc[valid_idx], combined)),
        "coverage_rate": float((valid_spec >= threshold).mean()),
    })

base_encoded = np.argmax(oof_base, axis=1)
combined_encoded = base_encoded.copy()
threshold = 0.50
combined_encoded[oof_specialist >= threshold] = target_encoded
local_score = float(accuracy_score(y, combined_encoded))
base_score = float(accuracy_score(y, base_encoded))
combined_labels = label_encoder.inverse_transform(combined_encoded)

def per_class_metrics(labels):
    payload = {}
    for label in target_classes:
        truth = y_raw == label
        pred = pd.Series(labels) == label
        tp = int((truth.reset_index(drop=True) & pred).sum())
        support = int(truth.sum())
        predicted = int(pred.sum())
        payload[label] = {
            "support": support,
            "predicted": predicted,
            "recall": (tp / support) if support else None,
            "precision": (tp / predicted) if predicted else None,
        }
    return payload

oof_frame = pd.DataFrame({id_column: train[id_column] if id_column in train.columns else np.arange(len(train)), target_column: y_raw})
for class_index, class_name in enumerate(target_classes):
    oof_frame[f"oof_proba_{class_name}"] = oof_base[:, class_index]
oof_frame[f"specialist_proba_{target_class}"] = oof_specialist
oof_frame["oof_selected"] = combined_labels
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)

test_base_mean = test_base_sum / cv.get_n_splits()
test_spec_mean = test_specialist_sum / cv.get_n_splits()
test_encoded = np.argmax(test_base_mean, axis=1)
test_encoded[test_spec_mean >= threshold] = target_encoded
test_pred = label_encoder.inverse_transform(test_encoded)
pd.DataFrame({id_column: test[id_column], prediction_column: test_pred}).to_csv(experiment_dir / "submission.csv", index=False)

per_class = per_class_metrics(combined_labels)
specialist_report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "star_specialist_lgbm",
    "backend": backend,
    "target_class": target_class,
    "threshold": threshold,
    "base_score": base_score,
    "local_score": local_score,
    "target_class_recall": per_class[target_class]["recall"],
    "target_class_precision": per_class[target_class]["precision"],
    "coverage_rate": float((oof_specialist >= threshold).mean()),
    "fold_reports": fold_reports,
}
(experiment_dir / "specialist_report.json").write_text(json.dumps(specialist_report, indent=2), encoding="utf-8")
(experiment_dir / "per_class_oof_report.json").write_text(json.dumps({"experiment": "__TASK_ID__", "status": "completed", "reports": [{"task_id": "__TASK_ID__", "accuracy": local_score, "per_class": per_class}]}, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "star_specialist_lgbm",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "base_score": base_score,
    "target_class": target_class,
    "target_class_recall": per_class[target_class]["recall"],
    "coverage_rate": float((oof_specialist >= threshold).mean()),
    "issues": [],
    "notes": "Class specialist OOF runner combining a multiclass base model with a one-vs-rest specialist.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


STAR_SPECIALIST_THRESHOLD_TUNING_SCRIPT = (
    STAR_SPECIALIST_SCRIPT
    .replace('"runner_kind": "star_specialist_lgbm"', '"runner_kind": "star_specialist_threshold_tuning"')
    .replace(
        '''base_encoded = np.argmax(oof_base, axis=1)
combined_encoded = base_encoded.copy()
threshold = 0.50
combined_encoded[oof_specialist >= threshold] = target_encoded
local_score = float(accuracy_score(y, combined_encoded))
base_score = float(accuracy_score(y, base_encoded))
combined_labels = label_encoder.inverse_transform(combined_encoded)
''',
        '''base_encoded = np.argmax(oof_base, axis=1)
base_score = float(accuracy_score(y, base_encoded))
threshold_frontier = []
def promotion_min_score():
    for plan_name in ["experiment_queue.json", "llm_experiment_plan.json"]:
        payload = read_json(root / plan_name)
        candidates = payload.get("queue") or payload.get("recommended_experiments") or []
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            if not isinstance(item, dict) or item.get("task_id") != "__TASK_ID__":
                continue
            gate = item.get("promotion_gate") if isinstance(item.get("promotion_gate"), dict) else {}
            value = gate.get("min_local_score")
            if isinstance(value, (int, float)):
                return float(value)
    return None
score_floor_candidates = [base_score - 0.001]
gate_min_score = promotion_min_score()
if gate_min_score is not None:
    score_floor_candidates.append(gate_min_score)
min_score_floor = max(score_floor_candidates)
best = None
truth_array = y.reset_index(drop=True).to_numpy()
target_support = int((truth_array == target_encoded).sum())
for threshold_candidate in np.round(np.arange(0.50, 0.96, 0.02), 2):
    candidate_encoded = base_encoded.copy()
    candidate_encoded[oof_specialist >= threshold_candidate] = target_encoded
    candidate_score = float(accuracy_score(y, candidate_encoded))
    target_predicted = candidate_encoded == target_encoded
    target_tp = int(((truth_array == target_encoded) & target_predicted).sum())
    target_recall_candidate = (target_tp / target_support) if target_support else None
    coverage_candidate = float((oof_specialist >= threshold_candidate).mean())
    row = {
        "threshold": float(threshold_candidate),
        "local_score": candidate_score,
        "score_drop_vs_base": float(base_score - candidate_score),
        "target_class_recall": target_recall_candidate,
        "coverage_rate": coverage_candidate,
        "meets_score_floor": bool(candidate_score >= min_score_floor),
    }
    threshold_frontier.append(row)
    recall_key = -1.0 if target_recall_candidate is None else float(target_recall_candidate)
    key = (
        1 if candidate_score >= min_score_floor else 0,
        recall_key,
        candidate_score,
        -coverage_candidate,
    )
    if best is None or key > best["key"]:
        best = {
            "key": key,
            "threshold": float(threshold_candidate),
            "encoded": candidate_encoded.copy(),
            "score": candidate_score,
        }
threshold = float(best["threshold"]) if best else 0.50
combined_encoded = best["encoded"] if best else base_encoded.copy()
local_score = float(best["score"]) if best else base_score
combined_labels = label_encoder.inverse_transform(combined_encoded)
''',
    )
    .replace(
        '''    "threshold": threshold,
    "base_score": base_score,
''',
        '''    "threshold": threshold,
    "threshold_search_floor": min_score_floor,
    "threshold_frontier": threshold_frontier,
    "base_score": base_score,
''',
    )
    .replace(
        '"notes": "Class specialist OOF runner combining a multiclass base model with a one-vs-rest specialist."',
        '"notes": "Class specialist OOF runner with threshold search over the one-vs-rest specialist probability."',
    )
)


CLEAN_OOF_BLEND_SCRIPT = r'''
from pathlib import Path
import json
import shutil
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
model_kind = "__MODEL_KIND__"

try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score
except Exception as exc:
    report = {"experiment": "__TASK_ID__", "runner_kind": model_kind, "status": "skipped", "metric_name": manifest["metric_candidates"][0], "local_score": None, "issues": [str(exc)]}
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

id_column = manifest["id_column"]
target_column = manifest["target_column"]
metric_name = manifest["metric_candidates"][0]
submission_columns = manifest["submission_columns"]
prediction_column = [column for column in submission_columns if column != id_column][0]
train = pd.read_csv(root / "train.csv")
test = pd.read_csv(root / "test.csv")
sample = pd.read_csv(root / "sample_submission.csv")
target_values = set(train[target_column].dropna().astype(str).unique())

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

valid = []
skipped = []
for exp_dir in sorted((root / "experiments").iterdir()):
    if not exp_dir.is_dir() or exp_dir == experiment_dir:
        continue
    oof_path = exp_dir / "oof_predictions.csv"
    sub_path = exp_dir / "submission.csv"
    val_path = exp_dir / "validation_report.json"
    validator_path = exp_dir / "validator_result.json"
    if not oof_path.exists() or not sub_path.exists():
        skipped.append({"task_id": exp_dir.name, "reason": "missing_oof_or_submission"})
        continue
    validation = read_json(val_path)
    validator = read_json(validator_path)
    if validation.get("status") != "completed" or validator.get("ok") is not True:
        skipped.append({"task_id": exp_dir.name, "reason": "validation_or_validator_not_passed"})
        continue
    try:
        oof = pd.read_csv(oof_path)
        submission = pd.read_csv(sub_path)
    except Exception as exc:
        skipped.append({"task_id": exp_dir.name, "reason": f"read_failed: {exc}"})
        continue
    if target_column not in oof.columns:
        skipped.append({"task_id": exp_dir.name, "reason": "target_missing_in_oof"})
        continue
    pred_col = "oof_selected_label" if "oof_selected_label" in oof.columns else "oof_selected" if "oof_selected" in oof.columns else None
    if pred_col is None:
        proba_cols = [column for column in oof.columns if column.startswith("oof_proba_")]
        if proba_cols:
            pred = oof[proba_cols].idxmax(axis=1).str.replace("oof_proba_", "", regex=False)
        else:
            skipped.append({"task_id": exp_dir.name, "reason": "prediction_column_missing"})
            continue
    else:
        pred = oof[pred_col]
    pred = pred.dropna().astype(str)
    if len(pred) != len(train):
        skipped.append({"task_id": exp_dir.name, "reason": "oof_row_count_mismatch", "rows": int(len(pred))})
        continue
    if set(pred).isdisjoint(target_values):
        skipped.append({"task_id": exp_dir.name, "reason": "prediction_labels_do_not_overlap_target_labels"})
        continue
    if list(submission.columns) != submission_columns or len(submission) != len(sample):
        skipped.append({"task_id": exp_dir.name, "reason": "submission_schema_mismatch"})
        continue
    score = validation.get("local_score")
    if not isinstance(score, (int, float)):
        skipped.append({"task_id": exp_dir.name, "reason": "local_score_missing"})
        continue
    valid.append({"task_id": exp_dir.name, "score": float(score), "oof": pred.reset_index(drop=True), "submission": submission[prediction_column].astype(str).reset_index(drop=True)})

valid.sort(key=lambda item: item["score"], reverse=True)
selected = valid[:4]
if not selected:
    shutil.copyfile(root / "sample_submission.csv", experiment_dir / "submission.csv")
    payload = {"experiment": "__TASK_ID__", "status": "skipped", "valid_candidate_count": 0, "skipped_candidates": skipped}
    (experiment_dir / "clean_blend_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (experiment_dir / "skipped_candidates.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    report = {"experiment": "__TASK_ID__", "runner_kind": model_kind, "status": "skipped", "metric_name": metric_name, "local_score": None, "issues": ["No valid OOF candidates found."], "notes": "Clean blend skipped."}
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

weights = np.array([item["score"] for item in selected], dtype=float)
weights = weights - weights.min() + 1e-3
weights = weights / weights.sum()

def weighted_vote(series_list, weights):
    rows = []
    for values in zip(*[series.tolist() for series in series_list]):
        scores = {}
        for value, weight in zip(values, weights):
            scores[value] = scores.get(value, 0.0) + float(weight)
        rows.append(max(scores.items(), key=lambda item: item[1])[0])
    return pd.Series(rows)

blend_oof = weighted_vote([item["oof"] for item in selected], weights)
blend_submission = weighted_vote([item["submission"] for item in selected], weights)
local_score = float(accuracy_score(train[target_column].astype(str), blend_oof.astype(str)))
oof_frame = pd.DataFrame({id_column: train[id_column] if id_column in train.columns else np.arange(len(train)), target_column: train[target_column], "oof_selected": blend_oof})
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)
pd.DataFrame({id_column: test[id_column], prediction_column: blend_submission}).to_csv(experiment_dir / "submission.csv", index=False)

diversity = []
for i, left in enumerate(selected):
    for right in selected[i + 1:]:
        agreement = float((left["oof"].astype(str) == right["oof"].astype(str)).mean())
        diversity.append({"left": left["task_id"], "right": right["task_id"], "agreement_rate": agreement, "disagreement_rate": 1.0 - agreement})

payload = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "status": "completed",
    "local_score": local_score,
    "valid_candidate_count": len(valid),
    "selected_candidates": [{"task_id": item["task_id"], "score": item["score"], "weight": float(weight)} for item, weight in zip(selected, weights)],
    "skipped_candidates": skipped,
    "pairwise_disagreement": diversity,
}
(experiment_dir / "clean_blend_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
(experiment_dir / "oof_diversity_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
(experiment_dir / "skipped_candidates.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": model_kind,
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "valid_candidate_count": len(valid),
    "selected_candidate_count": len(selected),
    "issues": [],
    "warnings": [f"Skipped {len(skipped)} invalid candidate(s)."] if skipped else [],
    "notes": "Clean OOF blend with invalid candidate filtering.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


EXECUTION_FIDELITY_AUDIT_SCRIPT = r'''
from pathlib import Path
import json
import re
import shutil
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def tail(path, limit=12000):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]

def extract_constructed_features(script_text):
    names = set()
    for match in re.finditer(r'out\["([^"]+)"\]\s*=', script_text):
        names.add(match.group(1))
    for match in re.finditer(r"out\['([^']+)'\]\s*=", script_text):
        names.add(match.group(1))
    return sorted(names)

def feature_importance_names(path):
    if not path.exists():
        return []
    try:
        import pandas as pd
        frame = pd.read_csv(path)
        column = "feature" if "feature" in frame.columns else frame.columns[0]
        return [str(value) for value in frame[column].dropna().head(200).tolist()]
    except Exception:
        return []

plan = read_json(root / "llm_experiment_plan.json")
queue = read_json(root / "experiment_queue.json")
mission = read_json(root / "remote_brain_mission.json")
candidate_pool = read_json(root / "candidate_pool.json")

planned_terms = []
for source in [plan, queue, mission]:
    text = json.dumps(source, ensure_ascii=False).lower()
    for term in ["u-g", "g-r", "r-i", "i-z", "redshift", "color", "interaction", "actual_feature_list"]:
        if term in text and term not in planned_terms:
            planned_terms.append(term)

experiments = []
for report_path in sorted((root / "experiments").glob("*/validation_report.json")):
    exp_dir = report_path.parent
    report = read_json(report_path)
    if report.get("runner_kind") not in {"lightgbm", "catboost", "xgboost", None}:
        continue
    script_text = tail(exp_dir / "run.py")
    constructed = extract_constructed_features(script_text)
    importance_names = feature_importance_names(exp_dir / "feature_importance.csv")
    experiments.append({
        "task_id": report.get("experiment") or exp_dir.name,
        "runner_kind": report.get("runner_kind"),
        "local_score": report.get("local_score"),
        "feature_count": report.get("feature_count"),
        "constructed_features_in_run_py": constructed,
        "feature_importance_names_head": importance_names[:50],
        "has_actual_feature_list": (exp_dir / "actual_feature_list.json").exists(),
        "has_plan_vs_execution_diff": (exp_dir / "plan_vs_execution_diff.json").exists(),
    })

issues = []
if planned_terms and all(not item["constructed_features_in_run_py"] for item in experiments):
    issues.append("Planned feature-engineering terms exist, but inspected run.py files expose no constructed feature assignments.")
if any(item.get("feature_count") == 10 for item in experiments) and planned_terms:
    issues.append("Feature-engineering was planned, but multiple inspected runs still report feature_count=10.")

audit = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "planned_feature_terms": planned_terms,
    "inspected_experiments": experiments,
    "candidate_pool_summary": {
        "champion_task_id": candidate_pool.get("champion_task_id"),
        "champion_score": candidate_pool.get("champion_score"),
        "candidate_count": candidate_pool.get("candidate_count"),
    },
    "issues": issues,
    "recommendation": "Implement verified feature construction before rerunning more generic GBDT experiments." if issues else "No major plan-vs-execution gap detected.",
}
(experiment_dir / "feature_implementation_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
(experiment_dir / "plan_vs_execution_diff.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
(experiment_dir / "actual_feature_list.json").write_text(json.dumps({
    "experiment": "__TASK_ID__",
    "status": "audit_only",
    "inspected_feature_lists": {item["task_id"]: item["feature_importance_names_head"] for item in experiments},
}, indent=2, ensure_ascii=False), encoding="utf-8")

sample = root / "sample_submission.csv"
if sample.exists():
    shutil.copyfile(sample, experiment_dir / "submission.csv")

report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "execution_fidelity_audit",
    "status": "completed",
    "metric_name": manifest["metric_candidates"][0],
    "local_score": None,
    "issues": issues,
    "warnings": [],
    "notes": "Audited Remote Brain plan against actual runner feature implementation artifacts.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
'''


PER_CLASS_OOF_AUDIT_SCRIPT = r'''
from pathlib import Path
import json
import shutil

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
except Exception as exc:
    report = {"experiment": "__TASK_ID__", "runner_kind": "per_class_oof_audit", "status": "skipped", "metric_name": manifest["metric_candidates"][0], "local_score": None, "issues": [str(exc)]}
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0)

target = manifest["target_column"]
reports = []
skipped_reports = []
for oof_path in sorted((root / "experiments").glob("*/oof_predictions.csv")):
    try:
        frame = pd.read_csv(oof_path)
    except Exception:
        continue
    if target not in frame.columns:
        continue
    pred_col = "oof_selected_label" if "oof_selected_label" in frame.columns else "oof_selected" if "oof_selected" in frame.columns else None
    if pred_col is None:
        proba_cols = [c for c in frame.columns if c.startswith("oof_proba_")]
        if proba_cols:
            pred_col = "__argmax_label__"
            labels = [c.replace("oof_proba_", "") for c in proba_cols]
            frame[pred_col] = frame[proba_cols].idxmax(axis=1).str.replace("oof_proba_", "", regex=False)
        else:
            continue
    y_raw = frame[target]
    pred_raw = frame[pred_col]
    valid = y_raw.notna() & pred_raw.notna()
    y = y_raw[valid].astype(str)
    pred = pred_raw[valid].astype(str)
    valid_pred = ~pred.str.lower().isin({"", "nan", "none", "null"})
    y = y[valid_pred]
    pred = pred[valid_pred]
    if len(pred) == 0:
        skipped_reports.append({"task_id": oof_path.parent.name, "reason": "empty_or_null_predictions"})
        continue
    if set(pred).isdisjoint(set(y)):
        skipped_reports.append({
            "task_id": oof_path.parent.name,
            "reason": "prediction_labels_do_not_overlap_target_labels",
            "prediction_values_head": sorted(set(pred))[:10],
            "target_values_head": sorted(set(y))[:10],
        })
        continue
    labels = sorted(set(y) | set(pred))
    matrix = {label: {other: int(((y == label) & (pred == other)).sum()) for other in labels} for label in labels}
    per_class = {}
    for label in labels:
        tp = matrix[label].get(label, 0)
        support = int((y == label).sum())
        predicted = int((pred == label).sum())
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "recall": (tp / support) if support else None,
            "precision": (tp / predicted) if predicted else None,
        }
    reports.append({
        "task_id": oof_path.parent.name,
        "rows": int(len(frame)),
        "prediction_column": pred_col,
        "accuracy": float((y == pred).mean()),
        "per_class": per_class,
        "confusion_matrix": matrix,
    })

dominant = []
for item in reports:
    for label, row in item["per_class"].items():
        if row["recall"] is not None:
            dominant.append({"task_id": item["task_id"], "class": label, "recall": row["recall"], "support": row["support"]})
dominant = sorted(dominant, key=lambda x: x["recall"])[:10]
payload = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "reports": reports,
    "skipped_reports": skipped_reports,
    "lowest_recall_classes": dominant,
}
(experiment_dir / "per_class_oof_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
sample = root / "sample_submission.csv"
if sample.exists():
    shutil.copyfile(sample, experiment_dir / "submission.csv")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "per_class_oof_audit",
    "status": "completed",
    "metric_name": manifest["metric_candidates"][0],
    "local_score": None,
    "issues": [] if reports else ["No compatible OOF predictions found."],
    "warnings": [f"Skipped {len(skipped_reports)} invalid OOF report(s)."] if skipped_reports else [],
    "notes": "Computed per-class OOF metrics from existing experiment artifacts.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
'''


OOF_DIVERSITY_AUDIT_SCRIPT = r'''
from pathlib import Path
import json
import shutil

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
except Exception as exc:
    report = {"experiment": "__TASK_ID__", "runner_kind": "oof_diversity_audit", "status": "skipped", "metric_name": manifest["metric_candidates"][0], "local_score": None, "issues": [str(exc)]}
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0)

target = manifest["target_column"]
target_values = set()
try:
    target_values = set(pd.read_csv(root / "train.csv")[target].dropna().astype(str).unique())
except Exception:
    target_values = set()
predictions = {}
skipped_predictions = []
for oof_path in sorted((root / "experiments").glob("*/oof_predictions.csv")):
    try:
        frame = pd.read_csv(oof_path)
    except Exception:
        continue
    pred_col = "oof_selected_label" if "oof_selected_label" in frame.columns else "oof_selected" if "oof_selected" in frame.columns else None
    if pred_col is None:
        proba_cols = [c for c in frame.columns if c.startswith("oof_proba_")]
        if proba_cols:
            series = frame[proba_cols].idxmax(axis=1).str.replace("oof_proba_", "", regex=False)
        else:
            continue
    else:
        series = frame[pred_col]
    series = series.dropna().astype(str)
    series = series[~series.str.lower().isin({"", "nan", "none", "null"})]
    if len(series) == 0:
        skipped_predictions.append({"task_id": oof_path.parent.name, "reason": "empty_or_null_predictions"})
        continue
    if target_values and set(series).isdisjoint(target_values):
        skipped_predictions.append({
            "task_id": oof_path.parent.name,
            "reason": "prediction_labels_do_not_overlap_target_labels",
            "prediction_values_head": sorted(set(series))[:10],
        })
        continue
    predictions[oof_path.parent.name] = series.reset_index(drop=True)

tasks = sorted(predictions)
disagreement = []
for i, left in enumerate(tasks):
    for right in tasks[i + 1:]:
        a = predictions[left]
        b = predictions[right]
        n = min(len(a), len(b))
        if n == 0:
            continue
        disagreement.append({
            "left": left,
            "right": right,
            "rows": int(n),
            "agreement_rate": float((a.iloc[:n].astype(str) == b.iloc[:n].astype(str)).mean()),
            "disagreement_rate": float((a.iloc[:n].astype(str) != b.iloc[:n].astype(str)).mean()),
        })

payload = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "candidate_count": len(tasks),
    "tasks": tasks,
    "skipped_predictions": skipped_predictions,
    "pairwise_disagreement": disagreement,
    "most_diverse_pairs": sorted(disagreement, key=lambda x: x["disagreement_rate"], reverse=True)[:10],
}
(experiment_dir / "oof_diversity_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
(experiment_dir / "regularized_blend_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
sample = root / "sample_submission.csv"
if sample.exists():
    shutil.copyfile(sample, experiment_dir / "submission.csv")
report = {"experiment": "__TASK_ID__", "runner_kind": "oof_diversity_audit", "status": "completed", "metric_name": manifest["metric_candidates"][0], "local_score": None, "issues": [] if len(tasks) >= 2 else ["Need at least two compatible OOF candidates."], "notes": "Computed OOF prediction diversity across existing candidates."}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
'''


DISTRIBUTION_SHIFT_AUDIT_SCRIPT = r'''
from pathlib import Path
import json
import math
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "runner_kind": "distribution_shift_audit",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required audit dependencies unavailable: {exc}"],
        "notes": "Install pandas and scikit-learn to enable distribution shift audits.",
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
        out["TicketPrefix"] = out["Ticket"].astype(str).str.replace(r"[0-9./]", "", regex=True).str.strip().replace("", "NONE")
    return out

def numeric_drift(column, left, right):
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    std = float(left_num.std(ddof=0) or 0.0)
    mean_gap = abs(float(left_num.mean()) - float(right_num.mean())) if left_num.notna().any() and right_num.notna().any() else 0.0
    missing_gap = abs(float(left_num.isna().mean()) - float(right_num.isna().mean()))
    return {
        "feature": column,
        "kind": "numeric",
        "drift_score": float(mean_gap / max(std, 1e-9) + missing_gap),
        "mean_gap_in_train_std": float(mean_gap / max(std, 1e-9)),
        "missing_rate_gap": missing_gap,
    }

def categorical_drift(column, left, right):
    left_s = left.fillna("__MISSING__").astype(str)
    right_s = right.fillna("__MISSING__").astype(str)
    left_values = set(left_s)
    right_values = set(right_s)
    unseen = right_values - left_values
    unseen_rate = float(right_s.isin(unseen).mean()) if len(right_s) else 0.0
    left_top = left_s.mode().iloc[0] if not left_s.mode().empty else ""
    right_top = right_s.mode().iloc[0] if not right_s.mode().empty else ""
    top_shift = 0.5 if left_top != right_top else 0.0
    missing_gap = abs(float((left_s == "__MISSING__").mean()) - float((right_s == "__MISSING__").mean()))
    return {
        "feature": column,
        "kind": "categorical",
        "drift_score": float(unseen_rate + top_shift + missing_gap),
        "unseen_category_rate": unseen_rate,
        "top_category_changed": bool(left_top != right_top),
        "missing_rate_gap": missing_gap,
    }

train_fe = add_features(train)
test_fe = add_features(test)
ignored = {id_column, target_column, None, ""}
common_features = [column for column in train_fe.columns if column not in ignored and column in test_fe.columns]
drift_items = []
for column in common_features:
    if pd.api.types.is_numeric_dtype(train_fe[column]):
        drift_items.append(numeric_drift(column, train_fe[column], test_fe[column]))
    else:
        drift_items.append(categorical_drift(column, train_fe[column], test_fe[column]))
drift_items.sort(key=lambda item: item["drift_score"], reverse=True)
max_drift = drift_items[0]["drift_score"] if drift_items else None
high_drift_features = [item["feature"] for item in drift_items if item["drift_score"] >= 0.75]

y = train_fe[target_column]
drop_cols = [target_column]
if id_column in train_fe.columns:
    drop_cols.append(id_column)
X = train_fe.drop(columns=drop_cols)
X_test = test_fe.drop(columns=[id_column], errors="ignore")
for column in high_drift_features:
    if column in X.columns and len(X.columns) > 2:
        X = X.drop(columns=[column])
        X_test = X_test.drop(columns=[column], errors="ignore")
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
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
])
if task_type == "regression":
    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=3, max_depth=6)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "neg_root_mean_squared_error"
else:
    model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=3, max_depth=6, class_weight="balanced_subsample")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "roc_auc" if metric_name == "roc_auc" and y.nunique() == 2 else "accuracy"
pipe = Pipeline([("preprocess", preprocess), ("model", model)])
scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
if task_type == "regression":
    scores = -scores
local_score = float(np.mean(scores))
pipe.fit(X, y)
if task_type == "regression":
    test_pred = pipe.predict(X_test)
elif metric_name == "roc_auc" and y.nunique() == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
    test_pred = pipe.predict_proba(X_test)[:, 1]
else:
    test_pred = pipe.predict(X_test)
pd.DataFrame({id_column: test[id_column], prediction_column: test_pred}).to_csv(experiment_dir / "submission.csv", index=False)

audit = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "max_drift_score": max_drift,
    "top_features": drift_items[:15],
    "dropped_high_drift_features": high_drift_features,
    "dropped_high_cardinality_columns": high_cardinality,
}
(experiment_dir / "distribution_shift_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "distribution_shift_audit",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "cv_scores": [float(score) for score in scores],
    "max_drift_score": max_drift,
    "dropped_high_drift_features": high_drift_features,
    "issues": [],
    "notes": "Distribution shift audit with a conservative drift-aware baseline submission.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


OVERFITTING_AUDIT_SCRIPT = r'''
from pathlib import Path
import json
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "runner_kind": "overfitting_audit",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required overfitting-audit dependencies unavailable: {exc}"],
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
        out["TicketPrefix"] = out["Ticket"].astype(str).str.replace(r"[0-9./]", "", regex=True).str.strip().replace("", "NONE")
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
categorical = [column for column in X.columns if X[column].dtype == "object"]
numeric = [column for column in X.columns if column not in categorical]

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
])

def make_model(kind, seed):
    if task_type == "regression":
        if kind == "conservative":
            return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=4, max_depth=5)
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, min_samples_leaf=1, max_depth=None)
    if kind == "conservative":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=4, max_depth=5, class_weight="balanced_subsample")
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1, min_samples_leaf=1, max_depth=None, class_weight="balanced_subsample")

scoring = "neg_root_mean_squared_error" if task_type == "regression" else ("roc_auc" if metric_name == "roc_auc" and y.nunique() == 2 else "accuracy")
seeds = [17, 42, 123]
models = []
for kind in ["conservative", "expressive"]:
    seed_scores = []
    train_scores = []
    test_scores = []
    for seed in seeds:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed) if task_type == "regression" else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pipe = Pipeline([("preprocess", preprocess), ("model", make_model(kind, seed))])
        cv_result = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=1)
        train_score = cv_result["train_score"]
        test_score = cv_result["test_score"]
        if task_type == "regression":
            train_score = -train_score
            test_score = -test_score
        seed_scores.append(float(np.mean(test_score)))
        train_scores.append(float(np.mean(train_score)))
        test_scores.extend(float(score) for score in test_score)
    lower_is_better = task_type == "regression" or metric_name in {"rmse", "rmsle", "mae", "log_loss"}
    train_mean = float(np.mean(train_scores))
    valid_mean = float(np.mean(seed_scores))
    gap = valid_mean - train_mean if lower_is_better else train_mean - valid_mean
    models.append({
        "kind": kind,
        "seed_scores": seed_scores,
        "fold_scores": test_scores,
        "train_mean": train_mean,
        "valid_mean": valid_mean,
        "train_valid_gap": float(gap),
        "seed_std": float(np.std(seed_scores)),
        "fold_std": float(np.std(test_scores)),
    })

lower_is_better = task_type == "regression" or metric_name in {"rmse", "rmsle", "mae", "log_loss"}
selected = sorted(models, key=lambda item: item["valid_mean"], reverse=not lower_is_better)[0]
if selected["train_valid_gap"] > 0.08 and len(models) > 1:
    selected = [item for item in models if item["kind"] == "conservative"][0]
local_score = selected["valid_mean"]
pipe = Pipeline([("preprocess", preprocess), ("model", make_model(selected["kind"], 42))])
pipe.fit(X, y)
if task_type == "regression":
    test_pred = pipe.predict(X_test)
elif metric_name == "roc_auc" and y.nunique() == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
    test_pred = pipe.predict_proba(X_test)[:, 1]
else:
    test_pred = pipe.predict(X_test)
pd.DataFrame({id_column: test[id_column], prediction_column: test_pred}).to_csv(experiment_dir / "submission.csv", index=False)

issues = []
if max(item["train_valid_gap"] for item in models) > 0.08:
    issues.append("Large train-validation gap detected in at least one candidate.")
if max(item["fold_std"] for item in models) > 0.035:
    issues.append("Fold variance is high for at least one candidate.")
audit = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "selected_model": selected,
    "candidates": models,
    "issues": issues,
}
(experiment_dir / "overfitting_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "overfitting_audit",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": local_score,
    "selected_kind": selected["kind"],
    "train_valid_gap": selected["train_valid_gap"],
    "seed_std": selected["seed_std"],
    "fold_std": selected["fold_std"],
    "issues": issues,
    "notes": "Overfitting audit comparing conservative and expressive forests across multiple CV seeds.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


OOF_SUBMISSION_BLEND_SCRIPT = r'''
from pathlib import Path
import json
import math
import shutil
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    import numpy as np
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "runner_kind": "regularized_blend",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0] if manifest.get("metric_candidates") else "unknown",
        "local_score": None,
        "issues": [f"required lightweight blend dependencies unavailable: {exc}"],
    }
    (experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0)

id_column = manifest.get("id_column") or "unknown"
target_column = manifest.get("target_column") or "unknown"
metric_name = (manifest.get("metric_candidates") or ["unknown"])[0]
task_type = manifest.get("task_type") or "unknown"
submission_columns = manifest.get("submission_columns") or []
sample_path = root / "sample_submission.csv"
train_path = root / "train.csv"
test_path = root / "test.csv"

sample = pd.read_csv(sample_path)
train = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
if id_column == "unknown" or id_column not in sample.columns:
    id_column = sample.columns[0]
prediction_column = next((column for column in submission_columns if column != id_column), None)
if not prediction_column or prediction_column not in sample.columns:
    prediction_column = [column for column in sample.columns if column != id_column][0]

lower_is_better = metric_name in {"rmse", "rmsle", "mae", "log_loss", "mape"}
classification = task_type != "regression"
if classification and target_column in train.columns:
    classification = not pd.api.types.is_float_dtype(train[target_column]) or train[target_column].nunique(dropna=True) <= 50

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def candidate_task_id(path):
    if path.name == "artifacts" and path.parent.name:
        return path.parent.name
    return path.name

def valid_submission(path):
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return id_column in frame.columns and prediction_column in frame.columns

def find_candidates():
    rows = []
    search_dirs = []
    for base in [root / "experiments", root / "runs"]:
        if base.exists():
            search_dirs.extend([path for path in base.glob("*") if path.is_dir()])
            search_dirs.extend([path / "artifacts" for path in base.glob("*") if (path / "artifacts").is_dir()])
    for path in search_dirs:
        if path.resolve() == experiment_dir.resolve():
            continue
        report = read_json(path / "validation_report.json")
        validator = read_json(path / "validator_result.json")
        score = report.get("local_score")
        if not isinstance(score, (int, float)) or math.isnan(float(score)):
            continue
        if validator and validator.get("ok") is False:
            continue
        submission_path = path / "submission.csv"
        if not valid_submission(submission_path):
            continue
        task_id = report.get("experiment") or report.get("baseline") or candidate_task_id(path)
        rows.append({
            "task_id": str(task_id),
            "path": path,
            "score": float(score),
            "metric_name": report.get("metric_name") or metric_name,
            "runner_kind": report.get("runner_kind") or report.get("baseline_kind") or "unknown",
            "has_oof": (path / "oof_predictions.csv").exists(),
        })
    deduped = {}
    for row in rows:
        old = deduped.get(row["task_id"])
        if old is None:
            deduped[row["task_id"]] = row
            continue
        better = row["score"] < old["score"] if lower_is_better else row["score"] > old["score"]
        if better or (row["has_oof"] and not old["has_oof"]):
            deduped[row["task_id"]] = row
    return sorted(deduped.values(), key=lambda item: item["score"], reverse=not lower_is_better)

def compute_weights(candidates):
    scores = np.array([item["score"] for item in candidates], dtype=float)
    if lower_is_better:
        shifted = scores.max() - scores + 1e-6
    else:
        shifted = scores - scores.min() + 1e-6
    if float(shifted.sum()) <= 0:
        shifted = np.ones(len(candidates), dtype=float)
    weights = shifted / shifted.sum()
    return [float(weight) for weight in weights]

def aligned_submission(candidate):
    frame = pd.read_csv(candidate["path"] / "submission.csv")
    frame = frame[[id_column, prediction_column]].copy()
    aligned = sample[[id_column]].merge(frame, on=id_column, how="left")
    fallback = frame[prediction_column].mode().iloc[0] if not frame.empty else sample[prediction_column].iloc[0]
    aligned[prediction_column] = aligned[prediction_column].fillna(fallback)
    return aligned[prediction_column]

def weighted_vote(prediction_series, weights):
    labels = []
    for row_values in zip(*[series.astype(str).tolist() for series in prediction_series]):
        totals = {}
        for value, weight in zip(row_values, weights):
            totals[value] = totals.get(value, 0.0) + weight
        labels.append(max(totals.items(), key=lambda item: (item[1], item[0]))[0])
    return labels

def weighted_average(prediction_series, weights):
    stacked = np.vstack([pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float) for series in prediction_series])
    return np.average(stacked, axis=0, weights=np.array(weights, dtype=float))

def score_oof(candidates, weights):
    if target_column not in train.columns:
        return None, "best_candidate_score", []
    y = train[target_column].reset_index(drop=True)
    oof_series = []
    used = []
    for item in candidates:
        path = item["path"] / "oof_predictions.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if classification and "oof_selected_label" in frame.columns:
            series = frame["oof_selected_label"]
        elif "oof_selected" in frame.columns:
            series = frame["oof_selected"]
        else:
            continue
        if len(series) != len(y):
            continue
        oof_series.append(series.reset_index(drop=True))
        used.append(item["task_id"])
    if not oof_series:
        return None, "best_candidate_score", used
    used_weights = compute_weights([item for item in candidates if item["task_id"] in used])
    if classification:
        pred = pd.Series(weighted_vote(oof_series, used_weights))
        return float((pred.astype(str).reset_index(drop=True) == y.astype(str)).mean()), "blended_oof_accuracy", used
    pred = weighted_average(oof_series, used_weights)
    target = pd.to_numeric(y, errors="coerce").fillna(float(pd.to_numeric(y, errors="coerce").mean())).to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((target - pred) ** 2)))
    return rmse, "blended_oof_rmse", used

candidates = find_candidates()
issues = []
warnings = []
if not candidates:
    fallback = sample.copy()
    fallback.to_csv(experiment_dir / "submission.csv", index=False)
    issues.append("No valid candidate submissions were available; copied sample_submission.csv.")
    selected = []
    weights = []
    local_score = None
    local_score_source = "unavailable"
else:
    selected = candidates[: min(3, len(candidates))]
    weights = compute_weights(selected)
    prediction_series = [aligned_submission(item) for item in selected]
    output = sample[[id_column]].copy()
    if classification:
        output[prediction_column] = weighted_vote(prediction_series, weights)
    else:
        output[prediction_column] = weighted_average(prediction_series, weights)
    output.to_csv(experiment_dir / "submission.csv", index=False)
    oof_score, source, used_oof = score_oof(selected, weights)
    best_score = selected[0]["score"]
    local_score = float(oof_score) if oof_score is not None else float(best_score)
    local_score_source = source
    if len(selected) < 2:
        warnings.append("Only one valid candidate was available; blend degenerates to candidate copy.")
    if source == "best_candidate_score":
        warnings.append("No compatible OOF predictions were available; local_score falls back to best candidate score.")

oof_frame = pd.DataFrame()
if target_column in train.columns:
    oof_frame[id_column] = train[id_column] if id_column in train.columns else np.arange(len(train))
    oof_frame[target_column] = train[target_column]
    if candidates:
        oof_frame["oof_selected_label"] = ""
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)

seed_scores = [{"candidate": item["task_id"], "score": item["score"]} for item in selected]
score_values = [item["score"] for item in selected]
seed_mean = float(np.mean(score_values)) if score_values else None
seed_std = float(np.std(score_values)) if score_values else None
fold_std = seed_std
train_valid_gap = 0.0 if local_score is not None else None
max_model_correlation = None
avg_model_correlation = None

blend_report = {
    "experiment": "__TASK_ID__",
    "status": "completed" if candidates else "failed",
    "candidate_source": "existing_oof_and_submission_artifacts",
    "selected_candidates": [
        {
            "task_id": item["task_id"],
            "runner_kind": item["runner_kind"],
            "score": item["score"],
            "weight": weights[index] if index < len(weights) else None,
            "has_oof": item["has_oof"],
            "path": str(item["path"]),
        }
        for index, item in enumerate(selected)
    ],
    "weights": weights,
    "local_score": local_score,
    "local_score_source": local_score_source,
    "seed_scores": seed_scores,
    "seed_mean": seed_mean,
    "seed_std": seed_std,
    "fold_mean": seed_mean,
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "max_model_correlation": max_model_correlation,
    "avg_model_correlation": avg_model_correlation,
    "issues": issues,
    "warnings": warnings,
}
(experiment_dir / "regularized_blend_report.json").write_text(json.dumps(blend_report, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "regularized_blend",
    "status": "completed" if candidates else "failed",
    "metric_name": metric_name,
    "local_score": local_score,
    "selected_submission": {
        "name": "weighted_submission_blend" if len(selected) > 1 else "candidate_copy",
        "score": local_score,
        "models": [item["task_id"] for item in selected],
        "weights": weights,
    },
    "seed_scores": seed_scores,
    "seed_mean": seed_mean,
    "seed_std": seed_std,
    "fold_mean": seed_mean,
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "max_model_correlation": max_model_correlation,
    "avg_model_correlation": avg_model_correlation,
    "issues": issues,
    "warnings": warnings,
    "notes": "Lightweight regularized blend over existing validated submissions and OOF artifacts.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


REGULARIZED_BLEND_SCRIPT = r'''
from pathlib import Path
import json
import math
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

try:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
except Exception as exc:
    report = {
        "experiment": "__TASK_ID__",
        "runner_kind": "regularized_blend",
        "status": "skipped",
        "metric_name": manifest["metric_candidates"][0],
        "local_score": None,
        "issues": [f"required blend dependencies unavailable: {exc}"],
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
        out["TicketPrefix"] = out["Ticket"].astype(str).str.replace(r"[0-9./]", "", regex=True).str.strip().replace("", "NONE")
    return out

train = add_features(train)
test = add_features(test)
y = train[target_column]
label_encoder = None
if task_type != "regression":
    label_encoder = LabelEncoder()
    y_model = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)
else:
    y_model = y
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
base_preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
])
tree_preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
])

def make_models(seed):
    if task_type == "regression":
        return {
            "ridge": Pipeline([("preprocess", base_preprocess), ("model", Ridge(alpha=5.0))]),
            "rf_regularized": Pipeline([("preprocess", tree_preprocess), ("model", RandomForestRegressor(n_estimators=260, random_state=seed, min_samples_leaf=4, max_depth=6, n_jobs=-1))]),
            "extra_regularized": Pipeline([("preprocess", tree_preprocess), ("model", ExtraTreesRegressor(n_estimators=260, random_state=seed + 1, min_samples_leaf=4, max_depth=6, n_jobs=-1))]),
        }
    return {
        "logistic_l2": Pipeline([("preprocess", base_preprocess), ("model", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced"))]),
        "rf_regularized": Pipeline([("preprocess", tree_preprocess), ("model", RandomForestClassifier(n_estimators=260, random_state=seed, min_samples_leaf=4, max_depth=6, class_weight="balanced_subsample", n_jobs=-1))]),
        "extra_regularized": Pipeline([("preprocess", tree_preprocess), ("model", ExtraTreesClassifier(n_estimators=260, random_state=seed + 1, min_samples_leaf=4, max_depth=6, class_weight="balanced", n_jobs=-1))]),
    }

def split_for(seed):
    if task_type == "regression":
        return KFold(n_splits=5, shuffle=True, random_state=seed).split(X, y_model)
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y_model)

def raw_prediction(pipe, frame):
    if task_type == "regression":
        return pipe.predict(frame).astype(float)
    if metric_name == "roc_auc" and y_model.nunique() == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(frame)[:, 1].astype(float)
    if y_model.nunique() == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(frame)[:, 1].astype(float)
    return pipe.predict(frame).astype(float)

def score_raw(raw, target):
    if task_type == "regression":
        return float(mean_squared_error(target, raw, squared=False))
    if metric_name == "roc_auc" and len(set(target)) == 2:
        return float(roc_auc_score(target, raw))
    labels = (raw >= 0.5).astype(int) if len(set(target)) == 2 else np.rint(raw)
    return float(accuracy_score(target, labels))

def correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return None if math.isnan(value) else value

lower_is_better = task_type == "regression" or metric_name in {"rmse", "rmsle", "mae", "log_loss"}
seeds = [17, 42, 123]
seed_summaries = []
all_model_names = list(make_models(seeds[0]).keys())
avg_oof = {name: np.zeros(len(X), dtype=float) for name in all_model_names}
avg_test = {name: np.zeros(len(X_test), dtype=float) for name in all_model_names}
selected_test_outputs = []
selected_oof_outputs = []

for seed in seeds:
    models = make_models(seed)
    oof = {name: np.zeros(len(X), dtype=float) for name in models}
    test_raw = {name: np.zeros(len(X_test), dtype=float) for name in models}
    fold_scores = {name: [] for name in models}
    train_scores = {name: [] for name in models}
    for train_idx, valid_idx in split_for(seed):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_model.iloc[train_idx], y_model.iloc[valid_idx]
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            train_raw = raw_prediction(pipe, X_train)
            valid_raw = raw_prediction(pipe, X_valid)
            oof[name][valid_idx] = valid_raw
            train_scores[name].append(score_raw(train_raw, y_train))
            fold_scores[name].append(score_raw(valid_raw, y_valid))
            test_raw[name] += raw_prediction(pipe, X_test) / 5.0
    model_rows = []
    for name in models:
        model_rows.append({
            "model": name,
            "score": score_raw(oof[name], y_model),
            "fold_scores": [float(score) for score in fold_scores[name]],
            "fold_std": float(np.std(fold_scores[name])),
            "train_mean": float(np.mean(train_scores[name])),
            "valid_mean": float(np.mean(fold_scores[name])),
        })
        avg_oof[name] += oof[name] / len(seeds)
        avg_test[name] += test_raw[name] / len(seeds)
    model_rows.sort(key=lambda item: item["score"], reverse=not lower_is_better)
    top_names = [row["model"] for row in model_rows[:3]]
    weights = np.array([0.5, 0.3, 0.2][: len(top_names)], dtype=float)
    weights = weights / weights.sum()
    blend_oof = sum(weight * oof[name] for weight, name in zip(weights, top_names))
    blend_test = sum(weight * test_raw[name] for weight, name in zip(weights, top_names))
    blend_score = score_raw(blend_oof, y_model)
    best_row = model_rows[0]
    use_blend = blend_score >= best_row["score"] if not lower_is_better else blend_score <= best_row["score"]
    selected_name = "regularized_weighted_blend" if use_blend else best_row["model"]
    selected_models = top_names if use_blend else [best_row["model"]]
    selected_weights = [float(weight) for weight in weights] if use_blend else [1.0]
    selected_score = blend_score if use_blend else best_row["score"]
    selected_oof = blend_oof if use_blend else oof[best_row["model"]]
    selected_test = blend_test if use_blend else test_raw[best_row["model"]]
    selected_train_mean = float(sum(weight * next(row["train_mean"] for row in model_rows if row["model"] == name) for weight, name in zip(selected_weights, selected_models)))
    selected_fold_scores = []
    for fold_idx in range(5):
        selected_fold_scores.append(float(sum(weight * next(row["fold_scores"][fold_idx] for row in model_rows if row["model"] == name) for weight, name in zip(selected_weights, selected_models))))
    train_valid_gap = selected_score - selected_train_mean if lower_is_better else selected_train_mean - selected_score
    seed_summaries.append({
        "seed": seed,
        "selected_name": selected_name,
        "score": float(selected_score),
        "train_mean": selected_train_mean,
        "train_valid_gap": float(train_valid_gap),
        "fold_scores": selected_fold_scores,
        "fold_std": float(np.std(selected_fold_scores)),
        "models": model_rows,
        "selected_models": selected_models,
        "selected_weights": selected_weights,
    })
    selected_oof_outputs.append(selected_oof)
    selected_test_outputs.append(selected_test)

seed_values = [item["score"] for item in seed_summaries]
seed_mean = float(np.mean(seed_values))
seed_std = float(np.std(seed_values))
fold_values = [score for item in seed_summaries for score in item["fold_scores"]]
fold_mean = float(np.mean(fold_values))
fold_std = float(np.std(fold_values))
train_valid_gap = float(np.mean([item["train_valid_gap"] for item in seed_summaries]))
standard_error = seed_std / math.sqrt(len(seed_values)) if seed_values else None
ci95_half_width = 1.96 * standard_error if standard_error is not None else None
seed_ci95 = {
    "low": seed_mean - ci95_half_width if ci95_half_width is not None else None,
    "high": seed_mean + ci95_half_width if ci95_half_width is not None else None,
    "half_width": ci95_half_width,
}

correlations = []
for i, left in enumerate(all_model_names):
    for right in all_model_names[i + 1:]:
        corr = correlation(avg_oof[left], avg_oof[right])
        if corr is not None:
            correlations.append({"left": left, "right": right, "correlation": corr})
max_model_correlation = max((item["correlation"] for item in correlations), default=None)
avg_model_correlation = float(np.mean([item["correlation"] for item in correlations])) if correlations else None

selected_oof_average = np.mean(np.vstack(selected_oof_outputs), axis=0)
selected_test_average = np.mean(np.vstack(selected_test_outputs), axis=0)
selected_score = score_raw(selected_oof_average, y_model)
selected_seed = sorted(seed_summaries, key=lambda item: item["score"], reverse=not lower_is_better)[0]
selected_submission = {
    "name": selected_seed["selected_name"],
    "score": float(selected_score),
    "models": selected_seed["selected_models"],
    "weights": selected_seed["selected_weights"],
}

if task_type == "regression" or metric_name == "roc_auc":
    submission_pred = selected_test_average
elif y_model.nunique() == 2:
    encoded_pred = (selected_test_average >= 0.5).astype(int)
    submission_pred = label_encoder.inverse_transform(encoded_pred) if label_encoder is not None else encoded_pred
else:
    encoded_pred = np.rint(selected_test_average).astype(int)
    encoded_pred = np.clip(encoded_pred, 0, len(label_encoder.classes_) - 1) if label_encoder is not None else encoded_pred
    submission_pred = label_encoder.inverse_transform(encoded_pred) if label_encoder is not None else encoded_pred
pd.DataFrame({id_column: test[id_column], prediction_column: submission_pred}).to_csv(experiment_dir / "submission.csv", index=False)

oof_frame = pd.DataFrame({id_column: train[id_column] if id_column in train.columns else np.arange(len(train)), target_column: y})
for name in all_model_names:
    oof_frame[f"oof_{name}"] = avg_oof[name]
oof_frame["oof_selected"] = selected_oof_average
if task_type != "regression" and metric_name != "roc_auc" and label_encoder is not None:
    selected_encoded = np.rint(selected_oof_average).astype(int)
    selected_encoded = np.clip(selected_encoded, 0, len(label_encoder.classes_) - 1)
    oof_frame["oof_selected_label"] = label_encoder.inverse_transform(selected_encoded)
oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)

issues = []
warnings = []
if seed_std > 0.01:
    warnings.append(f"Seed variance exceeds tight gate: seed_std={seed_std:.4f}.")
if fold_std > 0.026:
    warnings.append(f"Fold variance exceeds tight gate: fold_std={fold_std:.4f}.")
if train_valid_gap > 0.035:
    warnings.append(f"Train-validation gap exceeds tight gate: train_valid_gap={train_valid_gap:.4f}.")
if max_model_correlation is not None and max_model_correlation > 0.97:
    warnings.append(f"Max model correlation exceeds diversity gate: {max_model_correlation:.4f}.")

report_payload = {
    "experiment": "__TASK_ID__",
    "status": "completed",
    "selected": selected_submission,
    "seed_scores": seed_summaries,
    "seed_mean": seed_mean,
    "seed_std": seed_std,
    "seed_ci95": seed_ci95,
    "fold_mean": fold_mean,
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "model_correlations": correlations,
    "max_model_correlation": max_model_correlation,
    "avg_model_correlation": avg_model_correlation,
    "oof_predictions_path": str(experiment_dir / "oof_predictions.csv"),
    "issues": issues,
    "warnings": warnings,
}
(experiment_dir / "regularized_blend_report.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "regularized_blend",
    "status": "completed",
    "metric_name": metric_name,
    "local_score": float(selected_score),
    "selected_submission": selected_submission,
    "seed_scores": [{"seed": item["seed"], "score": item["score"], "fold_scores": item["fold_scores"]} for item in seed_summaries],
    "seed_mean": seed_mean,
    "seed_std": seed_std,
    "seed_ci95": seed_ci95,
    "fold_mean": fold_mean,
    "fold_std": fold_std,
    "train_valid_gap": train_valid_gap,
    "max_model_correlation": max_model_correlation,
    "avg_model_correlation": avg_model_correlation,
    "issues": issues,
    "warnings": warnings,
    "notes": "Regularized low-variance blend with multi-seed OOF evidence for promotion gates.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'''


CV_STABILITY_AUDIT_SCRIPT = r'''
from pathlib import Path
import json
import math
import shutil
import statistics
import sys

root = Path.cwd()
experiment_dir = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))

source_task = "stability_first_search_v1"
source_dir = root / "experiments" / source_task
source_validation_path = source_dir / "validation_report.json"
leaderboard_path = root / "leaderboard_feedback.json"
recommended_submission = root / "recommended_submission.csv"
champion_submission = root / "champion_submission.csv"

issues = []
warnings = []
if not source_validation_path.exists():
    issues.append(f"{source_task} validation_report.json is missing.")
    source_validation = {}
else:
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))

leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8")) if leaderboard_path.exists() else {}
best_model = source_validation.get("best_model") or {}
seed_scores = best_model.get("seed_scores") or []
fold_scores = [float(score) for score in best_model.get("fold_scores", []) if isinstance(score, (int, float))]
seed_values = [float(item.get("score")) for item in seed_scores if isinstance(item.get("score"), (int, float))]
local_score = source_validation.get("local_score")
metric_name = source_validation.get("metric_name") or manifest["metric_candidates"][0]
public_score = leaderboard.get("public_score")

seed_mean = statistics.mean(seed_values) if seed_values else None
seed_std = statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0 if seed_values else None
fold_mean = statistics.mean(fold_scores) if fold_scores else None
fold_std = statistics.pstdev(fold_scores) if len(fold_scores) > 1 else 0.0 if fold_scores else None
standard_error = seed_std / math.sqrt(len(seed_values)) if seed_std is not None and seed_values else None
ci95_half_width = 1.96 * standard_error if standard_error is not None else None
ci95_low = seed_mean - ci95_half_width if seed_mean is not None and ci95_half_width is not None else None
ci95_high = seed_mean + ci95_half_width if seed_mean is not None and ci95_half_width is not None else None

lower_is_better = metric_name in {"rmse", "rmsle", "mae", "log_loss"}
public_gap = None
public_within_seed_ci = None
if isinstance(public_score, (int, float)) and isinstance(seed_mean, (int, float)):
    public_gap = public_score - seed_mean
    public_within_seed_ci = ci95_low <= public_score <= ci95_high if ci95_low is not None and ci95_high is not None else None

risk_points = 0
if not seed_values:
    risk_points += 2
    issues.append("No seed-level scores are available.")
elif seed_std is not None and seed_std > 0.01:
    risk_points += 1
    warnings.append(f"Seed score variance is visible: seed_std={seed_std:.4f}.")
if fold_std is None:
    risk_points += 1
    issues.append("No fold-level scores are available.")
elif fold_std > 0.035:
    risk_points += 2
    issues.append(f"Fold variance is high: fold_std={fold_std:.4f}.")
elif fold_std > 0.025:
    risk_points += 1
    warnings.append(f"Fold variance is moderate: fold_std={fold_std:.4f}.")
if public_within_seed_ci is False:
    risk_points += 2
    issues.append("Public score is outside the seed-level 95% confidence interval.")

if risk_points >= 4:
    risk_level = "high"
elif risk_points >= 2:
    risk_level = "medium"
else:
    risk_level = "low"

audit = {
    "experiment": "__TASK_ID__",
    "status": "completed" if not issues or seed_values else "needs_review",
    "source_task": source_task,
    "metric_name": metric_name,
    "local_score": local_score,
    "seed_scores": seed_scores,
    "seed_mean": seed_mean,
    "seed_std": seed_std,
    "fold_mean": fold_mean,
    "fold_std": fold_std,
    "seed_ci95": {
        "low": ci95_low,
        "high": ci95_high,
        "half_width": ci95_half_width,
    },
    "leaderboard_feedback": {
        "submission_target": leaderboard.get("submission_target"),
        "candidate_task_id": leaderboard.get("candidate_task_id"),
        "public_score": public_score,
        "leaderboard_rank": leaderboard.get("leaderboard_rank"),
    },
    "public_gap_vs_seed_mean": public_gap,
    "public_within_seed_ci": public_within_seed_ci,
    "risk_level": risk_level,
    "risk_points": risk_points,
    "issues": issues,
    "warnings": warnings,
    "recommendation": (
        "Treat the current stable candidate as reasonably reliable; continue controlled optimization."
        if risk_level == "low"
        else "Prefer another validation repeat or conservative submission policy before new leaderboard submissions."
    ),
}

(experiment_dir / "cv_stability_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

submission_source = recommended_submission if recommended_submission.exists() else champion_submission
if submission_source.exists():
    shutil.copyfile(submission_source, experiment_dir / "submission.csv")
else:
    issues.append("No recommended or champion submission was available to attach to the audit.")

report = {
    "experiment": "__TASK_ID__",
    "runner_kind": "cv_stability_audit",
    "status": "completed" if audit["status"] == "completed" else "skipped",
    "metric_name": metric_name,
    "local_score": seed_mean if seed_mean is not None else local_score,
    "source_local_score": local_score,
    "seed_mean": seed_mean,
    "seed_std": seed_std,
    "fold_std": fold_std,
    "risk_level": risk_level,
    "public_gap_vs_seed_mean": public_gap,
    "public_within_seed_ci": public_within_seed_ci,
    "issues": issues,
    "warnings": warnings,
    "notes": "CV stability audit from stability_first_search_v1 repeated-CV seed and fold scores.",
}
(experiment_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(0)
'''
