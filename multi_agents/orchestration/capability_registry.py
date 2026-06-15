from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    path: str
    word_count: int
    triggers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    purpose: str
    runner_kinds: List[str]
    required_artifacts: List[str]
    risk_controls: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillRegistry:
    """Loads AutoKaggle methodology skills without putting full bodies in every prompt."""

    def __init__(self, skill_root: Path = SKILL_ROOT):
        self.skill_root = skill_root.resolve()

    def list_skills(self) -> List[SkillSpec]:
        skills: List[SkillSpec] = []
        for path in sorted(self.skill_root.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            meta, body = self._split_frontmatter(text)
            name = meta.get("name") or path.parent.name
            description = meta.get("description", "")
            skills.append(
                SkillSpec(
                    name=str(name),
                    description=str(description),
                    path=str(path.relative_to(self.skill_root.parents[1])),
                    word_count=len(body.split()),
                    triggers=self._triggers(description),
                )
            )
        return skills

    def summary(self) -> Dict[str, Any]:
        skills = self.list_skills()
        return {
            "skill_count": len(skills),
            "skills": [skill.to_dict() for skill in skills],
        }

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[Dict[str, str], str]:
        stripped = text.lstrip()
        if not stripped.startswith("---"):
            return {}, text
        parts = stripped.split("---", 2)
        if len(parts) < 3:
            return {}, text
        raw_meta = parts[1]
        body = parts[2]
        meta: Dict[str, str] = {}
        for line in raw_meta.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta, body

    @staticmethod
    def _triggers(description: str) -> List[str]:
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", description.lower())
        keep = []
        stop = {"when", "auto", "needs", "after", "with", "into", "against", "whether", "before"}
        for word in words:
            if len(word) < 4 or word in stop:
                continue
            if word not in keep:
                keep.append(word)
        return keep[:12]


class HarnessRegistry:
    """Static v1 registry for experiment harness capabilities."""

    def list_harnesses(self) -> List[HarnessSpec]:
        return [
            HarnessSpec(
                name="leaderboard_snapshot_harness",
                purpose="Fetch or ingest Kaggle leaderboard snapshots and estimate target gaps.",
                runner_kinds=["leaderboard_target"],
                required_artifacts=["leaderboard_target.json"],
                risk_controls=["timestamped_snapshot", "metric_direction_check", "target_confidence"],
            ),
            HarnessSpec(
                name="stratified_cv_stability_harness",
                purpose="Run stratified CV stability audits for tabular classification.",
                runner_kinds=["cv_stability_audit"],
                required_artifacts=["cv_stability_audit.json", "validation_report.json", "validator_result.json"],
                risk_controls=["fold_std", "oof_confusion_matrix", "class_distribution_check"],
            ),
            HarnessSpec(
                name="stratified_gbdt_oof_harness",
                purpose="Train GBDT models with stratified OOF validation and validated submission artifacts.",
                runner_kinds=["lightgbm", "catboost", "xgboost"],
                required_artifacts=["validation_report.json", "validator_result.json", "submission.csv", "oof_predictions.csv"],
                risk_controls=["fold_scores", "feature_importance", "train_valid_gap"],
            ),
            HarnessSpec(
                name="tabular_nn_oof_harness",
                purpose="Train tabular neural network style models with OOF validation and submission artifacts.",
                runner_kinds=["tabular_mlp", "tabular_resnet"],
                required_artifacts=["validation_report.json", "validator_result.json", "submission.csv", "oof_predictions.csv", "nn_training_report.json", "model_config.json"],
                risk_controls=["fold_scores", "train_valid_gap", "backend_fallback"],
            ),
            HarnessSpec(
                name="class_specialist_oof_harness",
                purpose="Train class-specialist correction models using existing per-class OOF evidence.",
                runner_kinds=["star_specialist_lgbm", "star_specialist_threshold_tuning"],
                required_artifacts=["validation_report.json", "validator_result.json", "submission.csv", "oof_predictions.csv", "specialist_report.json", "per_class_oof_report.json"],
                risk_controls=["target_class_recall", "overall_score_floor", "coverage_rate", "threshold_search"],
            ),
            HarnessSpec(
                name="regularized_oof_blend_harness",
                purpose="Build constrained OOF blends from validated candidate models.",
                runner_kinds=["regularized_blend"],
                required_artifacts=["regularized_blend_report.json", "validation_report.json", "validator_result.json", "submission.csv"],
                risk_controls=["pairwise_correlation", "weight_regularization", "fold_level_comparison"],
            ),
            HarnessSpec(
                name="clean_oof_blend_harness",
                purpose="Filter invalid OOF candidates and build clean diversity-aware blends.",
                runner_kinds=["classwise_blend", "clean_oof_blend"],
                required_artifacts=["clean_blend_report.json", "oof_diversity_report.json", "validation_report.json", "validator_result.json", "submission.csv"],
                risk_controls=["invalid_oof_filter", "candidate_count", "classwise_accuracy"],
            ),
        ]

    def summary(self) -> Dict[str, Any]:
        harnesses = self.list_harnesses()
        return {
            "harness_count": len(harnesses),
            "harnesses": [harness.to_dict() for harness in harnesses],
        }

    def default_for_runner(self, runner_kind: Optional[str]) -> str:
        runner = str(runner_kind or "").strip()
        for harness in self.list_harnesses():
            if runner in harness.runner_kinds:
                return harness.name
        if runner in {"distribution_shift_audit", "overfitting_audit", "per_class_oof_audit", "oof_diversity_audit"}:
            return "stratified_cv_stability_harness"
        return "stratified_gbdt_oof_harness"
