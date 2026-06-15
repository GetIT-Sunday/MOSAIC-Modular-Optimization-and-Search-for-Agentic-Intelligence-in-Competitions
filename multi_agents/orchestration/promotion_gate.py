from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class PromotionGateResult:
    status: str
    review_path: Path
    markdown_path: Path
    promoted_submission_path: Optional[Path]


class PromotionGateEvaluator:
    """Evaluate Remote Brain promotion gates against experiment evidence."""

    LOWER_IS_BETTER = {"rmse", "rmsle", "mae", "log_loss"}
    RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "unknown": 2, "high": 3}

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def evaluate(self) -> PromotionGateResult:
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        plan = self._read_json(self.competition_dir / "llm_experiment_plan.json")
        queue = self._read_json(self.competition_dir / "experiment_queue.json")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        recommendations = plan.get("recommended_experiments") or queue.get("queue") or []
        champion = champion_selection.get("champion") or {}
        lower_is_better = self._lower_is_better(champion, recommendations)

        evaluations = [
            self._evaluate_item(item, champion, lower_is_better)
            for item in recommendations
            if isinstance(item, dict)
        ]
        promoted = self._select_promoted(evaluations, lower_is_better)
        promoted_submission_path = None
        if promoted:
            source = Path(promoted["submission_path"])
            if source.exists():
                promoted_submission_path = self.competition_dir / "promoted_submission.csv"
                shutil.copyfile(source, promoted_submission_path)
            else:
                promoted["decision"] = "hold_candidate"
                promoted["issues"].append("Promoted candidate submission file is missing.")
                promoted = None

        status = "pass" if promoted else ("needs_review" if evaluations else "blocked")
        review = {
            "competition_name": manifest.competition_name,
            "status": status,
            "decision": "promote_candidate" if promoted else "hold_all_candidates",
            "promoted_candidate": self._public_evaluation(promoted),
            "promoted_submission_path": str(promoted_submission_path) if promoted_submission_path else None,
            "champion": champion,
            "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "evaluations": [self._public_evaluation(item) for item in evaluations],
            "issues": [] if promoted else self._aggregate_issues(evaluations),
            "next_action": (
                "Use promoted_submission.csv for submission policy or run champion selection with promotion context."
                if promoted
                else "Ask Remote Brain to revise the plan or run experiments that satisfy promotion gates."
            ),
        }
        review_path = self.competition_dir / "promotion_gate_review.json"
        markdown_path = self.competition_dir / "promotion_gate_review.md"
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(self._render_markdown(review), encoding="utf-8")

        ledger_entry = self.ledger.create_entry(
            task_id="promotion_gate_review",
            agent="promotion_gate",
            title="Evaluate promotion gates",
            status=status,
            input_payload=review,
            prompt=(
                "Evaluate Remote Brain promotion_gate rules against validation reports, "
                "validator results, and runner-specific evidence before allowing promotion."
            ),
            scorecard={
                "agent": "promotion_gate",
                "task_id": "promotion_gate_review",
                "status": status,
                "scores": {
                    "evaluated_candidates": len(evaluations),
                    "promoted_task_id": promoted.get("task_id") if promoted else "n/a",
                    "promoted_score": promoted.get("local_score") if promoted else "n/a",
                },
                "metric_name": promoted.get("metric_name") if promoted else champion.get("metric_name"),
                "local_score": promoted.get("local_score") if promoted else None,
                "issues": review["issues"],
                "recommended_human_action": "continue" if promoted else "patch_prompt",
            },
            artifacts={
                "promotion_gate_review": review_path,
                "promotion_gate_review_markdown": markdown_path,
                "promoted_submission": promoted_submission_path or Path(""),
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest.competition_name,
                profile_name="tabular_classic",
                task_id="promotion_gate_review",
                status=status,
                metric_name=promoted.get("metric_name") if promoted else champion.get("metric_name"),
                local_score=promoted.get("local_score") if promoted else None,
                submission_path=str(promoted_submission_path) if promoted_submission_path else None,
                brain_review_path=str(review_path),
                artifacts=[str(review_path), str(markdown_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=review["next_action"],
            )
        )
        return PromotionGateResult(status, review_path, markdown_path, promoted_submission_path)

    def _evaluate_item(
        self,
        item: Dict[str, Any],
        champion: Dict[str, Any],
        lower_is_better: bool,
    ) -> Dict[str, Any]:
        task_id = self._task_id(item)
        experiment_dir = self.competition_dir / "experiments" / task_id
        validation = self._read_json(experiment_dir / "validation_report.json")
        validator = self._read_json(experiment_dir / "validator_result.json")
        gate = item.get("promotion_gate") if isinstance(item.get("promotion_gate"), dict) else {}
        evidence_needed = item.get("evidence_needed") if isinstance(item.get("evidence_needed"), list) else []
        runner_kind = item.get("runner_kind") or validation.get("runner_kind")
        evidence = self._collect_evidence(experiment_dir, validation, validator, runner_kind)
        issues: List[str] = []
        warnings: List[str] = []

        if validation.get("status") != "completed":
            issues.append(f"validation_report status is {validation.get('status') or 'missing'}.")
        if validator.get("ok") is not True:
            issues.append("validator_result.json did not pass.")
        if str(runner_kind or "").lower() in {
            "tabular_mlp",
            "tabular_resnet",
            "star_specialist_lgbm",
            "star_specialist_threshold_tuning",
            "classwise_blend",
            "clean_oof_blend",
        } and evidence.get("valid_oof") is not True:
            issues.append("runner requires a valid oof_predictions.csv before promotion.")
        for required in evidence_needed:
            if not self._has_evidence(required, experiment_dir, validation, evidence):
                warnings.append(f"Requested evidence is missing or indirect: {required}")
        issues.extend(self._evaluate_gate(gate, validation, validator, evidence, champion, lower_is_better))

        action_type = self._action_type(item)
        local_score = validation.get("local_score")
        can_promote = (
            action_type != "audit"
            and not issues
            and validator.get("ok") is True
            and isinstance(local_score, (int, float))
            and (experiment_dir / "submission.csv").exists()
        )
        if action_type == "audit" and not issues:
            decision = "diagnostic_complete"
        else:
            decision = "promote_candidate" if can_promote else "hold_candidate"
        return {
            "task_id": task_id,
            "title": item.get("title") or task_id,
            "runner_kind": runner_kind,
            "action_type": action_type,
            "decision": decision,
            "gate_passed": can_promote or decision == "diagnostic_complete",
            "can_promote": can_promote,
            "metric_name": validation.get("metric_name") or champion.get("metric_name"),
            "local_score": local_score,
            "champion_score": champion.get("local_score"),
            "validation_report_path": experiment_dir / "validation_report.json",
            "validator_result_path": experiment_dir / "validator_result.json",
            "submission_path": experiment_dir / "submission.csv",
            "promotion_gate": gate,
            "evidence_needed": evidence_needed,
            "evidence": evidence,
            "issues": issues,
            "warnings": warnings,
        }

    def _evaluate_gate(
        self,
        gate: Dict[str, Any],
        validation: Dict[str, Any],
        validator: Dict[str, Any],
        evidence: Dict[str, Any],
        champion: Dict[str, Any],
        lower_is_better: bool,
    ) -> List[str]:
        issues: List[str] = []
        if gate.get("validator_must_pass") is True and validator.get("ok") is not True:
            issues.append("promotion_gate requires validator_must_pass.")
        if gate.get("manual_submit_allowed") is False:
            # This gate blocks submit routing, not candidate promotion.
            pass
        min_delta = self._number(gate.get("min_local_score_delta"))
        score = self._number(validation.get("local_score"))
        champion_score = self._number(champion.get("local_score"))
        if min_delta is not None and score is not None and champion_score is not None:
            delta = champion_score - score if lower_is_better else score - champion_score
            if delta < min_delta:
                issues.append(f"local score delta {delta:.4f} is below required {min_delta:.4f}.")
        elif min_delta is not None:
            issues.append("promotion_gate requires min_local_score_delta, but score evidence is missing.")

        max_risk = gate.get("max_risk_level")
        if max_risk:
            risk = str(evidence.get("risk_level") or validation.get("risk_level") or "unknown")
            if self.RISK_ORDER.get(risk, 2) > self.RISK_ORDER.get(str(max_risk), 2):
                issues.append(f"risk_level {risk} exceeds max_risk_level {max_risk}.")

        if gate.get("public_within_seed_ci") is True and evidence.get("public_within_seed_ci") is not True:
            issues.append("promotion_gate requires public_within_seed_ci=True.")

        min_score = self._number(gate.get("min_local_score"))
        if min_score is not None and score is not None:
            if lower_is_better:
                if score > min_score:
                    issues.append(f"local score {score:.4f} is above maximum target {min_score:.4f}.")
            elif score < min_score:
                issues.append(f"local score {score:.4f} is below required {min_score:.4f}.")
        elif min_score is not None:
            issues.append("promotion_gate requires min_local_score, but score evidence is missing.")

        max_drop = self._number(gate.get("max_local_score_drop"))
        if max_drop is not None and score is not None and champion_score is not None:
            drop = score - champion_score if lower_is_better else champion_score - score
            if drop > max_drop:
                issues.append(f"local score drop {drop:.4f} exceeds allowed {max_drop:.4f}.")
        elif max_drop is not None:
            issues.append("promotion_gate requires max_local_score_drop, but score evidence is missing.")

        for key, raw_expected in gate.items():
            if key in {"validator_must_pass", "manual_submit_allowed", "min_local_score", "min_local_score_delta", "max_local_score_drop", "max_risk_level", "public_within_seed_ci", "notes", "max_drift_score_reviewed"}:
                continue
            actual = evidence.get(key)
            if actual is None:
                actual = validation.get(key)
            issue = self._compare_threshold(key, actual, raw_expected)
            if issue:
                issues.append(issue)
        return issues

    def _compare_threshold(self, key: str, actual: Any, expected: Any) -> Optional[str]:
        if isinstance(expected, bool):
            if bool(actual) != expected:
                return f"{key} expected {expected}, got {actual}."
            return None
        if isinstance(expected, (int, float)):
            if self._number(actual) is None:
                return f"{key} threshold requires numeric evidence."
            if self._number(actual) != float(expected):
                return f"{key} expected {expected}, got {actual}."
            return None
        if not isinstance(expected, str):
            return None
        match = re.match(r"^\s*(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?)", expected)
        if not match:
            return None
        actual_number = self._number(actual)
        if actual_number is None:
            return f"{key} requires numeric evidence for threshold {expected}."
        op, threshold_text = match.groups()
        threshold = float(threshold_text)
        ok = {
            "<=": actual_number <= threshold,
            ">=": actual_number >= threshold,
            "<": actual_number < threshold,
            ">": actual_number > threshold,
            "==": actual_number == threshold,
        }[op]
        if not ok:
            return f"{key}={actual_number:.4f} does not satisfy {expected}."
        return None

    def _collect_evidence(
        self,
        experiment_dir: Path,
        validation: Dict[str, Any],
        validator: Dict[str, Any],
        runner_kind: Any,
    ) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {
            "validator_ok": validator.get("ok"),
            "local_score": validation.get("local_score"),
            "metric_name": validation.get("metric_name"),
            "runner_kind": runner_kind,
            "valid_oof": self._valid_oof_file(experiment_dir / "oof_predictions.csv"),
        }
        for name in [
            "cv_stability_audit.json",
            "distribution_shift_audit.json",
            "overfitting_audit.json",
            "regularized_blend_report.json",
            "nn_training_report.json",
            "specialist_report.json",
            "clean_blend_report.json",
            "risk_audit.json",
        ]:
            data = self._read_json(experiment_dir / name)
            if data:
                evidence.update(self._flatten_evidence(data))
                evidence[name] = data
        evidence.update(self._flatten_evidence(validation))
        if "fold_std" not in evidence:
            fold_stds = [
                self._number(item.get("fold_std"))
                for item in validation.get("model_scores", [])
                if isinstance(item, dict)
            ]
            fold_stds = [item for item in fold_stds if item is not None]
            if fold_stds:
                evidence["fold_std"] = max(fold_stds)
        return evidence

    def _valid_oof_file(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            import csv

            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                if not header:
                    return False
                return any(True for _ in reader)
        except Exception:
            return False

    def _flatten_evidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flat = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flat[key] = value
            elif key == "selected" and isinstance(value, dict):
                score = value.get("score")
                if score is not None:
                    flat["selected_score"] = score
            elif key == "selected_submission" and isinstance(value, dict):
                score = value.get("score")
                if score is not None:
                    flat["selected_score"] = score
        return flat

    def _has_evidence(
        self,
        required: Any,
        experiment_dir: Path,
        validation: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> bool:
        text = str(required).strip()
        if not text:
            return True
        if text.endswith((".json", ".csv", ".txt", ".log")):
            return (experiment_dir / text).exists()
        normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        if normalized in evidence:
            return True
        if normalized in {re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") for key in evidence}:
            return True
        return text in json.dumps(validation, ensure_ascii=False) or text in json.dumps(evidence, ensure_ascii=False)

    def _select_promoted(
        self,
        evaluations: List[Dict[str, Any]],
        lower_is_better: bool,
    ) -> Optional[Dict[str, Any]]:
        candidates = [item for item in evaluations if item.get("can_promote")]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.get("local_score"), reverse=not lower_is_better)
        return candidates[0]

    def _aggregate_issues(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        issues = []
        for item in evaluations:
            if item.get("decision") == "diagnostic_complete":
                continue
            if item.get("issues"):
                issues.extend(f"{item.get('task_id')}: {issue}" for issue in item["issues"])
            else:
                issues.append(f"{item.get('task_id')}: promotion gate did not select this candidate.")
        return issues or ["No promotable candidate was found."]

    def _lower_is_better(self, champion: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> bool:
        metric = champion.get("metric_name")
        if not metric:
            metric = next((item.get("metric_name") for item in recommendations if item.get("metric_name")), "")
        return metric in self.LOWER_IS_BETTER

    def _action_type(self, item: Dict[str, Any]) -> str:
        runner_kind = str(item.get("runner_kind") or "").lower()
        if runner_kind in {"cv_stability_audit", "distribution_shift_audit", "overfitting_audit"}:
            return "audit"
        identity = " ".join(str(item.get(key, "")) for key in ["task_id", "title"]).lower()
        if "submit" in identity:
            return "manual_submit"
        return "coding_experiment"

    def _task_id(self, item: Dict[str, Any]) -> str:
        raw = item.get("task_id") or item.get("experiment_id") or "unknown_task"
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(raw))
        return safe.strip("_") or "unknown_task"

    def _public_evaluation(self, item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        public = dict(item)
        for key in ["validation_report_path", "validator_result_path", "submission_path"]:
            value = public.get(key)
            public[key] = str(value) if value else None
        return public

    def _render_markdown(self, review: Dict[str, Any]) -> str:
        lines = [
            "# Promotion Gate Review",
            "",
            f"Status: `{review.get('status')}`",
            f"Decision: `{review.get('decision')}`",
            "",
            "## Promoted Candidate",
            "",
            "```json",
            json.dumps(review.get("promoted_candidate"), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Evaluations",
        ]
        for item in review.get("evaluations", []):
            lines.extend(
                [
                    "",
                    f"### {item.get('task_id')}",
                    "",
                    f"- Decision: {item.get('decision')}",
                    f"- Runner kind: {item.get('runner_kind')}",
                    f"- Local score: {item.get('local_score')}",
                    f"- Issues: {'; '.join(item.get('issues') or []) or 'none'}",
                    f"- Warnings: {'; '.join(item.get('warnings') or []) or 'none'}",
                ]
            )
        if review.get("issues"):
            lines.extend(["", "## Review Issues", ""])
            lines.extend(f"- {issue}" for issue in review["issues"])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        return None

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
