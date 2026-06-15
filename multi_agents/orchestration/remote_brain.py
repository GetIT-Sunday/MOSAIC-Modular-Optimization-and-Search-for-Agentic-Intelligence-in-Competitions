from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .capability_registry import HarnessRegistry, SkillRegistry
from .human_gate import HumanGate, HumanGateDecision
from .ingestion import CompetitionIngestor
from .leaderboard_feedback_freshness import LeaderboardFeedbackFreshnessAuditor
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class RemoteBrainReviewResult:
    markdown_path: Path
    json_path: Path
    ledger_html_path: Path
    used_llm: bool


class RemoteBrainReviewer:
    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
        use_llm: bool = True,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)
        self.use_llm = use_llm
        self.skill_registry = SkillRegistry()
        self.harness_registry = HarnessRegistry()

    def review(self) -> RemoteBrainReviewResult:
        context = self._build_context()
        prompt = self._build_prompt(context)
        plan, used_llm, raw_reply = self._generate_plan(prompt, context)
        markdown_path = self.competition_dir / "llm_experiment_plan.md"
        json_path = self.competition_dir / "llm_experiment_plan.json"
        reply_path = self.competition_dir / "remote_brain_reply.md"

        markdown_path.write_text(self._render_markdown(plan), encoding="utf-8")
        json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        reply_path.write_text(raw_reply, encoding="utf-8")

        latest_score = self._best_score(plan)
        leaderboard = plan.get("leaderboard_feedback") or context.get("leaderboard_feedback") or {}
        ledger_entry = self.ledger.create_entry(
            task_id="remote_brain_review",
            agent="remote_brain",
            title="Review baselines and plan next LLM-guided experiments",
            status="pass" if plan.get("next_action") != "stop" else "needs_review",
            input_payload=context,
            prompt=prompt,
            scorecard={
                "agent": "remote_brain",
                "task_id": "remote_brain_review",
                "status": "pass" if plan.get("recommended_experiments") else "needs_review",
                "scores": {
                    "baseline_context_loaded": 5 if context.get("baseline_review") else 1,
                    "human_gate_respected": 5,
                    "recommended_experiment_count": len(plan.get("recommended_experiments", [])),
                    "llm_used": 5 if used_llm else 2,
                    "leaderboard_feedback_loaded": 5 if context.get("leaderboard_feedback") else 1,
                    "public_score": leaderboard.get("public_score", "n/a"),
                    "leaderboard_rank": leaderboard.get("leaderboard_rank", "n/a"),
                },
                "metric_name": plan.get("current_best_baseline", {}).get("metric_name"),
                "local_score": latest_score,
                "issues": plan.get("risks", []),
                "recommended_human_action": (
                    "continue" if plan.get("next_action") != "stop" else "stop"
                ),
            },
            artifacts={
                "llm_experiment_plan": markdown_path,
                "llm_experiment_plan_json": json_path,
                "agent_reply": reply_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=context["competition_name"],
                profile_name=context.get("profile_name", "unknown"),
                task_id="remote_brain_review",
                status="planned" if plan.get("recommended_experiments") else "needs_review",
                metric_name=plan.get("current_best_baseline", {}).get("metric_name"),
                local_score=latest_score,
                public_score=leaderboard.get("public_score") if isinstance(leaderboard.get("public_score"), (int, float)) else None,
                leaderboard_rank=leaderboard.get("leaderboard_rank") if isinstance(leaderboard.get("leaderboard_rank"), int) else None,
                brain_review_path=str(json_path),
                artifacts=[
                    str(markdown_path),
                    str(json_path),
                    str(reply_path),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=plan.get("diagnosis", ""),
            )
        )
        return RemoteBrainReviewResult(
            markdown_path=markdown_path,
            json_path=json_path,
            ledger_html_path=self.competition_dir / ledger_entry.html_report_path,
            used_llm=used_llm,
        )

    def refresh_existing_plan_gates(self) -> RemoteBrainReviewResult:
        """Normalize an existing Brain plan without changing its chosen experiment."""

        context = self._build_context()
        plan_path = self.competition_dir / "llm_experiment_plan.json"
        plan = self._read_json(plan_path)
        normalized = self._normalize_plan(plan, context, rename_completed=False)
        markdown_path = self.competition_dir / "llm_experiment_plan.md"
        reply_path = self.competition_dir / "remote_brain_reply.md"
        markdown_path.write_text(self._render_markdown(normalized), encoding="utf-8")
        plan_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
        reply_path.write_text(
            json.dumps(
                {
                    "status": "refreshed_existing_plan_gates",
                    "recommended_experiments": normalized.get("recommended_experiments", []),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        entry = self.ledger.create_entry(
            task_id="remote_brain_plan_gate_refresh",
            agent="remote_brain",
            title="Refresh runner-specific plan gates",
            status="pass" if normalized.get("recommended_experiments") else "needs_review",
            input_payload=normalized,
            prompt="Normalize the existing Brain plan with runner-specific evidence and promotion gate rules without selecting a new experiment.",
            scorecard={
                "agent": "remote_brain",
                "task_id": "remote_brain_plan_gate_refresh",
                "status": "pass" if normalized.get("recommended_experiments") else "needs_review",
                "scores": {
                    "recommended_experiment_count": len(normalized.get("recommended_experiments", [])),
                    "llm_used": 0,
                },
                "metric_name": normalized.get("current_best_baseline", {}).get("metric_name"),
                "local_score": self._best_score(normalized),
                "issues": normalized.get("risks", []),
                "recommended_human_action": "continue",
            },
            artifacts={
                "llm_experiment_plan": markdown_path,
                "llm_experiment_plan_json": plan_path,
                "agent_reply": reply_path,
            },
        )
        return RemoteBrainReviewResult(
            markdown_path=markdown_path,
            json_path=plan_path,
            ledger_html_path=self.competition_dir / entry.html_report_path,
            used_llm=False,
        )

    def _build_context(self) -> Dict[str, Any]:
        manifest_model = CompetitionIngestor(self.competition_dir).build_manifest()
        manifest_model.write_json(self.competition_dir / "data_manifest.json")
        manifest = manifest_model.to_dict()
        baseline_review = self._read_json(self.competition_dir / "baseline_review.json")
        enhancement_review = self._read_json(self.competition_dir / "enhancement_review.json")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        leaderboard_target = self._read_json(self.competition_dir / "leaderboard_target.json")
        leaderboard_feedback = self._read_json(self.competition_dir / "leaderboard_feedback.json")
        leaderboard_gap_audit = self._read_json(self.competition_dir / "leaderboard_gap_audit.json")
        leaderboard_feedback_loop = self._read_json(self.competition_dir / "leaderboard_feedback_loop.json")
        agent_loop_state = self._read_json(self.competition_dir / "agent_loop_state.json")
        champion_state = self._read_json(self.competition_dir / "champion_state.json")
        candidate_pool = self._read_json(self.competition_dir / "candidate_pool.json")
        mac_brain_handoff = self._read_json(self.competition_dir / "mac_brain_handoff.json")
        mac_brain_diagnosis = self._read_json(self.competition_dir / "mac_brain_diagnosis.json")
        mac_brain_context = self._read_json(self.competition_dir / "mac_brain_context.json")
        remote_brain_mission = self._read_json(self.competition_dir / "remote_brain_mission.json")
        manual_package = self._read_json(self.competition_dir / "manual_submission_package" / "manifest.json")
        post_submit_workflow = self._read_json(self.competition_dir / "post_submit_workflow.json")
        leaderboard_feedback_freshness = LeaderboardFeedbackFreshnessAuditor(self.competition_dir).audit(
            leaderboard_feedback=leaderboard_feedback,
            manual_package=manual_package,
            post_submit_workflow=post_submit_workflow,
        )
        experiment_queue = self._read_json(self.competition_dir / "experiment_queue.json")
        submit_decision_handoff = self._read_json(self.competition_dir / "submit_decision_handoff.json")
        submission_decision_review = self._read_json(
            self.competition_dir / "submission_decision_review.json"
        )
        risk_audits = []
        for path in sorted((self.competition_dir / "experiments").glob("*/risk_audit.json")):
            risk_audits.append(
                {
                    "experiment": path.parent.name,
                    "risk_audit": self._read_json(path),
                }
            )
        cv_stability_audits = []
        for path in sorted((self.competition_dir / "experiments").glob("*/cv_stability_audit.json")):
            cv_stability_audits.append(
                {
                    "experiment": path.parent.name,
                    "cv_stability_audit": self._read_json(path),
                }
            )
        ledgers = self.ledger.list_entries()
        human_gates = HumanGate.collect(
            [self.competition_dir / entry.human_review_path for entry in ledgers]
        )
        intervention_gates = [gate for gate in human_gates if gate.is_intervention]
        baseline_reports = []
        for path in sorted((self.competition_dir / "experiments").glob("*/validation_report.json")):
            validator_path = path.parent / "validator_result.json"
            baseline_reports.append(
                {
                    "experiment": path.parent.name,
                    "validation_report": self._read_json(path),
                    "validator_result": self._read_json(validator_path),
                    "run_log_tail": self._read_tail(path.parent / "run.log"),
                }
            )
        return {
            "competition_name": manifest.get("competition_name", self.competition_dir.name),
            "profile_name": "tabular_classic",
            "manifest": manifest,
            "baseline_review": baseline_review,
            "enhancement_review": enhancement_review,
            "champion_selection": champion_selection,
            "leaderboard_target": leaderboard_target,
            "leaderboard_feedback": leaderboard_feedback,
            "leaderboard_feedback_freshness": leaderboard_feedback_freshness,
            "leaderboard_gap_audit": leaderboard_gap_audit,
            "leaderboard_feedback_loop": leaderboard_feedback_loop,
            "agent_loop_state": agent_loop_state,
            "champion_state": champion_state,
            "candidate_pool": candidate_pool,
            "mac_brain_handoff": mac_brain_handoff,
            "mac_brain_diagnosis": mac_brain_diagnosis,
            "mac_brain_context": mac_brain_context,
            "remote_brain_mission": remote_brain_mission,
            "experiment_queue": experiment_queue,
            "submit_decision_handoff": submit_decision_handoff,
            "submission_decision_review": submission_decision_review,
            "leaderboard_memory_summary": self.memory.leaderboard_summary("tabular_classic"),
            "risk_audits": risk_audits,
            "cv_stability_audits": cv_stability_audits,
            "baseline_reports": baseline_reports,
            "skill_registry": self.skill_registry.summary(),
            "harness_registry": self.harness_registry.summary(),
            "human_gates": [gate.to_dict() for gate in human_gates],
            "active_human_interventions": [gate.to_dict() for gate in intervention_gates],
        }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        compact = json.dumps(context, ensure_ascii=False, indent=2)
        return f"""You are the remote project Brain inside AutoKaggle.

Analyze the latest remote Linux experiment artifacts and produce the next experiment plan.

Rules:
- Respect human gates. If any gate says stop, next_action must be stop.
- If gates say patch_prompt, incorporate the notes into recommended experiments.
- Do not write code. Produce a plan for the next CodingAgent/Runner stage.
- Recommend concrete, small experiments that can be audited in the control panel.
- Prefer experiments that improve the current best baseline.
- Use leaderboard_feedback and leaderboard_memory_summary when present.
- Use leaderboard_target when present. Do not treat a high local baseline as sufficient if gap_to_silver or gap_to_top is still material.
- Treat leaderboard_feedback as current-candidate evidence only when leaderboard_feedback_freshness.is_current is true; stale feedback is historical context and should trigger a new upload/feedback gate for the current package.
- Use leaderboard_gap_audit and leaderboard_feedback_loop when present; they are stronger evidence than raw public score alone.
- Use submit_decision_handoff to understand which candidate was actually cleared for human leaderboard upload.
- If public leaderboard feedback is worse than local CV, prioritize stability, leakage, validation split, and overfitting audits before chasing local CV gains.
- If submission_decision_review says pause_manual_submit, do not recommend another manual submit. Recommend a concrete experiment or audit that addresses the listed issues.
- If experiment_queue has no next_runnable, produce a new runnable plan that unblocks the loop.
- If remote_brain_mission is present, treat it as the Mac Brain strategy contract. Follow its do_not_repeat rules, must_use_evidence list, experiment_portfolio, coding_runner_requirements, and handoff_back_to_mac_brain_if conditions.
- Do not repeat completed methods unless remote_brain_mission names a specific control variable that changes.
- If Mac Brain diagnosis reports execution_fidelity_gap, prioritize plan-vs-execution and actual-feature-list audits before more generic model search.
- Every recommended experiment must include:
  - skill_used: one of the available skill_registry skills.
  - harness: one of the available harness_registry harnesses.
  - hypothesis: why this experiment might improve leaderboard-relevant performance or reduce risk.
  - runner_kind: one of random_forest, tuned_random_forest, lightgbm, catboost, xgboost, tabular_mlp, tabular_resnet, star_specialist_lgbm, star_specialist_threshold_tuning, classwise_blend, clean_oof_blend, cv_stability_audit, distribution_shift_audit, overfitting_audit, regularized_blend.
  - validation_plan: how the harness will falsify or confirm the hypothesis.
  - evidence_needed: a short list of artifacts/signals the runner must produce.
  - promotion_gate: a JSON object describing when this result can replace the champion or unlock submission.

Return a single JSON object with these keys:
current_best_baseline, leaderboard_target, leaderboard_feedback, leaderboard_feedback_freshness, leaderboard_gap_audit, leaderboard_diagnosis, submit_decision_handoff, submission_decision_review, diagnosis, human_gate, next_action, recommended_experiments, risks.

Context:
{compact}
"""

    def _generate_plan(self, prompt: str, context: Dict[str, Any]) -> tuple[Dict[str, Any], bool, str]:
        if self._should_stop(context):
            plan = self._fallback_plan(context)
            plan["next_action"] = "stop"
            plan["diagnosis"] = "A human gate requested stop; no further experiment should run."
            return plan, False, json.dumps(plan, indent=2, ensure_ascii=False)

        if self.use_llm:
            try:
                raw_reply = self._call_llm(prompt)
                parsed = self._parse_json(raw_reply)
                return self._normalize_plan(parsed, context), True, raw_reply
            except Exception as exc:
                plan = self._fallback_plan(context)
                plan["risks"].append(f"LLM review fallback used: {exc}")
                return plan, False, json.dumps(plan, indent=2, ensure_ascii=False)

        plan = self._fallback_plan(context)
        return plan, False, json.dumps(plan, indent=2, ensure_ascii=False)

    def _call_llm(self, prompt: str) -> str:
        from api_handler import APIHandler, APISettings

        model = os.getenv("AUTOKAGGLE_REMOTE_BRAIN_MODEL") or os.getenv(
            "AUTOKAGGLE_PLANNER_MODEL",
            self._configured_planner_model(),
        )
        handler = APIHandler(model)
        messages = [
            {"role": "system", "content": "You are a rigorous AutoKaggle remote Brain."},
            {"role": "user", "content": prompt},
        ]
        return handler.get_output(messages, APISettings(max_completion_tokens=4096, temperature=0.2))

    def _configured_planner_model(self) -> str:
        config_path = Path(__file__).resolve().parents[1] / "config.json"
        config = self._read_json(config_path)
        return config.get("llm_models", {}).get("Planner", "mimo-v2.5-pro")

    def _fallback_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        baseline_review = context.get("baseline_review", {})
        best = self._best_completed_result(context) or baseline_review.get("best_baseline") or {}
        enhancement_review = context.get("enhancement_review") or {}
        if isinstance(enhancement_review.get("local_score"), (int, float)):
            if not isinstance(best.get("local_score"), (int, float)) or enhancement_review["local_score"] > best["local_score"]:
                best = {
                    "task_id": enhancement_review.get("task_id") or "latest_enhancement",
                    "status": enhancement_review.get("status", "validated"),
                    "metric_name": enhancement_review.get("metric_name"),
                    "local_score": enhancement_review.get("local_score"),
                    "submission_valid": bool(enhancement_review.get("submission_valid", True)),
                    "source": "enhancement_review",
                }
        champion_selection = context.get("champion_selection") or {}
        champion = champion_selection.get("champion") if isinstance(champion_selection.get("champion"), dict) else champion_selection
        if isinstance(champion.get("local_score"), (int, float)):
            best = {
                "task_id": champion.get("task_id") or champion.get("source_id") or "champion",
                "status": champion.get("status", "champion_selected"),
                "metric_name": champion.get("metric_name"),
                "local_score": champion.get("local_score"),
                "submission_valid": True,
                "source": "champion_selection",
            }
        leaderboard = context.get("leaderboard_feedback") or {}
        leaderboard_target = context.get("leaderboard_target") or {}
        leaderboard_freshness = context.get("leaderboard_feedback_freshness") or {}
        leaderboard_for_diagnosis = leaderboard if leaderboard_freshness.get("is_current", True) else {}
        submission_decision = context.get("submission_decision_review") or {}
        leaderboard_diagnosis = self._leaderboard_diagnosis(
            best,
            leaderboard_for_diagnosis,
            context.get("leaderboard_gap_audit") or {},
        )
        completed_experiments = self._completed_experiments(context)
        intervention_notes = "\n".join(
            gate.get("notes", "")
            for gate in context.get("active_human_interventions", [])
            if gate.get("notes")
        ).strip()
        task_prefix = "patched" if intervention_notes else "enhance"
        if submission_decision.get("decision") == "pause_manual_submit":
            decision_issues = submission_decision.get("issues") or []
            decision_warnings = submission_decision.get("warnings") or []
            if "stability_replan_after_pause_v1" in completed_experiments:
                task_id, runner_kind, title = self._first_uncompleted_task(
                    [
                        (
                            "post_pause_tuned_random_forest_v1",
                            "tuned_random_forest",
                            "Tuned random forest after manual gate pause",
                        ),
                        (
                            "post_pause_cv_stability_audit_v2",
                            "cv_stability_audit",
                            "Second stability audit after manual gate pause",
                        ),
                        (
                            "post_pause_regularized_blend_v1",
                            "regularized_blend",
                            "Regularized blend after manual gate pause",
                        ),
                        (
                            "post_pause_overfitting_audit_v1",
                            "overfitting_audit",
                            "Overfitting audit after manual gate pause",
                        ),
                    ],
                    completed_experiments,
                )
                recommendation = {
                    "task_id": task_id,
                    "title": title,
                    "skill_used": "validation_risk",
                    "harness": self.harness_registry.default_for_runner(runner_kind),
                    "hypothesis": (
                        "A conservative post-pause experiment can reduce validation and submission risk "
                        "before another leaderboard-facing decision."
                    ),
                    "runner_kind": runner_kind,
                    "expected_gain": "small_to_medium",
                    "risk": "low",
                    "compute_cost": "low",
                    "validation_plan": "Use stable validation evidence and compare against the paused champion before promotion.",
                    "evidence_needed": [
                        "validation_report.json",
                        "validator_result.json",
                        "feature_importance_or_feature_count",
                        "comparison_to_paused_champion",
                    ],
                    "promotion_gate": {
                        "min_local_score_delta": 0.002,
                        "validator_must_pass": True,
                        "manual_submit_allowed": False,
                    },
                    "coding_agent_task": (
                        "Do not perform a Kaggle/manual submission. Run the selected pause-safe "
                        f"{runner_kind} experiment with conservative feature handling, repeated or 5-fold "
                        "validation where applicable, OOF/stability evidence, and a validated submission "
                        "artifact. Compare against the paused champion and the latest stability re-plan result."
                    ),
                }
            else:
                recommendation = {
                    "task_id": "stability_replan_after_pause_v1",
                    "title": "Stability re-plan after manual gate pause",
                    "skill_used": "validation_risk",
                    "harness": "stratified_cv_stability_harness",
                    "hypothesis": (
                        "The paused submission decision may reflect unstable validation or leaderboard mismatch; "
                        "a stability audit should test whether the champion is reliable."
                    ),
                    "runner_kind": "cv_stability_audit",
                    "expected_gain": "risk_reduction",
                    "risk": "low",
                    "compute_cost": "low",
                    "validation_plan": "Run repeated or 5-fold validation, seed/fold variance, and OOF stability checks.",
                    "evidence_needed": [
                        "cv_stability_audit.json",
                        "validation_report.json",
                        "validator_result.json",
                        "seed_std",
                        "fold_std",
                        "public_within_seed_ci",
                    ],
                    "promotion_gate": {
                        "max_risk_level": "low",
                        "public_within_seed_ci": True,
                        "manual_submit_allowed": False,
                    },
                    "coding_agent_task": (
                        "Do not perform a Kaggle/manual submission. Address the paused manual gate by "
                        "running a stability-focused LightGBM/CatBoost comparison against the current "
                        "champion and recommended candidate. Use repeated CV where feasible, report seed "
                        "and fold variance, compare OOF stability, and produce a validated submission only "
                        "as an artifact for later review. Explicitly discuss these pause reasons: "
                        f"{'; '.join(str(item) for item in decision_issues + decision_warnings)}"
                    ),
                }
            diagnosis = (
                "Manual gate is paused. Remote Brain should re-open the loop with a runnable "
                "stability-focused experiment before another submission decision."
            )
            risks = list(decision_issues) + list(decision_warnings) + leaderboard_diagnosis["risks"]
        elif leaderboard_diagnosis["risk_level"] == "high":
            task_id, runner_kind, title = self._first_uncompleted_task(
                [
                    (
                        f"{task_prefix}_leaderboard_gap_audit_v1",
                        "distribution_shift_audit",
                        "Audit local CV versus public leaderboard gap",
                    ),
                    (
                        "post_feedback_overfitting_audit_v1",
                        "overfitting_audit",
                        "Audit post-feedback overfitting risk",
                    ),
                    (
                        "post_feedback_cv_stability_audit_v1",
                        "cv_stability_audit",
                        "Audit post-feedback CV stability",
                    ),
                ],
                completed_experiments,
            )
            recommendation = {
                "task_id": task_id,
                "title": title,
                "skill_used": "validation_risk",
                "harness": self.harness_registry.default_for_runner(runner_kind),
                "hypothesis": (
                    "Public leaderboard feedback may be inconsistent with local validation; "
                    "audit drift and overfitting before chasing additional local score."
                ),
                "runner_kind": runner_kind,
                "expected_gain": "risk_reduction",
                "risk": "low",
                "compute_cost": "low",
                "validation_plan": "Compare local CV, public feedback, train/test drift, and OOF stability.",
                "evidence_needed": [
                    "distribution_shift_audit.json",
                    "validation_report.json",
                    "validator_result.json",
                    "top_drift_features",
                ],
                "promotion_gate": {
                    "validator_must_pass": True,
                    "max_drift_score_reviewed": True,
                    "manual_submit_allowed": False,
                },
                "coding_agent_task": (
                    "Compare the current champion local CV score against public leaderboard feedback. "
                    "Run repeated CV, inspect OOF stability, check train/test feature drift, validate "
                    "split strategy, and recommend whether to trust the champion or select a more stable "
                    "alternative before further leaderboard submissions."
                ),
            }
            diagnosis = leaderboard_diagnosis["message"]
            risks = leaderboard_diagnosis["risks"]
        else:
            advanced_candidates = self._advanced_tabular_candidates(task_prefix)
            task_id, runner_kind, title = self._first_uncompleted_task(
                advanced_candidates
                + [
                    (
                        f"{task_prefix}_lightgbm_5fold_v1",
                        "lightgbm",
                        "Train a stronger tabular model with 5-fold CV",
                    ),
                    (
                        "post_feedback_catboost_5fold_v1",
                        "catboost",
                        "Train a post-feedback CatBoost model with 5-fold CV",
                    ),
                    (
                        "post_feedback_xgboost_5fold_v1",
                        "xgboost",
                        "Train a post-feedback XGBoost model with 5-fold CV",
                    ),
                    (
                        "post_feedback_regularized_blend_v1",
                        "regularized_blend",
                        "Build a post-feedback regularized blend",
                    ),
                    (
                        "post_feedback_tuned_random_forest_v1",
                        "tuned_random_forest",
                        "Train a post-feedback tuned random forest",
                    ),
                ],
                completed_experiments,
            )
            recommendation = {
                "task_id": task_id,
                "title": title,
                "skill_used": self._default_skill_for_runner(runner_kind),
                "harness": self.harness_registry.default_for_runner(runner_kind),
                "hypothesis": self._default_hypothesis_for_runner(runner_kind),
                "runner_kind": runner_kind,
                "expected_gain": "medium",
                "risk": "low",
                "compute_cost": "low",
                "validation_plan": self._default_validation_plan_for_runner(runner_kind),
                "evidence_needed": self._default_evidence_for_runner(runner_kind),
                "promotion_gate": {
                    "min_local_score": 0.962 if runner_kind == "tabular_mlp" else 0.964 if runner_kind in {"star_specialist_lgbm", "star_specialist_threshold_tuning"} else 0.965 if runner_kind == "clean_oof_blend" else None,
                    "validator_must_pass": True,
                    "manual_submit_allowed": False,
                },
                "coding_agent_task": self._default_coding_task_for_runner(runner_kind),
            }
            recommendation["promotion_gate"] = {
                key: value for key, value in recommendation["promotion_gate"].items() if value is not None
            }
            diagnosis = (
                "Baseline loop is valid. The next useful step is a non-repeating model-family "
                "or ensemble experiment with OOF evidence and validator-checked submission."
            )
            risks = leaderboard_diagnosis["risks"]
        if intervention_notes:
            recommendation["human_patch_notes"] = intervention_notes
        recommendation = self._with_runner_specific_evidence(recommendation)
        if leaderboard and not leaderboard_freshness.get("is_current", True):
            freshness_issues = leaderboard_freshness.get("issues") or []
            risks = list(risks) + [
                "leaderboard feedback is not bound to the current packaged submission: "
                + ("; ".join(str(item) for item in freshness_issues) if freshness_issues else leaderboard_freshness.get("status", "stale"))
            ]
        recommendations = self._apply_strategy_guardrails([recommendation], context)
        return {
            "current_best_baseline": best,
            "leaderboard_target": leaderboard_target,
            "leaderboard_feedback": leaderboard,
            "leaderboard_feedback_freshness": leaderboard_freshness,
            "leaderboard_gap_audit": context.get("leaderboard_gap_audit") or {},
            "leaderboard_diagnosis": leaderboard_diagnosis,
            "skill_registry": context.get("skill_registry") or {},
            "harness_registry": context.get("harness_registry") or {},
            "submit_decision_handoff": context.get("submit_decision_handoff") or {},
            "submission_decision_review": submission_decision,
            "diagnosis": diagnosis,
            "human_gate": {
                "decision": "patch_prompt" if intervention_notes else "continue",
                "notes": intervention_notes,
            },
            "next_action": "generate_enhancement_tasks",
            "recommended_experiments": recommendations,
            "risks": risks,
        }

    def _advanced_tabular_candidates(self, task_prefix: str) -> List[tuple[str, str, str]]:
        has_per_class = (self.competition_dir / "experiments" / "per_class_oof_audit_v1" / "per_class_oof_report.json").exists()
        has_diversity = (self.competition_dir / "experiments" / "oof_diversity_matrix_v1" / "oof_diversity_report.json").exists()
        is_playground = self.competition_dir.name == "playground-series-s6e6"
        if not (has_per_class or has_diversity or is_playground):
            return []
        return [
            (
                "star_specialist_threshold_tuning_v1",
                "star_specialist_threshold_tuning",
                "Tune STAR specialist threshold against OOF accuracy and recall",
            ),
            (
                "star_specialist_threshold_tuning_v2",
                "star_specialist_threshold_tuning",
                "Tune STAR specialist threshold with promotion-gate score floor",
            ),
            (
                "star_specialist_lgbm_v1",
                "star_specialist_lgbm",
                "Train a STAR class specialist correction model",
            ),
            (
                "tabular_mlp_oof_v1",
                "tabular_mlp",
                "Train a tabular MLP OOF diversity model",
            ),
            (
                "xgboost_verified_features_v2",
                "xgboost",
                "Train XGBoost with verified feature evidence",
            ),
            (
                "clean_diversity_blend_v1",
                "clean_oof_blend",
                "Build a clean OOF blend from valid candidates",
            ),
        ]

    def _normalize_plan(
        self,
        plan: Dict[str, Any],
        context: Dict[str, Any],
        *,
        rename_completed: bool = True,
    ) -> Dict[str, Any]:
        fallback = self._fallback_plan(context)
        normalized = dict(fallback)
        for key in [
            "current_best_baseline",
            "leaderboard_target",
            "leaderboard_feedback",
            "leaderboard_feedback_freshness",
            "leaderboard_gap_audit",
            "leaderboard_diagnosis",
            "skill_registry",
            "harness_registry",
            "submit_decision_handoff",
            "submission_decision_review",
            "diagnosis",
            "human_gate",
            "next_action",
            "recommended_experiments",
            "risks",
        ]:
            if key in plan:
                normalized[key] = plan[key]
        if not isinstance(normalized.get("current_best_baseline"), dict):
            normalized["current_best_baseline"] = fallback["current_best_baseline"]
        if not isinstance(normalized.get("leaderboard_target"), dict):
            normalized["leaderboard_target"] = fallback.get("leaderboard_target", {})
        normalized["leaderboard_target"] = self._normalize_leaderboard_target(
            normalized["leaderboard_target"],
            normalized.get("current_best_baseline") if isinstance(normalized.get("current_best_baseline"), dict) else {},
        )
        if not isinstance(normalized.get("leaderboard_feedback"), dict):
            normalized["leaderboard_feedback"] = fallback["leaderboard_feedback"]
        if not isinstance(normalized.get("leaderboard_feedback_freshness"), dict):
            normalized["leaderboard_feedback_freshness"] = fallback["leaderboard_feedback_freshness"]
        if not isinstance(normalized.get("leaderboard_gap_audit"), dict):
            normalized["leaderboard_gap_audit"] = fallback["leaderboard_gap_audit"]
        if not isinstance(normalized.get("leaderboard_diagnosis"), dict):
            normalized["leaderboard_diagnosis"] = fallback["leaderboard_diagnosis"]
        if not isinstance(normalized.get("skill_registry"), dict):
            normalized["skill_registry"] = fallback.get("skill_registry", {})
        if not isinstance(normalized.get("harness_registry"), dict):
            normalized["harness_registry"] = fallback.get("harness_registry", {})
        if not isinstance(normalized.get("submit_decision_handoff"), dict):
            normalized["submit_decision_handoff"] = fallback["submit_decision_handoff"]
        if not isinstance(normalized.get("submission_decision_review"), dict):
            normalized["submission_decision_review"] = fallback["submission_decision_review"]
        context_submission_decision = context.get("submission_decision_review")
        if (
            isinstance(context_submission_decision, dict)
            and context_submission_decision.get("decision")
            and not normalized["submission_decision_review"].get("decision")
        ):
            normalized["submission_decision_review"] = context_submission_decision
        if not isinstance(normalized.get("human_gate"), dict):
            normalized["human_gate"] = {
                "decision": str(normalized.get("human_gate") or "continue"),
                "notes": "",
            }
        if not isinstance(normalized.get("recommended_experiments"), list):
            normalized["recommended_experiments"] = fallback["recommended_experiments"]
        normalized["recommended_experiments"] = [
            self._normalize_experiment(item, index)
            for index, item in enumerate(normalized["recommended_experiments"], start=1)
            if isinstance(item, dict)
        ]
        if rename_completed:
            normalized["recommended_experiments"] = self._dedupe_completed_recommendations(
                normalized["recommended_experiments"],
                context,
            )
            normalized["recommended_experiments"] = self._apply_strategy_guardrails(
                normalized["recommended_experiments"],
                context,
            )
        if not isinstance(normalized.get("risks"), list):
            normalized["risks"] = [str(normalized["risks"])]
        return normalized

    def _normalize_leaderboard_target(
        self,
        leaderboard_target: Dict[str, Any],
        current_best: Dict[str, Any],
    ) -> Dict[str, Any]:
        target = dict(leaderboard_target or {})
        current_score = current_best.get("local_score")
        metric_name = str(current_best.get("metric_name") or "").lower()
        if not isinstance(current_score, (int, float)):
            return target
        silver_score = target.get("estimated_silver_score")
        if not isinstance(silver_score, (int, float)):
            silver_score = target.get("silver_score")
        top_score = target.get("top_score")
        lower_is_better = metric_name in {"rmse", "rmsle", "mae", "mse", "log_loss"}
        if isinstance(silver_score, (int, float)):
            target["silver_score"] = silver_score
            target["gap_to_silver"] = (
                float(current_score - silver_score)
                if lower_is_better
                else float(silver_score - current_score)
            )
        if isinstance(top_score, (int, float)):
            target["gap_to_top"] = (
                float(current_score - top_score)
                if lower_is_better
                else float(top_score - current_score)
            )
        return target

    def _normalize_experiment(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        task_id = (
            item.get("task_id")
            or item.get("experiment_id")
            or self._slug(item.get("title") or item.get("description") or "", index)
        )
        normalized = {
            "task_id": task_id,
            "title": item.get("title") or item.get("description") or task_id,
            "skill_used": self._normalize_skill(item),
            "harness": self._normalize_harness(item),
            "hypothesis": item.get("hypothesis") or item.get("rationale") or self._default_hypothesis(item),
            "runner_kind": self._normalize_runner_kind(item, task_id),
            "expected_gain": item.get("expected_gain") or item.get("expected_outcome") or "unknown",
            "risk": item.get("risk") or "unknown",
            "compute_cost": item.get("compute_cost") or "unknown",
            "validation_plan": item.get("validation_plan") or "Use the selected harness to produce validation, risk, and submission evidence.",
            "evidence_needed": self._normalize_string_list(item.get("evidence_needed")),
            "promotion_gate": item.get("promotion_gate") if isinstance(item.get("promotion_gate"), dict) else {},
            "coding_agent_task": item.get("coding_agent_task")
            or item.get("coding_prompt_append")
            or item.get("description")
            or "",
        }
        if not str(normalized["coding_agent_task"]).strip():
            normalized["coding_agent_task"] = self._default_coding_agent_task(normalized)
        if not normalized["harness"]:
            normalized["harness"] = self.harness_registry.default_for_runner(normalized["runner_kind"])
        return self._with_runner_specific_evidence(normalized)

    def _default_coding_agent_task(self, experiment: Dict[str, Any]) -> str:
        evidence = ", ".join(str(item) for item in experiment.get("evidence_needed", [])[:5])
        gate = json.dumps(experiment.get("promotion_gate") or {}, ensure_ascii=False)
        return (
            f"Implement `{experiment.get('task_id')}` exactly as a narrow Remote Brain task. "
            f"Title: {experiment.get('title')}. "
            f"Hypothesis: {experiment.get('hypothesis')}. "
            f"Runner kind: {experiment.get('runner_kind')}; harness: {experiment.get('harness')}. "
            f"Validation plan: {experiment.get('validation_plan')}. "
            f"Required evidence: {evidence or 'validation_report.json, validator_result.json, run.log'}. "
            f"Promotion gate: {gate}. "
            "Do not silently fall back to a generic template; write plan_vs_execution_diff.json when the task requires custom behavior."
        )

    def _dedupe_completed_recommendations(
        self,
        experiments: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        completed = self._completed_experiments(context)
        seen: set[str] = set()
        output = []
        for index, experiment in enumerate(experiments, start=1):
            task_id = str(experiment.get("task_id") or f"brain_recommendation_{index}")
            if task_id in completed or task_id in seen:
                original_task_id = task_id
                seed = " ".join(
                    str(experiment.get(key, ""))
                    for key in ["runner_kind", "title", "hypothesis", "validation_plan"]
                )
                task_id = self._slug(seed, index) or f"brain_recommendation_{index}"
                if task_id in completed or task_id in seen:
                    runner = str(experiment.get("runner_kind") or "experiment").replace("-", "_")
                    task_id = f"{runner}_brain_recommendation_{index}"
                suffix = 2
                base = task_id
                while task_id in completed or task_id in seen:
                    task_id = f"{base}_{suffix}"
                    suffix += 1
                experiment = dict(experiment)
                experiment["original_task_id"] = original_task_id
                experiment["task_id"] = task_id
            seen.add(task_id)
            output.append(experiment)
        return output

    def _apply_strategy_guardrails(
        self,
        experiments: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidate_pool = context.get("candidate_pool") if isinstance(context.get("candidate_pool"), dict) else {}
        ensemble_candidates = [
            item for item in candidate_pool.get("ensemble_candidates", [])
            if isinstance(item, dict) and item.get("has_oof") and item.get("validator_ok")
        ]
        completed = self._completed_experiments(context)
        if len(ensemble_candidates) >= 2 and not any("regularized_blend" in item for item in completed):
            blend = self._with_runner_specific_evidence(
                {
                    "task_id": self._next_available_task_id(
                        "agent_loop_regularized_blend_v1",
                        completed,
                        {str(item.get("task_id")) for item in experiments if isinstance(item, dict)},
                    ),
                    "title": "Agent Loop regularized OOF blend",
                    "skill_used": "ensemble_strategy",
                    "harness": "regularized_oof_blend_harness",
                    "hypothesis": (
                        "The candidate pool already has multiple validator-passing OOF models. "
                        "A constrained regularized blend is now more goal-directed than another "
                        "single-model baseline because it can test whether model diversity reduces "
                        "the remaining leaderboard gap."
                    ),
                    "runner_kind": "regularized_blend",
                    "expected_gain": "small_to_medium",
                    "risk": "medium",
                    "compute_cost": "medium",
                    "validation_plan": (
                        "Blend existing OOF candidates with constrained weights, compare against the "
                        "current champion, report seed/fold stability and max model correlation, then "
                        "validate the produced submission."
                    ),
                    "evidence_needed": [
                        "regularized_blend_report.json",
                        "oof_predictions.csv",
                        "validation_report.json",
                        "validator_result.json",
                        "submission.csv",
                        "max_model_correlation",
                    ],
                    "promotion_gate": {
                        "min_delta_vs_best_single": 0.0005,
                        "max_train_valid_gap": 0.015,
                        "max_model_correlation": "<= 0.995",
                        "validator_must_pass": True,
                        "manual_submit_allowed": False,
                    },
                    "coding_agent_task": (
                        "Use the validated OOF candidate pool to build a constrained regularized "
                        "blend. Do not submit to Kaggle. Produce regularized_blend_report.json, "
                        "oof_predictions.csv, validation_report.json, validator_result.json, and "
                        "submission.csv. Compare against the current champion and explicitly report "
                        "whether the blend should be promoted."
                    ),
                    "strategy_guardrail": "candidate_pool_has_multiple_oof_models",
                    "input_candidates": [item.get("task_id") for item in ensemble_candidates],
                }
            )
            remaining = [
                item for item in experiments
                if isinstance(item, dict) and item.get("runner_kind") != "regularized_blend"
            ]
            return [blend] + remaining
        return experiments

    def _next_available_task_id(
        self,
        base: str,
        completed: set[str],
        reserved: set[str],
    ) -> str:
        task_id = base
        suffix = 2
        while task_id in completed or task_id in reserved:
            task_id = f"{base}_{suffix}"
            suffix += 1
        return task_id

    def _normalize_skill(self, item: Dict[str, Any]) -> str:
        available = {skill.name for skill in self.skill_registry.list_skills()}
        explicit = str(item.get("skill_used") or item.get("skill") or "").strip()
        if explicit in available:
            return explicit
        text = " ".join(str(item.get(key, "")) for key in ["task_id", "title", "runner_kind", "description"]).lower()
        if "leaderboard" in text:
            return "leaderboard_target"
        if "specialist" in text or "one_vs_rest" in text or "one-vs-rest" in text:
            return "class_specialist"
        if "mlp" in text or "resnet" in text or "neural" in text or "tabular_nn" in text:
            return "tabular_nn"
        if "clean_oof" in text or "clean ensemble" in text:
            return "clean_ensemble"
        if "blend" in text or "ensemble" in text:
            return "ensemble_strategy"
        if "audit" in text or "stability" in text or "drift" in text or "overfit" in text:
            return "validation_risk"
        return "tabular_optimization"

    def _normalize_harness(self, item: Dict[str, Any]) -> str:
        available = {harness.name for harness in self.harness_registry.list_harnesses()}
        explicit = str(item.get("harness") or item.get("harness_name") or "").strip()
        return explicit if explicit in available else ""

    @staticmethod
    def _default_hypothesis(item: Dict[str, Any]) -> str:
        title = item.get("title") or item.get("task_id") or "This experiment"
        return f"{title} may improve leaderboard-relevant performance or reduce validation risk."

    def _normalize_runner_kind(self, item: Dict[str, Any], task_id: str) -> str:
        allowed = {
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
        }
        explicit = str(item.get("runner_kind") or item.get("runner") or "").strip().lower()
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
            "star_specialist_threshold": "star_specialist_threshold_tuning",
            "clean_blend": "clean_oof_blend",
        }
        explicit = aliases.get(explicit, explicit)
        if explicit in allowed:
            return explicit
        text = " ".join(
            str(item.get(key, ""))
            for key in ["task_id", "experiment_id", "title", "description", "coding_agent_task", "coding_prompt_append"]
        ).lower()
        text = f"{task_id} {text}"
        if "distribution_shift" in text or "distribution shift" in text or "drift audit" in text:
            return "distribution_shift_audit"
        if "overfitting" in text or "overfit" in text:
            return "overfitting_audit"
        if "regularized_blend" in text or "regularized blend" in text or "blend_with_regularization" in text or ("blend" in text and "regularization" in text):
            return "regularized_blend"
        if "cv_stability" in text or "stability audit" in text or "stability_audit" in text:
            return "cv_stability_audit"
        if "lightgbm" in text or "lgbm" in text:
            return "lightgbm"
        if "catboost" in text:
            return "catboost"
        if "xgboost" in text or "xgb" in text:
            return "xgboost"
        if "threshold" in text and ("star_specialist" in text or "star specialist" in text or "star-vs-rest" in text or "one-vs-rest" in text):
            return "star_specialist_threshold_tuning"
        if "star_specialist" in text or "star specialist" in text or "star-vs-rest" in text or "one-vs-rest" in text:
            return "star_specialist_lgbm"
        if "clean_oof_blend" in text or "clean oof blend" in text or "clean blend" in text:
            return "clean_oof_blend"
        if "classwise_blend" in text or "class-wise blend" in text:
            return "classwise_blend"
        if "tabular_resnet" in text or "resnet" in text:
            return "tabular_resnet"
        if "tabular_mlp" in text or "tabular nn" in text or "mlp" in text or "neural" in text:
            return "tabular_mlp"
        if "gridsearch" in text or "grid search" in text or "tuning" in text or "tune" in text:
            return "tuned_random_forest"
        return "random_forest"

    def _default_skill_for_runner(self, runner_kind: str) -> str:
        if runner_kind in {"tabular_mlp", "tabular_resnet"}:
            return "tabular_nn"
        if runner_kind in {"star_specialist_lgbm", "star_specialist_threshold_tuning"}:
            return "class_specialist"
        if runner_kind in {"clean_oof_blend", "classwise_blend"}:
            return "clean_ensemble"
        if runner_kind in {"regularized_blend"}:
            return "ensemble_strategy"
        return "tabular_optimization"

    def _default_hypothesis_for_runner(self, runner_kind: str) -> str:
        if runner_kind == "star_specialist_lgbm":
            return "A weak-class specialist may improve STAR recall without sacrificing overall validation accuracy."
        if runner_kind == "star_specialist_threshold_tuning":
            return "A weak-class specialist may recover STAR recall if its probability threshold is tuned against an OOF accuracy floor instead of fixed at 0.50."
        if runner_kind in {"tabular_mlp", "tabular_resnet"}:
            return "A tabular neural model may add non-GBDT OOF diversity and capture smooth continuous interactions."
        if runner_kind in {"clean_oof_blend", "classwise_blend"}:
            return "A clean blend of valid OOF candidates may improve score while avoiding invalid or duplicate candidates."
        return "A cross-validated tabular model may capture nonlinear feature interactions missed by the current baseline."

    def _default_validation_plan_for_runner(self, runner_kind: str) -> str:
        if runner_kind == "star_specialist_lgbm":
            return "Use 5-fold OOF validation, report target-class recall, coverage, and overall accuracy."
        if runner_kind == "star_specialist_threshold_tuning":
            return "Use 5-fold OOF validation and search specialist probability thresholds; report the threshold frontier, selected threshold, target-class recall, coverage, and overall score drop."
        if runner_kind in {"tabular_mlp", "tabular_resnet"}:
            return "Use 5-fold OOF validation, train-valid gap, backend report, and OOF diversity checks."
        if runner_kind in {"clean_oof_blend", "classwise_blend"}:
            return "Filter invalid OOF candidates, blend valid predictions, and compare OOF score and per-class metrics."
        return "Use 5-fold stratified OOF validation, feature importance, and validator-checked submission."

    def _default_evidence_for_runner(self, runner_kind: str) -> List[str]:
        if runner_kind in {"star_specialist_lgbm", "star_specialist_threshold_tuning"}:
            return ["validation_report.json", "validator_result.json", "specialist_report.json", "per_class_oof_report.json", "oof_predictions.csv"]
        if runner_kind in {"tabular_mlp", "tabular_resnet"}:
            return ["validation_report.json", "validator_result.json", "nn_training_report.json", "model_config.json", "oof_predictions.csv"]
        if runner_kind in {"clean_oof_blend", "classwise_blend"}:
            return ["validation_report.json", "validator_result.json", "clean_blend_report.json", "skipped_candidates.json", "oof_diversity_report.json"]
        return ["validation_report.json", "validator_result.json", "cv_scores", "feature_count"]

    def _default_coding_task_for_runner(self, runner_kind: str) -> str:
        if runner_kind == "star_specialist_lgbm":
            return "Train a STAR-vs-rest specialist on top of a stable multiclass base model. Produce specialist_report.json, per_class_oof_report.json, OOF predictions, validation_report.json, and a validated submission."
        if runner_kind == "star_specialist_threshold_tuning":
            return "Train a STAR-vs-rest specialist on top of a stable multiclass base model, then tune the specialist probability threshold on OOF predictions against an overall-score floor. Produce specialist_report.json with threshold_frontier, per_class_oof_report.json, OOF predictions, validation_report.json, and a validated submission."
        if runner_kind in {"tabular_mlp", "tabular_resnet"}:
            return "Train a tabular neural-network style OOF model using available tabular features. Produce nn_training_report.json, model_config.json, OOF predictions, validation_report.json, and a validated submission."
        if runner_kind in {"clean_oof_blend", "classwise_blend"}:
            return "Filter invalid OOF candidates, blend valid candidate predictions, write clean_blend_report.json, skipped_candidates.json, OOF predictions, validation_report.json, and a validated submission."
        return "Use the current best baseline as reference. Implement a cross-validated tabular model with OOF metrics, validated submission, and full ledger artifacts."

    def _normalize_string_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return ["validation_report.json", "validator_result.json", "submission.csv"]

    def _slug(self, text: str, index: int) -> str:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())[:8]
        if not words:
            return f"experiment_{index}"
        return "_".join(words)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end >= start:
            candidate = candidate[start : end + 1]
        return json.loads(candidate)

    def _render_markdown(self, plan: Dict[str, Any]) -> str:
        experiments = plan.get("recommended_experiments", [])
        lines = [
            "# Remote Brain Experiment Plan",
            "",
            f"Next action: `{plan.get('next_action', 'unknown')}`",
            "",
            "## Current Best Baseline",
            "",
            "```json",
            json.dumps(plan.get("current_best_baseline", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Diagnosis",
            "",
            str(plan.get("diagnosis", "")),
            "",
            "## Leaderboard Target",
            "",
            "```json",
            json.dumps(plan.get("leaderboard_target", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Leaderboard Feedback",
            "",
            "```json",
            json.dumps(plan.get("leaderboard_feedback", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Leaderboard Gap Audit",
            "",
            "```json",
            json.dumps(plan.get("leaderboard_gap_audit", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Leaderboard Diagnosis",
            "",
            "```json",
            json.dumps(plan.get("leaderboard_diagnosis", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Submit Decision Handoff",
            "",
            "```json",
            json.dumps(plan.get("submit_decision_handoff", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Submission Decision Review",
            "",
            "```json",
            json.dumps(plan.get("submission_decision_review", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Human Gate",
            "",
            "```json",
            json.dumps(plan.get("human_gate", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Recommended Experiments",
        ]
        for item in experiments:
            lines.extend(
                [
                    "",
                    f"### {item.get('task_id', 'experiment')}",
                    "",
                    f"- Title: {item.get('title', '')}",
                    f"- Skill: {item.get('skill_used', '')}",
                    f"- Harness: {item.get('harness', '')}",
                    f"- Hypothesis: {item.get('hypothesis', '')}",
                    f"- Runner kind: {item.get('runner_kind', '')}",
                    f"- Expected gain: {item.get('expected_gain', '')}",
                    f"- Risk: {item.get('risk', '')}",
                    f"- Compute cost: {item.get('compute_cost', '')}",
                    f"- Validation plan: {item.get('validation_plan', '')}",
                    f"- Evidence needed: {', '.join(item.get('evidence_needed', []))}",
                    f"- Promotion gate: `{json.dumps(item.get('promotion_gate', {}), ensure_ascii=False)}`",
                    "",
                    str(item.get("coding_agent_task", "")),
                ]
            )
        if plan.get("risks"):
            lines.extend(["", "## Risks", ""])
            lines.extend(f"- {risk}" for risk in plan["risks"])
        return "\n".join(lines).rstrip() + "\n"

    def _best_score(self, plan: Dict[str, Any]) -> Optional[float]:
        score = plan.get("current_best_baseline", {}).get("local_score")
        return score if isinstance(score, (int, float)) else None

    def _completed_experiments(self, context: Dict[str, Any]) -> set[str]:
        completed = set()
        for item in context.get("baseline_reports", []):
            if not isinstance(item, dict):
                continue
            report = item.get("validation_report") or {}
            if report.get("status") == "completed":
                completed.add(str(item.get("experiment") or report.get("experiment") or ""))
        return {item for item in completed if item}

    def _best_completed_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        best: Dict[str, Any] = {}
        for item in context.get("baseline_reports", []):
            if not isinstance(item, dict):
                continue
            report = item.get("validation_report") or {}
            validator = item.get("validator_result") or {}
            score = report.get("local_score")
            if report.get("status") != "completed" or not isinstance(score, (int, float)):
                continue
            if validator and validator.get("ok") is False:
                continue
            if not isinstance(best.get("local_score"), (int, float)) or score > best["local_score"]:
                best = {
                    "task_id": str(item.get("experiment") or report.get("experiment") or "experiment"),
                    "status": report.get("status"),
                    "metric_name": report.get("metric_name"),
                    "local_score": score,
                    "runner_kind": report.get("runner_kind"),
                    "submission_valid": validator.get("ok", True),
                    "source": "validation_report",
                }
        return best

    @staticmethod
    def _first_uncompleted_task(
        candidates: List[tuple[str, str, str]],
        completed_experiments: set[str],
    ) -> tuple[str, str, str]:
        for task_id, runner_kind, title in candidates:
            if task_id not in completed_experiments:
                return task_id, runner_kind, title
        fallback_task_id, runner_kind, title = candidates[-1]
        return f"{fallback_task_id}_next", runner_kind, f"{title} next"

    @staticmethod
    def _with_runner_specific_evidence(recommendation: Dict[str, Any]) -> Dict[str, Any]:
        runner_kind = recommendation.get("runner_kind")
        if runner_kind != "regularized_blend":
            return recommendation
        upgraded = dict(recommendation)
        upgraded["evidence_needed"] = [
            "validation_report.json",
            "validator_result.json",
            "submission.csv",
            "regularized_blend_report.json",
            "oof_predictions.csv",
            "seed_mean",
            "seed_std",
            "fold_std",
            "train_valid_gap",
            "max_model_correlation",
        ]
        gate = dict(upgraded.get("promotion_gate") or {})
        gate.pop("min_local_score_delta", None)
        gate.update(
            {
                "validator_must_pass": True,
                "manual_submit_allowed": False,
                "max_local_score_drop": 0.03,
                "seed_std": "<= 0.010",
                "fold_std": "<= 0.030",
                "train_valid_gap": "<= 0.040",
                "max_model_correlation": "<= 0.995",
            }
        )
        upgraded["promotion_gate"] = gate
        return upgraded

    def _leaderboard_diagnosis(
        self,
        best: Dict[str, Any],
        leaderboard: Dict[str, Any],
        gap_audit: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gap_audit = gap_audit or {}
        if gap_audit.get("status") == "completed":
            score_gap = gap_audit.get("score_gap") or {}
            risk_level = gap_audit.get("risk_level") or "unknown"
            issues = gap_audit.get("issues") or []
            return {
                "available": True,
                "risk_level": risk_level,
                "gap": score_gap.get("gap"),
                "message": gap_audit.get("recommendation")
                or gap_audit.get("next_action")
                or "Use leaderboard gap audit to drive the next experiment.",
                "risks": issues,
            }
        local_score = best.get("local_score")
        public_score = leaderboard.get("public_score")
        metric_name = best.get("metric_name") or leaderboard.get("metric_name")
        if not isinstance(local_score, (int, float)) or not isinstance(public_score, (int, float)):
            return {
                "available": bool(leaderboard),
                "risk_level": "unknown" if leaderboard else "none",
                "gap": None,
                "message": "No comparable public leaderboard score is available yet.",
                "risks": [],
            }
        lower_is_better = metric_name in {"rmse", "rmsle", "mae", "log_loss"}
        gap = public_score - local_score
        worse = public_score > local_score if lower_is_better else public_score < local_score
        magnitude = abs(gap)
        high_gap = magnitude >= max(0.01, abs(local_score) * 0.03)
        if worse and high_gap:
            return {
                "available": True,
                "risk_level": "high",
                "gap": gap,
                "message": (
                    "Public leaderboard feedback is materially worse than local validation. "
                    "Prioritize validation reliability, split strategy, drift checks, and stable ensembles "
                    "before chasing higher local CV."
                ),
                "risks": [
                    "public_score_worse_than_local_cv",
                    "possible_cv_overfit_or_distribution_shift",
                ],
            }
        return {
            "available": True,
            "risk_level": "low",
            "gap": gap,
            "message": "Public leaderboard feedback is broadly consistent with local validation.",
            "risks": [],
        }

    def _should_stop(self, context: Dict[str, Any]) -> bool:
        return any(gate.get("decision") == "stop" for gate in context.get("human_gates", []))

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_tail(self, path: Path, max_chars: int = 2000) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[-max_chars:]
