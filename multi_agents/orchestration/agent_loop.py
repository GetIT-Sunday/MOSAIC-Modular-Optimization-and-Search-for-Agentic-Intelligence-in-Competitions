from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .champion_selector import ExperimentChampionSelector
from .enhancement_runner import EnhancementRunner
from .experiment_queue import ExperimentQueueBuilder
from .leaderboard_target import LeaderboardTargetAgent
from .memory import CompetitionMemory, ExperimentRecord
from .remote_brain import RemoteBrainReviewer
from .run_ledger import RunLedger


LOWER_IS_BETTER = {"rmse", "rmsle", "mae", "mse", "log_loss"}


@dataclass
class GoalStopConditions:
    public_score_reaches_target: bool = True
    no_improvement_rounds: int = 3
    validator_failure_rounds: int = 2
    mac_handoff_no_improvement_rounds: int = 5


@dataclass
class GoalSpec:
    target: str = "silver"
    metric_name: str = "unknown"
    target_score: Optional[float] = None
    current_champion: Optional[str] = None
    max_iterations: int = 1
    auto_submit: bool = False
    human_submit_required: bool = True
    stop_conditions: GoalStopConditions = field(default_factory=GoalStopConditions)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stop_conditions"] = asdict(self.stop_conditions)
        return data


@dataclass(frozen=True)
class AgentLoopResult:
    status: str
    decision: str
    iterations_completed: int
    goal_path: Path
    state_path: Path
    summary_path: Path
    champion_state_path: Path
    candidate_pool_path: Path
    mac_brain_handoff_path: Path
    ledger_html_path: Path


class AgentLoopController:
    """Goal-oriented controller for repeated Brain -> Runner -> Validator loops."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
        use_llm: bool = True,
        refresh_leaderboard: bool = True,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.use_llm = use_llm
        self.refresh_leaderboard = refresh_leaderboard
        self.ledger = RunLedger(self.competition_dir)

    def run(
        self,
        *,
        target: str = "silver",
        max_iterations: int = 1,
        target_score: Optional[float] = None,
    ) -> AgentLoopResult:
        goal = self._load_or_create_goal(target=target, max_iterations=max_iterations, target_score=target_score)
        state = self._load_state()
        rounds: List[Dict[str, Any]] = []
        final_decision = self._preflight_decision(goal, state)
        if final_decision in {"stop_target_reached", "prepare_manual_submit", "stop_blocked", "escalate_to_mac_brain"}:
            return self._finish(goal, state, rounds, final_decision)

        for iteration in range(1, max_iterations + 1):
            if self.refresh_leaderboard:
                self._refresh_leaderboard_target()
                goal = self._load_or_create_goal(target=target, max_iterations=max_iterations, target_score=target_score)

            review = RemoteBrainReviewer(self.competition_dir, memory=self.memory, use_llm=self.use_llm).review()
            queue_result = ExperimentQueueBuilder(self.competition_dir, memory=self.memory).build()
            queue = self._read_json(queue_result.queue_path)
            next_runnable = queue.get("next_runnable") or {}
            if not next_runnable:
                final_decision = "stop_blocked"
                rounds.append(
                    {
                        "iteration": iteration,
                        "decision": final_decision,
                        "reason": "No next_runnable item in experiment_queue.json.",
                        "queue_path": str(queue_result.queue_path),
                    }
                )
                break
            if next_runnable.get("action_type") == "manual_submit":
                final_decision = "prepare_manual_submit"
                rounds.append(
                    {
                        "iteration": iteration,
                        "decision": final_decision,
                        "reason": "Queue requested a manual_submit action.",
                        "queue_task": next_runnable,
                    }
                )
                break

            before = self._best_champion()
            enhancement = EnhancementRunner(self.competition_dir, memory=self.memory).run_next_recommendation()
            report = self._read_json(enhancement.validation_report)
            champion_selection = ExperimentChampionSelector(self.competition_dir, memory=self.memory).select()
            champion_state = self._write_champion_state(goal)
            candidate_pool = self._write_candidate_pool(goal, champion_state)
            after = champion_state.get("champion") or {}
            improved = self._is_improvement(after.get("local_score"), after.get("metric_name"), before)
            validator_ok = enhancement.validator_result.ok
            if validator_ok:
                state["validator_failure_rounds"] = 0
            else:
                state["validator_failure_rounds"] = int(state.get("validator_failure_rounds", 0)) + 1
            if improved:
                state["no_improvement_rounds"] = 0
            else:
                state["no_improvement_rounds"] = int(state.get("no_improvement_rounds", 0)) + 1

            decision = self._decide_after_iteration(goal, state, champion_state, candidate_pool, validator_ok)
            rounds.append(
                {
                    "iteration": iteration,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "decision": decision,
                    "brain_plan_path": str(review.json_path),
                    "queue_path": str(queue_result.queue_path),
                    "queue_task": next_runnable,
                    "task_id": enhancement.task_id,
                    "runner_status": enhancement.status,
                    "metric_name": report.get("metric_name"),
                    "local_score": report.get("local_score"),
                    "validator_ok": validator_ok,
                    "improved_champion": improved,
                    "champion_selection_path": str(champion_selection.selection_path),
                    "champion_after_iteration": after,
                    "state_after_iteration": dict(state),
                }
            )
            self._write_state(goal, state, rounds, decision)
            if decision in {"stop_target_reached", "prepare_manual_submit", "stop_blocked", "escalate_to_mac_brain"}:
                final_decision = decision
                break
            final_decision = decision

        return self._finish(goal, state, rounds, final_decision or "max_iterations_reached")

    def _load_or_create_goal(
        self,
        *,
        target: str,
        max_iterations: int,
        target_score: Optional[float],
    ) -> GoalSpec:
        path = self.competition_dir / "goal_spec.json"
        existing = self._read_json(path)
        leaderboard_target = self._read_json(self.competition_dir / "leaderboard_target.json")
        champion = self._best_champion()
        inferred_score = target_score
        if inferred_score is None:
            if target == "silver":
                inferred_score = leaderboard_target.get("estimated_silver_score") or leaderboard_target.get("silver_score")
            elif target == "top":
                inferred_score = leaderboard_target.get("top_score")
            elif target == "top10":
                inferred_score = leaderboard_target.get("top_10_score")
        metric_name = (
            champion.get("metric_name")
            or leaderboard_target.get("metric_name")
            or existing.get("metric_name")
            or "unknown"
        )
        goal = GoalSpec(
            target=target,
            metric_name=str(metric_name),
            target_score=inferred_score if isinstance(inferred_score, (int, float)) else None,
            current_champion=champion.get("task_id") if champion else existing.get("current_champion"),
            max_iterations=max_iterations,
            auto_submit=bool(existing.get("auto_submit", False)),
            human_submit_required=bool(existing.get("human_submit_required", True)),
            stop_conditions=GoalStopConditions(
                public_score_reaches_target=bool(
                    (existing.get("stop_conditions") or {}).get("public_score_reaches_target", True)
                ),
                no_improvement_rounds=int(
                    (existing.get("stop_conditions") or {}).get("no_improvement_rounds", 3)
                ),
                validator_failure_rounds=int(
                    (existing.get("stop_conditions") or {}).get("validator_failure_rounds", 2)
                ),
                mac_handoff_no_improvement_rounds=int(
                    (existing.get("stop_conditions") or {}).get("mac_handoff_no_improvement_rounds", 5)
                ),
            ),
        )
        path.write_text(json.dumps(goal.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return goal

    def _load_state(self) -> Dict[str, Any]:
        state = self._read_json(self.competition_dir / "agent_loop_state.json")
        return {
            "competition_name": self.competition_dir.name,
            "created_at": state.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "no_improvement_rounds": int(state.get("no_improvement_rounds", 0)),
            "validator_failure_rounds": int(state.get("validator_failure_rounds", 0)),
            "last_decision": state.get("last_decision") or "not_started",
        }

    def _write_state(
        self,
        goal: GoalSpec,
        state: Dict[str, Any],
        rounds: List[Dict[str, Any]],
        decision: str,
    ) -> Path:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["goal"] = goal.to_dict()
        state["last_decision"] = decision
        state["iterations_completed"] = len(rounds)
        state["rounds"] = rounds
        path = self.competition_dir / "agent_loop_state.json"
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _finish(
        self,
        goal: GoalSpec,
        state: Dict[str, Any],
        rounds: List[Dict[str, Any]],
        decision: str,
    ) -> AgentLoopResult:
        champion_state = self._write_champion_state(goal)
        candidate_pool = self._write_candidate_pool(goal, champion_state)
        handoff = self._write_mac_brain_handoff(goal, state, rounds, decision, champion_state, candidate_pool)
        state_path = self._write_state(goal, state, rounds, decision)
        summary = {
            "competition_name": self.competition_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "control_plane": self._control_plane(),
            "status": "completed" if rounds else "no_iteration",
            "decision": decision,
            "goal": goal.to_dict(),
            "champion_state": champion_state,
            "candidate_pool_summary": {
                "candidate_count": len(candidate_pool.get("candidates", [])),
                "ensemble_candidate_count": len(candidate_pool.get("ensemble_candidates", [])),
                "failed_candidate_count": len(candidate_pool.get("failed_candidates", [])),
            },
            "mac_brain_handoff": handoff,
            "iterations_completed": len(rounds),
            "rounds": rounds,
            "next_command": self._next_command(decision),
        }
        summary_path = self.competition_dir / "agent_loop_summary.md"
        summary_path.write_text(self._render_summary(summary), encoding="utf-8")
        json_summary_path = self.competition_dir / "agent_loop_summary.json"
        json_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        entry = self.ledger.create_entry(
            task_id="agent_loop",
            agent="agent_loop_controller",
            title="Run goal-oriented Agent Loop",
            status="pass" if decision not in {"stop_blocked"} else "needs_review",
            input_payload=summary,
            prompt="Drive AutoKaggle toward the configured leaderboard goal by repeatedly planning, executing, validating, and deciding the next action.",
            scorecard={
                "agent": "agent_loop_controller",
                "task_id": "agent_loop",
                "status": "pass" if decision not in {"stop_blocked"} else "needs_review",
                "scores": {
                    "iterations_completed": len(rounds),
                    "target_score": goal.target_score if goal.target_score is not None else "unknown",
                    "champion_score": (champion_state.get("champion") or {}).get("local_score", "n/a"),
                    "no_improvement_rounds": state.get("no_improvement_rounds", 0),
                    "validator_failure_rounds": state.get("validator_failure_rounds", 0),
                },
                "metric_name": goal.metric_name,
                "local_score": (champion_state.get("champion") or {}).get("local_score"),
                "issues": [] if decision != "stop_blocked" else ["Agent loop has no runnable next action."],
                "recommended_human_action": "continue" if not handoff.get("handoff_required") else "mac_brain_review",
            },
            artifacts={
                "goal_spec": self.competition_dir / "goal_spec.json",
                "agent_loop_state": state_path,
                "agent_loop_summary": summary_path,
                "agent_loop_summary_json": json_summary_path,
                "champion_state": self.competition_dir / "champion_state.json",
                "candidate_pool": self.competition_dir / "candidate_pool.json",
                "mac_brain_handoff": self.competition_dir / "mac_brain_handoff.json",
                "mac_brain_handoff_markdown": self.competition_dir / "mac_brain_handoff.md",
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=self.competition_dir.name,
                profile_name="tabular_classic",
                task_id="agent_loop",
                status=decision,
                metric_name=goal.metric_name,
                local_score=(champion_state.get("champion") or {}).get("local_score"),
                brain_review_path=str(json_summary_path),
                artifacts=[
                    str(self.competition_dir / "goal_spec.json"),
                    str(state_path),
                    str(summary_path),
                    str(self.competition_dir / "champion_state.json"),
                    str(self.competition_dir / "candidate_pool.json"),
                    str(self.competition_dir / "mac_brain_handoff.json"),
                    str(self.competition_dir / entry.html_report_path),
                ],
                notes=f"Agent loop decision: {decision}; Mac handoff required: {handoff.get('handoff_required')}",
            )
        )
        return AgentLoopResult(
            status=summary["status"],
            decision=decision,
            iterations_completed=len(rounds),
            goal_path=self.competition_dir / "goal_spec.json",
            state_path=state_path,
            summary_path=summary_path,
            champion_state_path=self.competition_dir / "champion_state.json",
            candidate_pool_path=self.competition_dir / "candidate_pool.json",
            mac_brain_handoff_path=self.competition_dir / "mac_brain_handoff.json",
            ledger_html_path=self.competition_dir / entry.html_report_path,
        )

    def _control_plane(self) -> Dict[str, Any]:
        return {
            "mode": "remote_autonomous_until_handoff",
            "mac_brain": {
                "role": "strategy_owner",
                "responsibilities": [
                    "set leaderboard goal and phase strategy",
                    "review bottlenecks and approve strategy shifts",
                    "approve real Kaggle submissions",
                ],
            },
            "remote_brain": {
                "role": "competition_tactical_planner",
                "responsibilities": [
                    "read candidate pool and leaderboard gap",
                    "design next experiment queue",
                    "produce narrow CodingLLM tasks",
                ],
            },
            "remote_coding_llm_runner": {
                "role": "execution_worker",
                "responsibilities": [
                    "execute only Remote Brain tasks",
                    "write validation, submission, OOF, and logs",
                    "avoid changing global goal or submission policy",
                ],
            },
        }

    def _write_mac_brain_handoff(
        self,
        goal: GoalSpec,
        state: Dict[str, Any],
        rounds: List[Dict[str, Any]],
        decision: str,
        champion_state: Dict[str, Any],
        candidate_pool: Dict[str, Any],
    ) -> Dict[str, Any]:
        champion = champion_state.get("champion") or {}
        handoff_required = decision in {
            "stop_target_reached",
            "prepare_manual_submit",
            "stop_blocked",
            "escalate_to_mac_brain",
        }
        reasons = []
        if decision == "prepare_manual_submit":
            reasons.append("Local champion reached the configured target; Mac Brain should decide submission strategy.")
        elif decision == "stop_target_reached":
            reasons.append("Public leaderboard feedback reached the configured target; Mac Brain should confirm closure or next target.")
        elif decision == "stop_blocked":
            reasons.append("Remote loop is blocked by queue, validation, or execution failures.")
        elif decision == "escalate_to_mac_brain":
            reasons.append("Remote loop hit the no-improvement handoff threshold and needs strategic replanning.")
        elif state.get("no_improvement_rounds", 0) >= goal.stop_conditions.no_improvement_rounds:
            reasons.append("Remote loop is exploring after repeated no-improvement rounds; Mac Brain review is optional.")
        else:
            reasons.append("Remote loop can continue autonomously.")

        recommended_actions = self._mac_brain_actions(decision)
        payload = {
            "competition_name": self.competition_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "handoff_required" if handoff_required else "remote_autonomy_continues",
            "handoff_required": handoff_required,
            "decision": decision,
            "reasons": reasons,
            "control_plane": self._control_plane(),
            "goal": goal.to_dict(),
            "remote_state": {
                "no_improvement_rounds": state.get("no_improvement_rounds", 0),
                "validator_failure_rounds": state.get("validator_failure_rounds", 0),
                "iterations_completed": len(rounds),
                "last_decision": decision,
            },
            "champion": {
                "task_id": champion.get("task_id"),
                "metric_name": champion.get("metric_name"),
                "local_score": champion.get("local_score"),
                "submission_valid": champion.get("submission_valid"),
                "gap_to_target": champion_state.get("gap_to_target"),
            },
            "candidate_pool_summary": {
                "candidate_count": candidate_pool.get("candidate_count", len(candidate_pool.get("candidates", []))),
                "ensemble_candidate_count": candidate_pool.get(
                    "ensemble_candidate_count",
                    len(candidate_pool.get("ensemble_candidates", [])),
                ),
                "failed_candidate_count": candidate_pool.get(
                    "failed_candidate_count",
                    len(candidate_pool.get("failed_candidates", [])),
                ),
            },
            "latest_round": rounds[-1] if rounds else {},
            "recommended_mac_brain_actions": recommended_actions,
            "next_remote_command_after_mac_plan": (
                f"python framework.py --competition {self.competition_dir.name} "
                "--agent-loop --target "
                f"{goal.target} --max-iterations 1 --execution-backend remote_linux"
            ),
        }
        json_path = self.competition_dir / "mac_brain_handoff.json"
        markdown_path = self.competition_dir / "mac_brain_handoff.md"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(self._render_mac_brain_handoff(payload), encoding="utf-8")
        return payload

    def _mac_brain_actions(self, decision: str) -> List[str]:
        if decision in {"prepare_manual_submit", "stop_target_reached"}:
            return [
                "Inspect champion_state.json and submit_decision_handoff.json.",
                "Decide whether to create a manual submission package or raise the leaderboard target.",
                "After public feedback is available, run the post-submit feedback workflow.",
            ]
        if decision == "stop_blocked":
            return [
                "Inspect agent_loop_summary.json, experiment_queue.json, and failed candidate logs.",
                "Revise the Mac-level strategy or patch Remote Brain constraints.",
                "Resume remote Agent Loop after the blocker is resolved.",
            ]
        if decision == "escalate_to_mac_brain":
            return [
                "Review candidate_pool.json and leaderboard gap to identify why current tactics saturated.",
                "Set a new phase plan: feature engineering, model diversity, ensembling, leakage audit, or external research.",
                "Write/update the Mac Brain plan, then resume the remote Agent Loop.",
            ]
        return [
            "No immediate Mac Brain intervention required.",
            "Let the remote Agent Loop continue unless the user wants to inspect the latest run.",
        ]

    def _render_mac_brain_handoff(self, payload: Dict[str, Any]) -> str:
        champion = payload.get("champion", {})
        lines = [
            "# Mac Brain Handoff",
            "",
            f"- Status: {payload.get('status')}",
            f"- Decision: {payload.get('decision')}",
            f"- Handoff required: {payload.get('handoff_required')}",
            f"- Champion: {champion.get('task_id', 'none')}",
            f"- Champion score: {champion.get('local_score', 'n/a')}",
            f"- Gap to target: {champion.get('gap_to_target', 'n/a')}",
            "",
            "## Reasons",
        ]
        for reason in payload.get("reasons", []):
            lines.append(f"- {reason}")
        lines.extend(["", "## Recommended Mac Brain Actions"])
        for action in payload.get("recommended_mac_brain_actions", []):
            lines.append(f"- {action}")
        lines.extend(
            [
                "",
                "## Resume Command",
                "",
                f"`{payload.get('next_remote_command_after_mac_plan')}`",
                "",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    def _preflight_decision(self, goal: GoalSpec, state: Dict[str, Any]) -> str:
        if self._public_score_reaches_target(goal):
            return "stop_target_reached"
        champion = self._best_champion()
        if self._score_reaches_target(champion.get("local_score"), champion.get("metric_name"), goal.target_score):
            return "prepare_manual_submit"
        if state["validator_failure_rounds"] >= goal.stop_conditions.validator_failure_rounds:
            return "stop_blocked"
        if state["no_improvement_rounds"] >= goal.stop_conditions.mac_handoff_no_improvement_rounds:
            return "escalate_to_mac_brain"
        return "continue_exploit"

    def _decide_after_iteration(
        self,
        goal: GoalSpec,
        state: Dict[str, Any],
        champion_state: Dict[str, Any],
        candidate_pool: Dict[str, Any],
        validator_ok: bool,
    ) -> str:
        if self._public_score_reaches_target(goal):
            return "stop_target_reached"
        champion = champion_state.get("champion") or {}
        if self._score_reaches_target(champion.get("local_score"), champion.get("metric_name"), goal.target_score):
            return "prepare_manual_submit"
        if not validator_ok:
            if state["validator_failure_rounds"] >= goal.stop_conditions.validator_failure_rounds:
                return "stop_blocked"
            return "run_validation_audit"
        if state["no_improvement_rounds"] >= goal.stop_conditions.mac_handoff_no_improvement_rounds:
            return "escalate_to_mac_brain"
        if state["no_improvement_rounds"] >= goal.stop_conditions.no_improvement_rounds:
            return "continue_explore"
        if len(candidate_pool.get("ensemble_candidates", [])) >= 2:
            return "build_ensemble"
        return "continue_exploit"

    def _write_champion_state(self, goal: GoalSpec) -> Dict[str, Any]:
        selection = self._read_json(self.competition_dir / "champion_selection.json")
        champion = selection.get("champion") or self._best_champion()
        leaderboard_target = self._read_json(self.competition_dir / "leaderboard_target.json")
        public_feedback = self._read_json(self.competition_dir / "leaderboard_feedback.json")
        gap_to_target = self._gap(champion.get("local_score"), champion.get("metric_name"), goal.target_score)
        state = {
            "competition_name": self.competition_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "goal_target": goal.target,
            "target_score": goal.target_score,
            "metric_name": goal.metric_name,
            "champion": champion,
            "gap_to_target": gap_to_target,
            "leaderboard_target": leaderboard_target,
            "leaderboard_feedback": public_feedback,
            "target_reached_by_public_score": self._public_score_reaches_target(goal),
            "target_reached_by_local_score": self._score_reaches_target(
                champion.get("local_score"),
                champion.get("metric_name"),
                goal.target_score,
            ),
        }
        (self.competition_dir / "champion_state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return state

    def _write_candidate_pool(self, goal: GoalSpec, champion_state: Dict[str, Any]) -> Dict[str, Any]:
        candidates = []
        failed = []
        ensemble = []
        champion_task = (champion_state.get("champion") or {}).get("task_id")
        champion_score = (champion_state.get("champion") or {}).get("local_score")
        artifact_roots = [
            path.parent
            for path in sorted((self.competition_dir / "experiments").glob("*/validation_report.json"))
        ]
        artifact_roots.extend(
            path.parent
            for path in sorted((self.competition_dir / "runs").glob("*/artifacts/validation_report.json"))
        )
        seen_roots = set()
        for exp_dir in artifact_roots:
            if exp_dir in seen_roots:
                continue
            seen_roots.add(exp_dir)
            validation_path = exp_dir / "validation_report.json"
            report = self._read_json(validation_path)
            validator = self._read_json(exp_dir / "validator_result.json")
            task_id = self._candidate_task_id(exp_dir, report)
            item = {
                "task_id": task_id,
                "experiment_dir": str(exp_dir),
                "runner_kind": report.get("runner_kind"),
                "status": report.get("status"),
                "metric_name": report.get("metric_name"),
                "local_score": report.get("local_score"),
                "fold_std": report.get("fold_std"),
                "train_valid_gap": report.get("train_valid_gap"),
                "validator_ok": validator.get("ok") is True,
                "has_oof": (exp_dir / "oof_predictions.csv").exists(),
                "has_submission": (exp_dir / "submission.csv").exists(),
                "is_champion": task_id == champion_task,
                "gap_to_champion": self._gap(
                    report.get("local_score"),
                    report.get("metric_name"),
                    champion_score,
                    target_is_champion=True,
                ),
            }
            if item["validator_ok"] and isinstance(item["local_score"], (int, float)):
                candidates.append(item)
                if item["has_oof"] and not item["is_champion"]:
                    ensemble.append(item)
            else:
                failed.append(item)
        candidates = self._dedupe_candidates(candidates)
        ensemble = self._dedupe_candidates(ensemble)
        failed = self._dedupe_candidates(failed)
        pool = {
            "competition_name": self.competition_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "goal": goal.to_dict(),
            "target_score": goal.target_score,
            "champion_task_id": champion_task,
            "champion_score": champion_score,
            "gap_to_target": champion_state.get("gap_to_target"),
            "candidate_count": len(candidates),
            "ensemble_candidate_count": len(ensemble),
            "failed_candidate_count": len(failed),
            "candidates": candidates,
            "ensemble_candidates": ensemble,
            "failed_candidates": failed,
        }
        (self.competition_dir / "candidate_pool.json").write_text(
            json.dumps(pool, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return pool

    def _candidate_task_id(self, exp_dir: Path, report: Dict[str, Any]) -> str:
        explicit = report.get("experiment") or report.get("baseline") or report.get("task_id")
        if explicit:
            return str(explicit)
        if exp_dir.name == "artifacts" and exp_dir.parent.name:
            run_name = exp_dir.parent.name
            return run_name.split("_", 1)[1] if "_" in run_name else run_name
        return exp_dir.name

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_task: Dict[str, Dict[str, Any]] = {}
        for item in candidates:
            task_id = str(item.get("task_id") or item.get("experiment_dir") or "")
            if not task_id:
                continue
            existing = by_task.get(task_id)
            if existing is None or self._candidate_completeness(item) > self._candidate_completeness(existing):
                by_task[task_id] = item
        return sorted(by_task.values(), key=lambda item: str(item.get("task_id") or ""))

    def _candidate_completeness(self, item: Dict[str, Any]) -> int:
        score = 0
        score += 4 if item.get("validator_ok") else 0
        score += 3 if isinstance(item.get("local_score"), (int, float)) else 0
        score += 2 if item.get("has_oof") else 0
        score += 1 if item.get("has_submission") else 0
        return score

    def _refresh_leaderboard_target(self) -> None:
        try:
            LeaderboardTargetAgent(self.competition_dir).run(page_size=200)
        except Exception as exc:
            path = self.competition_dir / "leaderboard_target_error.json"
            path.write_text(
                json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _best_champion(self) -> Dict[str, Any]:
        selection = self._read_json(self.competition_dir / "champion_selection.json")
        champion = selection.get("champion")
        if isinstance(champion, dict) and isinstance(champion.get("local_score"), (int, float)):
            return champion
        best: Dict[str, Any] = {}
        for validation_path in sorted((self.competition_dir / "experiments").glob("*/validation_report.json")):
            report = self._read_json(validation_path)
            score = report.get("local_score")
            metric = report.get("metric_name")
            if self._is_improvement(score, metric, best):
                best = {
                    "task_id": report.get("experiment") or validation_path.parent.name,
                    "metric_name": metric,
                    "local_score": score,
                    "status": report.get("status"),
                    "source": "validation_report",
                }
        return best

    def _public_score_reaches_target(self, goal: GoalSpec) -> bool:
        feedback = self._read_json(self.competition_dir / "leaderboard_feedback.json")
        public_score = feedback.get("public_score")
        metric_name = feedback.get("metric_name") or goal.metric_name
        return self._score_reaches_target(public_score, metric_name, goal.target_score)

    def _is_improvement(self, score: Any, metric_name: Optional[str], best: Dict[str, Any]) -> bool:
        if not isinstance(score, (int, float)):
            return False
        best_score = best.get("local_score")
        if not isinstance(best_score, (int, float)):
            return True
        lower_is_better = str(metric_name or "").lower() in LOWER_IS_BETTER
        return score < best_score if lower_is_better else score > best_score

    def _score_reaches_target(self, score: Any, metric_name: Optional[str], target_score: Optional[float]) -> bool:
        if not isinstance(score, (int, float)) or not isinstance(target_score, (int, float)):
            return False
        lower_is_better = str(metric_name or "").lower() in LOWER_IS_BETTER
        return score <= target_score if lower_is_better else score >= target_score

    def _gap(
        self,
        score: Any,
        metric_name: Optional[str],
        target_score: Optional[float],
        *,
        target_is_champion: bool = False,
    ) -> Optional[float]:
        if not isinstance(score, (int, float)) or not isinstance(target_score, (int, float)):
            return None
        lower_is_better = str(metric_name or "").lower() in LOWER_IS_BETTER
        if target_is_champion:
            return float(score - target_score) if lower_is_better else float(target_score - score)
        return float(score - target_score) if lower_is_better else float(target_score - score)

    def _next_command(self, decision: str) -> str:
        if decision == "prepare_manual_submit":
            return f"python framework.py --competition {self.competition_dir.name} --manual-submission-package --submission-target champion"
        if decision == "run_validation_audit":
            return f"python framework.py --competition {self.competition_dir.name} --tabular-risk-audit"
        if decision == "build_ensemble":
            return f"python framework.py --competition {self.competition_dir.name} --run-enhancement"
        if decision == "escalate_to_mac_brain":
            return f"review {self.competition_dir / 'mac_brain_handoff.md'}"
        if decision.startswith("continue"):
            return f"python framework.py --competition {self.competition_dir.name} --agent-loop --target silver --max-iterations 1"
        return "review agent_loop_summary.md"

    def _render_summary(self, summary: Dict[str, Any]) -> str:
        champion = summary.get("champion_state", {}).get("champion") or {}
        lines = [
            "# Agent Loop Summary",
            "",
            f"- Competition: {summary.get('competition_name')}",
            f"- Decision: {summary.get('decision')}",
            f"- Target: {summary.get('goal', {}).get('target')}",
            f"- Target score: {summary.get('goal', {}).get('target_score')}",
            f"- Champion: {champion.get('task_id', 'none')}",
            f"- Champion score: {champion.get('local_score', 'n/a')}",
            f"- Mac handoff: {summary.get('mac_brain_handoff', {}).get('status', 'unknown')}",
            f"- Iterations completed: {summary.get('iterations_completed')}",
            f"- Next command: `{summary.get('next_command')}`",
            "",
            "## Rounds",
        ]
        for item in summary.get("rounds", []):
            lines.extend(
                [
                    "",
                    f"### Round {item.get('iteration')}",
                    f"- Decision: {item.get('decision')}",
                    f"- Task: {item.get('task_id', 'n/a')}",
                    f"- Score: {item.get('local_score', 'n/a')}",
                    f"- Validator OK: {item.get('validator_ok', 'n/a')}",
                    f"- Improved champion: {item.get('improved_champion', 'n/a')}",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
