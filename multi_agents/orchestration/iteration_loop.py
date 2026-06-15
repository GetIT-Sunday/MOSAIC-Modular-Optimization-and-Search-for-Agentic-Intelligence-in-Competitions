from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .enhancement_runner import EnhancementRunner
from .memory import CompetitionMemory, ExperimentRecord
from .remote_brain import RemoteBrainReviewer
from .run_ledger import RunLedger


@dataclass(frozen=True)
class IterationLoopResult:
    summary_path: Path
    ledger_html_path: Path
    iterations_completed: int


class IterationOrchestrator:
    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
        use_llm: bool = True,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.use_llm = use_llm
        self.ledger = RunLedger(self.competition_dir)

    def run(
        self,
        max_iterations: int = 1,
        patience: int = 2,
        target_score: Optional[float] = None,
    ) -> IterationLoopResult:
        iterations: List[Dict[str, Any]] = []
        best_state = self._load_best_state()
        no_improvement_count = int(best_state.get("no_improvement_count", 0))
        stop_reason = None
        for index in range(1, max_iterations + 1):
            if target_score is not None and self._score_reaches_target(best_state, target_score):
                stop_reason = "target_score_reached_before_iteration"
                break
            if no_improvement_count >= patience:
                stop_reason = "patience_exhausted_before_iteration"
                break

            review = RemoteBrainReviewer(
                self.competition_dir,
                memory=self.memory,
                use_llm=self.use_llm,
            ).review()
            plan = self._read_json(review.json_path)
            if plan.get("next_action") == "stop":
                iterations.append(
                    {
                        "iteration": index,
                        "status": "stopped_by_remote_brain",
                        "plan_path": str(review.json_path),
                    }
                )
                stop_reason = "stopped_by_remote_brain"
                break
            enhancement = EnhancementRunner(
                self.competition_dir,
                memory=self.memory,
            ).run_next_recommendation()
            report = self._read_json(enhancement.validation_report)
            score = report.get("local_score")
            metric_name = report.get("metric_name")
            improved = self._is_improvement(score, metric_name, best_state)
            if improved:
                best_state = {
                    "task_id": enhancement.task_id,
                    "metric_name": metric_name,
                    "local_score": score,
                    "source": "iteration_loop",
                    "no_improvement_count": 0,
                }
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                best_state["no_improvement_count"] = no_improvement_count
            self._write_best_state(best_state)

            iterations.append(
                {
                    "iteration": index,
                    "status": enhancement.status,
                    "plan_path": str(review.json_path),
                    "task_id": enhancement.task_id,
                    "metric_name": metric_name,
                    "local_score": score,
                    "submission_valid": enhancement.validator_result.ok,
                    "improved_best": improved,
                    "best_state_after_iteration": dict(best_state),
                }
            )
            if target_score is not None and self._score_reaches_target(best_state, target_score):
                stop_reason = "target_score_reached"
                break
            if no_improvement_count >= patience:
                stop_reason = "patience_exhausted"
                break

        summary = {
            "competition_name": self.competition_dir.name,
            "max_iterations": max_iterations,
            "patience": patience,
            "target_score": target_score,
            "iterations_completed": len(iterations),
            "stop_reason": stop_reason or "max_iterations_reached",
            "best_state": best_state,
            "iterations": iterations,
        }
        summary_path = self.competition_dir / "optimization_loop_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        entry = self.ledger.create_entry(
            task_id="optimization_loop_summary",
            agent="iteration_orchestrator",
            title="Summarize automatic optimization loop",
            status="pass" if iterations else "needs_review",
            input_payload=summary,
            prompt=(
                "Summarize the completed remote Brain and enhancement iterations, "
                "including metric movement and whether another iteration should run."
            ),
            scorecard={
                "agent": "iteration_orchestrator",
                "task_id": "optimization_loop_summary",
                "status": "pass" if iterations else "needs_review",
                "scores": {
                    "iterations_completed": len(iterations),
                    "valid_submission_count": sum(
                        1 for item in iterations if item.get("submission_valid")
                    ),
                    "best_score": best_state.get("local_score", "n/a"),
                    "no_improvement_count": best_state.get("no_improvement_count", 0),
                    "automation_continuity": 5 if iterations else 1,
                },
                "metric_name": best_state.get("metric_name") or self._best_metric(iterations),
                "local_score": best_state.get("local_score") or self._best_score(iterations),
                "issues": [] if iterations else [summary["stop_reason"]],
                "recommended_human_action": (
                    "continue" if summary["stop_reason"] in {"max_iterations_reached", "target_score_reached"} else "patch_prompt"
                ),
            },
            artifacts={"optimization_loop_summary": summary_path},
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=self.competition_dir.name,
                profile_name="tabular_classic",
                task_id="optimization_loop_summary",
                status="completed" if iterations else "needs_review",
                metric_name=best_state.get("metric_name") or self._best_metric(iterations),
                local_score=best_state.get("local_score") or self._best_score(iterations),
                brain_review_path=str(summary_path),
                artifacts=[str(summary_path), str(self.competition_dir / entry.html_report_path)],
                notes=f"Completed {len(iterations)} automatic optimization iteration(s).",
            )
        )
        return IterationLoopResult(
            summary_path=summary_path,
            ledger_html_path=self.competition_dir / entry.html_report_path,
            iterations_completed=len(iterations),
        )

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_best_state(self) -> Dict[str, Any]:
        state_path = self.competition_dir / "best_score.json"
        if state_path.exists():
            return self._read_json(state_path)
        best = self._best_from_existing_reports()
        self._write_best_state(best)
        return best

    def _write_best_state(self, state: Dict[str, Any]) -> None:
        (self.competition_dir / "best_score.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _best_from_existing_reports(self) -> Dict[str, Any]:
        best: Dict[str, Any] = {"no_improvement_count": 0}
        for report_path in sorted((self.competition_dir / "experiments").glob("*/validation_report.json")):
            report = self._read_json(report_path)
            score = report.get("local_score")
            metric_name = report.get("metric_name")
            candidate = {
                "task_id": report_path.parent.name,
                "metric_name": metric_name,
                "local_score": score,
                "source": "existing_reports",
                "no_improvement_count": 0,
            }
            if self._is_improvement(score, metric_name, best):
                best = candidate
        return best

    def _is_improvement(
        self,
        score: Any,
        metric_name: Optional[str],
        best_state: Dict[str, Any],
    ) -> bool:
        if not isinstance(score, (int, float)):
            return False
        best_score = best_state.get("local_score")
        if not isinstance(best_score, (int, float)):
            return True
        lower_is_better = metric_name in {"rmse", "rmsle", "mae", "log_loss"}
        return score < best_score if lower_is_better else score > best_score

    def _score_reaches_target(self, best_state: Dict[str, Any], target_score: float) -> bool:
        score = best_state.get("local_score")
        metric_name = best_state.get("metric_name")
        if not isinstance(score, (int, float)):
            return False
        lower_is_better = metric_name in {"rmse", "rmsle", "mae", "log_loss"}
        return score <= target_score if lower_is_better else score >= target_score

    def _best_score(self, iterations: List[Dict[str, Any]]) -> Optional[float]:
        scores = [item.get("local_score") for item in iterations]
        numeric = [score for score in scores if isinstance(score, (int, float))]
        return max(numeric) if numeric else None

    def _best_metric(self, iterations: List[Dict[str, Any]]) -> Optional[str]:
        for item in reversed(iterations):
            if item.get("metric_name"):
                return item["metric_name"]
        return None
