from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .coding_task import CodingTask
from .ingestion import CompetitionIngestor, DataManifest
from .memory import CompetitionMemory, ExperimentRecord
from .profile import CompetitionProfile, load_profile
from .run_ledger import RunLedger
from .task_identifier import CompetitionSignal, ProfileDecision, identify_profile
from .validator import SubmissionValidator, ValidationResult


@dataclass(frozen=True)
class BrainDecision:
    competition_name: str
    profile: CompetitionProfile
    profile_decision: ProfileDecision
    data_manifest: DataManifest
    coding_tasks: List[CodingTask]
    memory_summary: dict


class BrainOrchestrator:
    """Deterministic Brain Agent skeleton for profile-driven competition work."""

    def __init__(self, memory: Optional[CompetitionMemory] = None):
        self.memory = memory or CompetitionMemory()

    def build_signal(self, competition_dir: Path) -> CompetitionSignal:
        competition_dir = competition_dir.resolve()
        files = [
            str(path.relative_to(competition_dir))
            for path in competition_dir.rglob("*")
            if path.is_file()
        ]
        overview_text = self._read_optional_text(competition_dir / "overview.txt")
        return CompetitionSignal(
            competition_name=competition_dir.name,
            files=files,
            overview_text=overview_text,
        )

    def decide(self, competition_dir: Path) -> BrainDecision:
        signal = self.build_signal(competition_dir)
        profile_decision = identify_profile(signal)
        profile = load_profile(profile_decision.profile_name)
        manifest = CompetitionIngestor(competition_dir).build_manifest()
        tasks = self.plan_initial_tasks(signal, profile, manifest)
        return BrainDecision(
            competition_name=signal.competition_name,
            profile=profile,
            profile_decision=profile_decision,
            data_manifest=manifest,
            coding_tasks=tasks,
            memory_summary=self.memory.leaderboard_summary(profile.name),
        )

    def plan_initial_tasks(
        self,
        signal: CompetitionSignal,
        profile: CompetitionProfile,
        manifest: Optional[DataManifest] = None,
    ) -> List[CodingTask]:
        submission_checks = profile.raw.get("submission_gate", {}).get("checks", [])
        common_constraints = [
            "write every artifact inside the competition workspace",
            "produce a reproducible Python script before running expensive experiments",
            "validate submission schema before any leaderboard submission",
        ]
        tasks = [
            CodingTask(
                task_id="competition_audit",
                title="Audit competition files, schema, metric, and submission contract",
                profile_name=profile.name,
                objective=(
                    "Inspect available files and overview text, then produce a concise "
                    "task brief with target, metric, split strategy, and submission format."
                ),
                inputs=signal.files,
                expected_outputs=["task_brief.md", "schema_report.json"],
                validation_checks=["all required files accounted for", "metric identified"],
                constraints=common_constraints,
                context=self._task_context(signal, manifest),
            )
        ]
        for index, baseline in enumerate(profile.baseline_ladder[:3], start=1):
            tasks.append(
                CodingTask(
                    task_id=f"baseline_{index}_{baseline}",
                    title=f"Implement {baseline}",
                    profile_name=profile.name,
                    objective=(
                        "Create, run, and evaluate this baseline. Save predictions, "
                        "local validation score, and a short experiment report."
                    ),
                    inputs=signal.files,
                    expected_outputs=[
                        f"experiments/{baseline}/run.py",
                        f"experiments/{baseline}/validation_report.json",
                        f"experiments/{baseline}/submission.csv",
                    ],
                    validation_checks=list(submission_checks),
                    constraints=common_constraints,
                    context=self._task_context(signal, manifest),
                )
            )
        return tasks

    def run_dry_loop(self, competition_dir: Path) -> BrainDecision:
        competition_dir = competition_dir.resolve()
        decision = self.decide(competition_dir)
        artifacts = self.persist_artifacts(competition_dir, decision)
        validation = self.validate_sample_submission(competition_dir, decision.data_manifest)
        review = self.write_brain_review(competition_dir, decision, validation)
        ledger = RunLedger(competition_dir)
        plan_entry = ledger.create_entry(
            task_id="brain_plan",
            agent="brain",
            title="Generate task card, metric spec, manifest, and experiment plan",
            status="pass",
            input_payload={
                "competition_name": decision.competition_name,
                "profile_decision": decision.profile_decision.__dict__,
                "manifest_notes": decision.data_manifest.notes,
                "memory_summary": decision.memory_summary,
            },
            prompt=self._brain_plan_prompt(decision),
            scorecard={
                "agent": "brain",
                "task_id": "brain_plan",
                "status": "pass",
                "scores": {
                    "profile_selected": 5,
                    "manifest_created": 5,
                    "task_card_created": 5,
                    "human_readability": 4,
                },
                "issues": decision.data_manifest.notes,
                "recommended_human_action": "continue",
            },
            artifacts=artifacts,
        )
        validation_entry = ledger.create_entry(
            task_id="sample_submission_validation",
            agent="validator",
            title="Validate sample submission against manifest schema",
            status="validated" if validation.ok else "validation_failed",
            input_payload={
                "competition_name": decision.competition_name,
                "submission_path": str(competition_dir / "sample_submission.csv"),
                "manifest": decision.data_manifest.to_dict(),
            },
            prompt="Validate sample_submission.csv against data_manifest.json before any leaderboard submission.",
            scorecard={
                "agent": "validator",
                "task_id": "sample_submission_validation",
                "status": "pass" if validation.ok else "needs_review",
                "scores": {
                    "columns": 5 if not any("columns mismatch" in error for error in validation.errors) else 1,
                    "row_count": 5 if not any("row count mismatch" in error for error in validation.errors) else 1,
                    "id_integrity": 5 if not any("id" in error.lower() for error in validation.errors) else 1,
                    "missing_predictions": 5 if not any("missing predictions" in error for error in validation.errors) else 1,
                },
                "issues": validation.errors + validation.warnings,
                "recommended_human_action": "continue" if validation.ok else "rerun",
            },
            artifacts={
                "brain_review": review,
                "sample_submission": competition_dir / "sample_submission.csv",
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=decision.competition_name,
                profile_name=decision.profile.name,
                task_id="brain_dry_loop",
                status="validated" if validation.ok else "validation_failed",
                metric_name=decision.data_manifest.metric_candidates[0],
                submission_path=str(competition_dir.resolve() / "sample_submission.csv"),
                failure_reason="; ".join(validation.errors),
                brain_review_path=str(review),
                artifacts=[
                    str(path) for path in artifacts.values()
                ] + [str(review), str(competition_dir / plan_entry.html_report_path)],
                notes="Brain dry loop generated planning artifacts and validated sample submission.",
            )
        )
        return decision

    def persist_artifacts(
        self,
        competition_dir: Path,
        decision: BrainDecision,
    ) -> Dict[str, Path]:
        competition_dir = competition_dir.resolve()
        artifacts = {
            "task_card": competition_dir / "task_card.md",
            "metric_spec": competition_dir / "metric_spec.json",
            "data_manifest": competition_dir / "data_manifest.json",
            "experiment_plan": competition_dir / "experiment_plan.json",
        }
        artifacts["task_card"].write_text(self.render_task_card(decision), encoding="utf-8")
        artifacts["metric_spec"].write_text(
            json.dumps(self.metric_spec(decision), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        decision.data_manifest.write_json(artifacts["data_manifest"])
        artifacts["experiment_plan"].write_text(
            json.dumps(self.experiment_plan(decision), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._persist_task_prompts(competition_dir, decision.coding_tasks)
        return artifacts

    def validate_sample_submission(
        self,
        competition_dir: Path,
        manifest: DataManifest,
    ) -> ValidationResult:
        return SubmissionValidator(manifest).validate(competition_dir.resolve() / "sample_submission.csv")

    def write_brain_review(
        self,
        competition_dir: Path,
        decision: BrainDecision,
        validation: ValidationResult,
    ) -> Path:
        path = competition_dir.resolve() / "brain_review.json"
        payload = {
            "competition_name": decision.competition_name,
            "profile_name": decision.profile.name,
            "profile_confidence": decision.profile_decision.confidence,
            "validation": validation.to_dict(),
            "decision": "continue_to_coding_agent" if validation.ok else "rerun_audit",
            "next_task_id": decision.coding_tasks[0].task_id if decision.coding_tasks else None,
            "memory_summary": decision.memory_summary,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def render_task_card(self, decision: BrainDecision) -> str:
        manifest = decision.data_manifest
        lines = [
            f"# Task Card: {decision.competition_name}",
            "",
            f"- Profile: `{decision.profile.name}`",
            f"- Profile confidence: `{decision.profile_decision.confidence}`",
            f"- Task type: `{manifest.task_type}`",
            f"- ID column: `{manifest.id_column}`",
            f"- Target column: `{manifest.target_column}`",
            f"- Metric candidates: `{', '.join(manifest.metric_candidates)}`",
            "",
            "## Profile Reasons",
            *[f"- {reason}" for reason in decision.profile_decision.reasons],
            "",
            "## Coding Tasks",
        ]
        for task in decision.coding_tasks:
            lines.extend(["", f"### {task.task_id}", "", task.to_prompt()])
        return "\n".join(lines).rstrip() + "\n"

    def metric_spec(self, decision: BrainDecision) -> dict:
        manifest = decision.data_manifest
        return {
            "competition_name": decision.competition_name,
            "profile_name": decision.profile.name,
            "task_type": manifest.task_type,
            "metric_candidates": manifest.metric_candidates,
            "id_column": manifest.id_column,
            "target_column": manifest.target_column,
            "submission_columns": manifest.submission_columns,
            "notes": manifest.notes,
        }

    def experiment_plan(self, decision: BrainDecision) -> dict:
        return {
            "competition_name": decision.competition_name,
            "profile_name": decision.profile.name,
            "baseline_ladder": decision.profile.baseline_ladder,
            "tasks": [task.to_dict() for task in decision.coding_tasks],
            "memory_summary": decision.memory_summary,
        }

    def _brain_plan_prompt(self, decision: BrainDecision) -> str:
        return (
            "Inspect the local competition files, infer the task profile and schema, "
            "write human-reviewable planning artifacts, and prepare narrow CodingAgent tasks.\n\n"
            f"Competition: {decision.competition_name}\n"
            f"Profile: {decision.profile.name}\n"
            f"Task type: {decision.data_manifest.task_type}\n"
            f"ID column: {decision.data_manifest.id_column}\n"
            f"Target column: {decision.data_manifest.target_column}\n"
            f"Metric candidates: {', '.join(decision.data_manifest.metric_candidates)}\n"
            f"Tasks: {', '.join(task.task_id for task in decision.coding_tasks)}\n"
        )

    def _persist_task_prompts(self, competition_dir: Path, tasks: List[CodingTask]) -> None:
        experiments_dir = competition_dir / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        for task in tasks:
            task_dir = experiments_dir / task.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "coding_prompt.md").write_text(task.to_prompt() + "\n", encoding="utf-8")

    def _task_context(
        self,
        signal: CompetitionSignal,
        manifest: Optional[DataManifest],
    ) -> Dict[str, str]:
        context = {"competition_name": signal.competition_name}
        if manifest is not None:
            context.update(
                {
                    "id_column": manifest.id_column,
                    "target_column": manifest.target_column,
                    "task_type": manifest.task_type,
                    "metric_candidates": ", ".join(manifest.metric_candidates),
                }
            )
        return context

    @staticmethod
    def _read_optional_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
