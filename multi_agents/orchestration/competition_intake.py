from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .brain import BrainOrchestrator
from .ingestion import CompetitionIngestor
from .memory import CompetitionMemory
from .run_ledger import RunLedger


@dataclass(frozen=True)
class CompetitionIntakeResult:
    status: str
    competition_slug: str
    intake_path: Path
    manifest_path: Path
    task_card_path: Path
    metric_spec_path: Path
    experiment_plan_path: Path
    unknown_fields: list[str]
    blocking_items: list[str]
    next_command: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in [
            "intake_path",
            "manifest_path",
            "task_card_path",
            "metric_spec_path",
            "experiment_plan_path",
        ]:
            payload[key] = str(payload[key])
        return payload


class CompetitionIntakeAgent:
    """Turns a selected Kaggle competition directory into Brain-ready artifacts."""

    REQUIRED_FILES = ["train.csv", "test.csv", "sample_submission.csv"]

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(self) -> CompetitionIntakeResult:
        self.competition_dir.mkdir(parents=True, exist_ok=True)
        intake = self._read_json(self.competition_dir / "competition_intake.json")
        manifest = CompetitionIngestor(self.competition_dir).build_manifest()
        unknown_fields = self._unknown_fields(manifest)
        blocking_items = self._blocking_items(manifest)

        brain = BrainOrchestrator(memory=self.memory)
        decision = brain.decide(self.competition_dir)
        artifacts = brain.persist_artifacts(self.competition_dir, decision)

        status = "ready_for_baseline" if not blocking_items else "needs_data_or_review"
        next_command = (
            f"python framework.py --competition {self.competition_dir.name} --agent-baseline-start"
            if status == "ready_for_baseline"
            else f"python framework.py --kaggle-select {self.competition_dir.name} --kaggle-download"
        )

        payload = {
            **intake,
            "status": status,
            "competition_slug": intake.get("competition_slug", self.competition_dir.name),
            "competition_dir": str(self.competition_dir),
            "intake_agent": {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "unknown_fields": unknown_fields,
                "blocking_items": blocking_items,
                "next_command": next_command,
                "generated_artifacts": {
                    "data_manifest": str(artifacts["data_manifest"]),
                    "task_card": str(artifacts["task_card"]),
                    "metric_spec": str(artifacts["metric_spec"]),
                    "experiment_plan": str(artifacts["experiment_plan"]),
                },
            },
            "next_step": "run_baseline" if status == "ready_for_baseline" else "download_or_fix_unknown_fields",
            "recommended_commands": self._recommended_commands(status),
        }
        intake_path = self.competition_dir / "competition_intake.json"
        intake_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self.ledger.create_entry(
            task_id="competition_intake",
            agent="competition_intake_agent",
            title="解析竞赛 Intake 并生成 Brain 初始产物",
            status=status,
            input_payload={
                "competition_name": self.competition_dir.name,
                "existing_intake": intake,
                "manifest": manifest.to_dict(),
            },
            prompt=(
                "读取 Kaggle 选题结果和本地数据文件，生成 data_manifest、task_card、"
                "metric_spec、experiment_plan，并标记 unknown 字段。"
            ),
            scorecard={
                "agent": "competition_intake_agent",
                "task_id": "competition_intake",
                "status": "pass" if status == "ready_for_baseline" else "needs_review",
                "scores": {
                    "required_files_present": 5 if not blocking_items else 1,
                    "unknown_field_count": len(unknown_fields),
                    "brain_artifacts_created": 5,
                },
                "issues": blocking_items + [f"{field} is unknown" for field in unknown_fields],
                "recommended_human_action": "continue" if status == "ready_for_baseline" else "patch_prompt",
            },
            artifacts={
                "competition_intake": intake_path,
                "data_manifest": artifacts["data_manifest"],
                "task_card": artifacts["task_card"],
                "metric_spec": artifacts["metric_spec"],
                "experiment_plan": artifacts["experiment_plan"],
            },
        )
        return CompetitionIntakeResult(
            status=status,
            competition_slug=str(payload["competition_slug"]),
            intake_path=intake_path,
            manifest_path=artifacts["data_manifest"],
            task_card_path=artifacts["task_card"],
            metric_spec_path=artifacts["metric_spec"],
            experiment_plan_path=artifacts["experiment_plan"],
            unknown_fields=unknown_fields,
            blocking_items=blocking_items,
            next_command=next_command,
        )

    def _recommended_commands(self, status: str) -> list[str]:
        slug = self.competition_dir.name
        commands = [
            f"python framework.py --competition {slug} --competition-intake",
        ]
        if status == "ready_for_baseline":
            commands.extend(
                [
                    f"python framework.py --competition {slug} --agent-baseline-start",
                    f"python framework.py --competition {slug} --remote-brain-review",
                ]
            )
        else:
            commands.extend(
                [
                    f"python framework.py --kaggle-select {slug} --kaggle-download",
                    f"python framework.py --competition {slug} --competition-intake",
                ]
            )
        return commands

    def _unknown_fields(self, manifest) -> list[str]:
        fields = []
        if manifest.id_column == "unknown":
            fields.append("id_column")
        if manifest.target_column == "unknown":
            fields.append("target_column")
        if not manifest.metric_candidates or manifest.metric_candidates == ["unknown"]:
            fields.append("metric")
        if manifest.task_type == "unknown":
            fields.append("task_type")
        return fields

    def _blocking_items(self, manifest) -> list[str]:
        items = []
        for name in self.REQUIRED_FILES:
            table = manifest.tables.get(name)
            if not table or not table.exists:
                items.append(f"missing {name}")
        return items

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
