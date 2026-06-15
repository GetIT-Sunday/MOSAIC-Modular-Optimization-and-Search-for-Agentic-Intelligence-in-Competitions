from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


MEMORY_DIR = Path(__file__).resolve().parents[1] / "competition_memory"


@dataclass(frozen=True)
class ExperimentRecord:
    competition_name: str
    profile_name: str
    task_id: str
    status: str
    metric_name: Optional[str] = None
    local_score: Optional[float] = None
    public_score: Optional[float] = None
    leaderboard_rank: Optional[int] = None
    script_path: Optional[str] = None
    submission_path: Optional[str] = None
    failure_reason: str = ""
    brain_review_path: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CompetitionMemory:
    def __init__(self, root: Path = MEMORY_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "experiments.jsonl"

    def append(self, record: ExperimentRecord) -> None:
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def list_records(self) -> List[ExperimentRecord]:
        if not self.records_path.exists():
            return []
        records = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(ExperimentRecord(**json.loads(line)))
        return records

    def query(
        self,
        *,
        profile_name: Optional[str] = None,
        competition_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ExperimentRecord]:
        records: Iterable[ExperimentRecord] = self.list_records()
        if profile_name is not None:
            records = [record for record in records if record.profile_name == profile_name]
        if competition_name is not None:
            records = [
                record
                for record in records
                if record.competition_name == competition_name
            ]
        if status is not None:
            records = [record for record in records if record.status == status]
        return list(records)

    def leaderboard_summary(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        records = [
            record
            for record in self.query(profile_name=profile_name)
            if record.public_score is not None or record.leaderboard_rank is not None
        ]
        best_public = max(
            (record.public_score for record in records if record.public_score is not None),
            default=None,
        )
        best_rank = min(
            (
                record.leaderboard_rank
                for record in records
                if record.leaderboard_rank is not None
            ),
            default=None,
        )
        return {
            "record_count": len(records),
            "best_public_score": best_public,
            "best_leaderboard_rank": best_rank,
        }
