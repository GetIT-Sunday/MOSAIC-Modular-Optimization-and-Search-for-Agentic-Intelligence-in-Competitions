from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


VALID_DECISIONS = {"continue", "rerun", "patch_prompt", "stop"}


@dataclass(frozen=True)
class HumanGateDecision:
    decision: str
    notes: str
    source_path: str

    @property
    def is_intervention(self) -> bool:
        return self.decision != "continue" or bool(self.notes.strip())

    def to_dict(self) -> dict:
        return asdict(self)


class HumanGate:
    @staticmethod
    def parse(path: Path) -> HumanGateDecision:
        if not path.exists():
            return HumanGateDecision("continue", "", str(path))

        text = path.read_text(encoding="utf-8", errors="ignore")
        decision = "continue"
        notes_lines: List[str] = []
        in_notes = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.lower().startswith("decision:"):
                decision = line.split(":", 1)[1].strip() or "continue"
                in_notes = False
                continue
            if line.lower().startswith("notes:"):
                in_notes = True
                remainder = raw_line.split(":", 1)[1].strip()
                if remainder:
                    notes_lines.append(remainder)
                continue
            if in_notes:
                notes_lines.append(raw_line.rstrip())

        if decision not in VALID_DECISIONS:
            decision = "continue"
            notes_lines.insert(0, f"Invalid human gate decision was ignored.")

        return HumanGateDecision(
            decision=decision,
            notes="\n".join(notes_lines).strip(),
            source_path=str(path),
        )

    @staticmethod
    def collect(paths: Iterable[Path]) -> List[HumanGateDecision]:
        return [HumanGate.parse(path) for path in paths]
