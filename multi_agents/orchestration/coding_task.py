from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    title: str
    profile_name: str
    objective: str
    inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt(self) -> str:
        sections = [
            f"Task ID: {self.task_id}",
            f"Title: {self.title}",
            f"Profile: {self.profile_name}",
            f"Objective: {self.objective}",
            "Inputs:\n" + "\n".join(f"- {item}" for item in self.inputs),
            "Expected outputs:\n" + "\n".join(f"- {item}" for item in self.expected_outputs),
            "Validation checks:\n" + "\n".join(f"- {item}" for item in self.validation_checks),
            "Constraints:\n" + "\n".join(f"- {item}" for item in self.constraints),
        ]
        if self.context:
            sections.append(
                "Context:\n"
                + "\n".join(f"- {key}: {value}" for key, value in self.context.items())
            )
        return "\n\n".join(sections)
