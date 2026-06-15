from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROFILE_DIR = Path(__file__).resolve().parents[1] / "domain_profiles"


@dataclass(frozen=True)
class CompetitionProfile:
    name: str
    description: str
    task_families: List[str]
    raw: Dict[str, Any]

    @property
    def phases(self) -> List[str]:
        return list(self.raw.get("phases", []))

    @property
    def baseline_ladder(self) -> List[str]:
        return list(self.raw.get("baseline_ladder", []))


def load_profile(name: str, profile_dir: Path = PROFILE_DIR) -> CompetitionProfile:
    path = profile_dir / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_profile_names(profile_dir))
        raise FileNotFoundError(f"Unknown profile '{name}'. Available profiles: {available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return CompetitionProfile(
        name=data["name"],
        description=data.get("description", ""),
        task_families=list(data.get("task_families", [])),
        raw=data,
    )


def load_profiles(profile_dir: Path = PROFILE_DIR) -> List[CompetitionProfile]:
    return [load_profile(path.stem, profile_dir) for path in sorted(profile_dir.glob("*.json"))]


def list_profile_names(profile_dir: Path = PROFILE_DIR) -> List[str]:
    return sorted(path.stem for path in profile_dir.glob("*.json"))


def required_file_groups(profile: CompetitionProfile) -> List[List[str]]:
    input_files = profile.raw.get("input_files", {})
    groups: List[List[str]] = []
    required = input_files.get("required", [])
    if required:
        groups.append(list(required))
    for group in input_files.get("required_any", []):
        groups.append(list(group))
    return groups


def profile_file_match_score(profile: CompetitionProfile, files: Iterable[str]) -> float:
    file_set = {Path(file).name for file in files}
    groups = required_file_groups(profile)
    if not groups:
        return 0.0
    group_scores = []
    for group in groups:
        if not group:
            continue
        matched = sum(1 for file in group if file in file_set)
        group_scores.append(matched / len(group))
    return max(group_scores) if group_scores else 0.0
