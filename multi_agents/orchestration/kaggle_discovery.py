from __future__ import annotations

import csv
import json
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional

from .run_ledger import RunLedger


CommandRunner = Callable[[List[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class KaggleCompetition:
    ref: str
    title: str = ""
    deadline: str = ""
    category: str = ""
    reward: str = ""
    team_count: str = ""
    user_has_entered: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KaggleDiscoveryResult:
    status: str
    cache_path: Path
    competitions: List[KaggleCompetition]
    command: List[str]
    issues: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["cache_path"] = str(self.cache_path)
        return payload


@dataclass(frozen=True)
class KaggleCompetitionSelectionResult:
    status: str
    competition_slug: str
    competition_dir: Path
    intake_path: Path
    files: List[dict]
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["competition_dir"] = str(self.competition_dir)
        payload["intake_path"] = str(self.intake_path)
        return payload


class KaggleDiscoveryAgent:
    """Discovers and selects Kaggle competitions through the official Kaggle CLI."""

    def __init__(
        self,
        competition_root: Path,
        runner: Optional[CommandRunner] = None,
    ):
        self.competition_root = competition_root.resolve()
        self.competition_root.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.competition_root / "kaggle_competitions_cache.json"
        self.uses_default_runner = runner is None
        self.runner = runner or self._run

    def discover(
        self,
        *,
        group: str = "general",
        category: str = "all",
        sort_by: str = "recentlyCreated",
        search: str = "",
        page: int = 1,
    ) -> KaggleDiscoveryResult:
        command = [
            "kaggle",
            "competitions",
            "list",
            "--group",
            group,
            "--category",
            category,
            "--sort-by",
            sort_by,
            "--page",
            str(page),
            "--csv",
        ]
        if search:
            command.extend(["--search", search])

        issues: List[str] = []
        competitions: List[KaggleCompetition] = []
        status = "pass"
        if shutil.which("kaggle") is None and self.uses_default_runner:
            status = "needs_kaggle_cli"
            issues.append("kaggle CLI is not installed or not on PATH.")
        else:
            completed = self.runner(command)
            if completed.returncode != 0:
                error = self._command_error(completed, "kaggle competitions list failed.")
                status = self._classify_error(error, default="needs_review")
                issues.append(error)
            else:
                competitions = self._parse_competitions_csv(completed.stdout)
                if not competitions:
                    status = "empty"
                    issues.append("No competitions returned by Kaggle CLI.")

        result = KaggleDiscoveryResult(
            status=status,
            cache_path=self.cache_path,
            competitions=competitions,
            command=command,
            issues=issues,
        )
        self.cache_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result

    def select(
        self,
        competition_slug: str,
        *,
        download: bool = False,
    ) -> KaggleCompetitionSelectionResult:
        safe_slug = self._safe_slug(competition_slug)
        competition_dir = self.competition_root / safe_slug
        competition_dir.mkdir(parents=True, exist_ok=True)

        issues: List[str] = []
        files: List[dict] = []
        status = "selected"
        cache = self._read_cache()
        selected = self._find_cached_competition(cache, safe_slug)
        if shutil.which("kaggle") is None and self.uses_default_runner:
            status = "needs_kaggle_cli"
            issues.append("kaggle CLI is not installed or not on PATH.")
        else:
            files_command = ["kaggle", "competitions", "files", safe_slug, "--csv"]
            completed = self.runner(files_command)
            if completed.returncode != 0:
                error = self._command_error(
                    completed,
                    "Cannot list competition files. Join the competition and accept rules on Kaggle first.",
                )
                status = self._classify_error(error, default="needs_rules_or_auth")
                issues.append(error)
            else:
                files = self._parse_csv_rows(completed.stdout)
            if download and status == "selected":
                intake_path = competition_dir / "competition_intake.json"
                intake_path.write_text(
                    json.dumps(
                        self._selection_payload(
                            status="downloading",
                            safe_slug=safe_slug,
                            competition_dir=competition_dir,
                            selected=selected,
                            files=files,
                            download=download,
                            issues=[],
                        ),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                RunLedger(competition_dir).write_html()
                download_command = [
                    "kaggle",
                    "competitions",
                    "download",
                    safe_slug,
                    "-p",
                    str(competition_dir),
                ]
                downloaded = self.runner(download_command)
                if downloaded.returncode != 0:
                    error = self._command_error(downloaded, "Competition download failed.")
                    status = self._classify_error(error, default="download_failed")
                    issues.append(error)
                else:
                    extract_issues = self._extract_downloaded_archives(competition_dir)
                    if extract_issues:
                        status = "extract_failed"
                        issues.extend(extract_issues)

        payload = self._selection_payload(
            status=status,
            safe_slug=safe_slug,
            competition_dir=competition_dir,
            selected=selected,
            files=files,
            download=download,
            issues=issues,
        )
        intake_path = competition_dir / "competition_intake.json"
        intake_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        RunLedger(competition_dir).write_html()
        return KaggleCompetitionSelectionResult(
            status=status,
            competition_slug=safe_slug,
            competition_dir=competition_dir,
            intake_path=intake_path,
            files=files,
            issues=issues,
        )

    def _selection_payload(
        self,
        *,
        status: str,
        safe_slug: str,
        competition_dir: Path,
        selected: dict,
        files: List[dict],
        download: bool,
        issues: List[str],
    ) -> dict:
        return {
            "status": status,
            "competition_slug": safe_slug,
            "competition_url": f"https://www.kaggle.com/competitions/{safe_slug}",
            "competition_dir": str(competition_dir),
            "source": "kaggle_cli",
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "cached_competition": selected,
            "files": files,
            "download_requested": download,
            "next_step": self._next_step_for_status(status),
            "recommended_commands": [
                f"open https://www.kaggle.com/competitions/{safe_slug}",
                f"kaggle competitions files {safe_slug} --csv",
                f"kaggle competitions download {safe_slug} -p {competition_dir}",
                f"python framework.py --competition {safe_slug} --task-card-mode",
                f"python framework.py --competition {safe_slug} --run-baselines",
            ],
            "issues": issues,
        }

    def _run(self, command: List[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def _command_error(self, completed: subprocess.CompletedProcess[str], fallback: str) -> str:
        return (completed.stderr or completed.stdout or "").strip() or fallback

    def _classify_error(self, message: str, *, default: str) -> str:
        lowered = message.lower()
        if "authentication required" in lowered or "unauthorized" in lowered or "api token" in lowered:
            return "auth_missing"
        if ("accept" in lowered and "rules" in lowered) or "403" in lowered or "forbidden" in lowered:
            return "rules_not_accepted"
        if "participate" in lowered or "join" in lowered:
            return "rules_not_accepted"
        if "404" in lowered or "not found" in lowered:
            return "competition_not_found"
        return default

    def _next_step_for_status(self, status: str) -> str:
        if status in {"needs_rules_or_auth", "rules_not_accepted"}:
            return "accept_rules_on_kaggle"
        if status == "auth_missing":
            return "configure_kaggle_auth"
        if status == "needs_kaggle_cli":
            return "install_kaggle_cli"
        if status == "download_failed":
            return "inspect_download_error"
        if status == "extract_failed":
            return "inspect_download_archive"
        return "run_competition_intake_or_download"

    def _extract_downloaded_archives(self, competition_dir: Path) -> List[str]:
        issues: List[str] = []
        for archive_path in sorted(competition_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(competition_dir)
            except zipfile.BadZipFile:
                issues.append(f"Downloaded archive is not a valid zip file: {archive_path.name}")
            except OSError as exc:
                issues.append(f"Failed to extract {archive_path.name}: {exc}")
        return issues

    def _parse_competitions_csv(self, text: str) -> List[KaggleCompetition]:
        rows = self._parse_csv_rows(text)
        competitions = []
        for row in rows:
            normalized = {key.strip(): value for key, value in row.items()}
            ref = self._first(normalized, ["ref", "Ref", "competition", "Competition"])
            title = self._first(normalized, ["title", "Title"])
            if not ref:
                continue
            normalized["source_ref"] = ref
            ref = self._safe_slug(ref)
            competitions.append(
                KaggleCompetition(
                    ref=ref,
                    title=title,
                    deadline=self._first(normalized, ["deadline", "Deadline"]),
                    category=self._first(normalized, ["category", "Category"]),
                    reward=self._first(normalized, ["reward", "Reward"]),
                    team_count=self._first(normalized, ["teamCount", "team_count", "TeamCount", "Teams"]),
                    user_has_entered=self._first(normalized, ["userHasEntered", "user_has_entered"]),
                    raw=normalized,
                )
            )
        return competitions

    def _parse_csv_rows(self, text: str) -> List[dict]:
        cleaned = "\n".join(line for line in text.splitlines() if line.strip())
        if not cleaned:
            return []
        return list(csv.DictReader(StringIO(cleaned)))

    def _read_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _find_cached_competition(self, cache: dict, slug: str) -> dict:
        for item in cache.get("competitions", []):
            if item.get("ref") == slug:
                return item
        return {}

    def _safe_slug(self, slug: str) -> str:
        cleaned = slug.strip().rstrip("/").split("/")[-1]
        if not cleaned or any(char in cleaned for char in "\\:"):
            raise ValueError(f"Invalid Kaggle competition slug: {slug!r}")
        if cleaned in {".", ".."} or "/" in cleaned:
            raise ValueError(f"Invalid Kaggle competition slug: {slug!r}")
        return cleaned

    def _first(self, row: dict, keys: List[str]) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None:
                return str(value)
        return ""
