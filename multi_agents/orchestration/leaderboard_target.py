from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .run_ledger import RunLedger


CommandRunner = Callable[[List[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class LeaderboardTargetResult:
    status: str
    target_path: Path
    raw_path: Path


class LeaderboardTargetAgent:
    """Fetch and summarize Kaggle leaderboard targets for rank-oriented planning."""

    def __init__(
        self,
        competition_dir: Path,
        runner: Optional[CommandRunner] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.runner = runner or self._run
        self.uses_default_runner = runner is None
        self.ledger = RunLedger(self.competition_dir)

    def run(self, *, page_size: int = 200) -> LeaderboardTargetResult:
        manifest = self._read_json(self.competition_dir / "data_manifest.json")
        metric_spec = self._read_json(self.competition_dir / "metric_spec.json")
        baseline_review = self._read_json(self.competition_dir / "baseline_review.json")
        champion_selection = self._read_json(self.competition_dir / "champion_selection.json")
        enhancement_review = self._read_json(self.competition_dir / "enhancement_review.json")
        slug = str(manifest.get("competition_name") or self.competition_dir.name)
        metric_name = self._metric_name(metric_spec, manifest, baseline_review)
        higher_is_better = not self._lower_is_better(metric_name)
        raw_path = self.competition_dir / "leaderboard_target_raw.csv"
        target_path = self.competition_dir / "leaderboard_target.json"

        issues: List[str] = []
        rows: List[Dict[str, str]] = []
        status = "completed"
        command = [
            "kaggle",
            "competitions",
            "leaderboard",
            slug,
            "--show",
            "--csv",
            "--page-size",
            str(page_size),
        ]
        if shutil.which("kaggle") is None and self.uses_default_runner:
            status = "needs_kaggle_cli"
            issues.append("kaggle CLI is not installed or not on PATH.")
            raw_text = ""
        else:
            completed = self.runner(command)
            raw_text = completed.stdout or ""
            if completed.returncode != 0:
                status = "needs_review"
                issues.append((completed.stderr or completed.stdout or "Kaggle leaderboard command failed.").strip())
            else:
                rows = self._parse_rows(raw_text)
                if not rows:
                    status = "empty"
                    issues.append("Kaggle leaderboard command returned no parseable rows.")
                elif not self._scores(rows):
                    status = "empty"
                    issues.append("Kaggle leaderboard command returned rows but no parseable score column.")
        raw_path.write_text(raw_text, encoding="utf-8")

        scores = self._scores(rows)
        best_local = self._best_local_score(baseline_review, champion_selection, enhancement_review)
        target = self._target_payload(
            status=status,
            slug=slug,
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            command=command,
            rows=rows,
            scores=scores,
            best_local=best_local,
            issues=issues,
            page_size=page_size,
        )
        target_path.write_text(json.dumps(target, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_ledger(target, target_path, raw_path)
        return LeaderboardTargetResult(status=status, target_path=target_path, raw_path=raw_path)

    def _target_payload(
        self,
        *,
        status: str,
        slug: str,
        metric_name: str,
        higher_is_better: bool,
        command: List[str],
        rows: List[Dict[str, str]],
        scores: List[float],
        best_local: Dict[str, Any],
        issues: List[str],
        page_size: int,
    ) -> Dict[str, Any]:
        ranked_scores = sorted(scores, reverse=higher_is_better)
        visible_count = len(ranked_scores)
        top_score = ranked_scores[0] if ranked_scores else None
        top_10_score = ranked_scores[min(9, visible_count - 1)] if visible_count else None
        silver_index = max(0, min(visible_count - 1, int(visible_count * 0.05) - 1)) if visible_count else None
        estimated_silver = ranked_scores[silver_index] if silver_index is not None else None
        current_score = best_local.get("local_score")
        gap_to_top = self._gap(current_score, top_score, higher_is_better)
        gap_to_silver = self._gap(current_score, estimated_silver, higher_is_better)
        confidence = "medium" if visible_count >= 50 else "low"
        if status != "completed":
            confidence = "none"
        return {
            "competition_name": slug,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "kaggle_cli",
            "command": command,
            "page_size": page_size,
            "visible_entry_count": visible_count,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "top_score": top_score,
            "top_10_score": top_10_score,
            "estimated_silver_score": estimated_silver,
            "estimated_silver_method": "visible_top_5_percent_proxy",
            "target_policy": "silver_or_better",
            "confidence": confidence,
            "current_best_local": best_local,
            "current_best_local_score": current_score,
            "gap_to_top": gap_to_top,
            "gap_to_silver": gap_to_silver,
            "next_decision": self._next_decision(status, gap_to_silver, gap_to_top),
            "top_entries": rows[:10],
            "issues": issues,
        }

    def _write_ledger(self, target: Dict[str, Any], target_path: Path, raw_path: Path) -> None:
        status = "pass" if target.get("status") == "completed" else "needs_review"
        self.ledger.create_entry(
            task_id="leaderboard_target",
            agent="leaderboard_target_agent",
            title="Fetch leaderboard target snapshot",
            status=status,
            input_payload={"competition_name": target.get("competition_name"), "source": "kaggle_cli"},
            prompt="Fetch Kaggle leaderboard top entries and estimate target gaps for Brain planning.",
            scorecard={
                "agent": "leaderboard_target_agent",
                "task_id": "leaderboard_target",
                "status": status,
                "scores": {
                    "visible_entry_count": target.get("visible_entry_count", 0),
                    "top_score": target.get("top_score", "n/a"),
                    "estimated_silver_score": target.get("estimated_silver_score", "n/a"),
                    "gap_to_silver": target.get("gap_to_silver", "n/a"),
                },
                "metric_name": target.get("metric_name"),
                "local_score": target.get("current_best_local_score"),
                "issues": target.get("issues", []),
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={"leaderboard_target": target_path, "leaderboard_target_raw": raw_path},
        )

    def _scores(self, rows: List[Dict[str, str]]) -> List[float]:
        scores: List[float] = []
        for row in rows:
            value = self._first(row, ["score", "Score", "publicScore", "PublicScore", "public_score"])
            number = self._number(value)
            if number is not None:
                scores.append(number)
        return scores

    def _parse_rows(self, text: str) -> List[Dict[str, str]]:
        lines = [line for line in text.splitlines() if line.strip()]
        header_index = 0
        for index, line in enumerate(lines):
            columns = [part.strip().lower() for part in line.split(",")]
            if "score" in columns:
                header_index = index
                break
        cleaned = "\n".join(lines[header_index:])
        if not cleaned:
            return []
        rows = list(csv.DictReader(StringIO(cleaned)))
        return [{str(key): str(value) for key, value in row.items()} for row in rows]

    def _best_local_score(self, *reports: Dict[str, Any]) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        for report in reports:
            if not isinstance(report, dict):
                continue
            if isinstance(report.get("best_baseline"), dict):
                candidates.append(report["best_baseline"])
            if isinstance(report.get("champion"), dict):
                candidates.append(report["champion"])
            if isinstance(report.get("local_score"), (int, float)):
                candidates.append(report)
        best: Dict[str, Any] = {}
        for candidate in candidates:
            score = candidate.get("local_score")
            if not isinstance(score, (int, float)):
                continue
            if not isinstance(best.get("local_score"), (int, float)) or score > best["local_score"]:
                best = {
                    "task_id": candidate.get("task_id") or candidate.get("source_id") or "local_candidate",
                    "metric_name": candidate.get("metric_name"),
                    "local_score": score,
                    "status": candidate.get("status"),
                }
        return best

    @staticmethod
    def _metric_name(*reports: Dict[str, Any]) -> str:
        for report in reports:
            if not isinstance(report, dict):
                continue
            if report.get("metric_name"):
                return str(report["metric_name"])
            candidates = report.get("metric_candidates")
            if isinstance(candidates, list) and candidates:
                return str(candidates[0])
        return "metric"

    @staticmethod
    def _lower_is_better(metric_name: str) -> bool:
        return metric_name.lower() in {"rmse", "rmsle", "mae", "mse", "log_loss", "loss"}

    @staticmethod
    def _gap(current: Any, target: Any, higher_is_better: bool) -> Optional[float]:
        if not isinstance(current, (int, float)) or not isinstance(target, (int, float)):
            return None
        return float(target - current if higher_is_better else current - target)

    @staticmethod
    def _next_decision(status: str, gap_to_silver: Optional[float], gap_to_top: Optional[float]) -> str:
        if status != "completed":
            return "leaderboard_target_unavailable"
        if gap_to_silver is None:
            return "baseline_validation"
        if gap_to_silver > 0:
            return "gap_closing"
        if gap_to_top is not None and gap_to_top > 0:
            return "silver_proxy_met_continue_top_gap"
        return "leaderboard_target_met"

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _first(row: Dict[str, str], names: List[str]) -> str:
        for name in names:
            if name in row:
                return row[name]
        return ""

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _run(self, command: List[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, check=False)
