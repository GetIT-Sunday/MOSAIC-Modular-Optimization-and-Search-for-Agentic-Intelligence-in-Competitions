from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .manual_submit_readiness import ManualSubmitReadinessChecker
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class PostSubmitWorkflowResult:
    status: str
    report_path: Path
    checklist_path: Path
    feedback_input_template_path: Path


class PostSubmitWorkflow:
    """Create the standard manual-submit to leaderboard-feedback handoff."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def run(
        self,
        *,
        submission_target: str = "champion",
        refresh: bool = True,
    ) -> PostSubmitWorkflowResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")

        readiness_result = ManualSubmitReadinessChecker(
            self.competition_dir,
            memory=self.memory,
        ).run(refresh=refresh, submission_target=submission_target)
        readiness = self._read_json(readiness_result.report_path)
        candidate = readiness.get("candidate") or {}
        submission_path = self._resolve_competition_path(readiness.get("submission_path"))
        issues = []
        warnings = list(readiness.get("warnings", []))

        if not readiness.get("manual_submission_ready"):
            issues.append("manual_submit_readiness.json is not ready for manual submission.")
        if not candidate.get("task_id"):
            issues.append("Submission candidate task_id is unavailable.")
        if not submission_path.exists():
            issues.append("Submission file is missing.")

        status = "ready_for_manual_submit" if not issues else "needs_review"
        competition_name = readiness.get("competition_name", self.competition_dir.name)
        feedback_loop_command = self._feedback_loop_command(
            competition_name=competition_name,
            submission_target=submission_target,
        )
        submission_file = self._file_summary(submission_path)
        candidate_risk = self._candidate_risk_summary(candidate)
        record_feedback_template = {
            "public_score": "<PUBLIC_SCORE>",
            "leaderboard_rank": None,
            "submission_id": "<SUBMISSION_ID_OR_NULL>",
            "submission_target": submission_target,
            "candidate_task_id": candidate.get("task_id"),
            "expected_submission_sha256": submission_file.get("sha256"),
            "expected_submission_rows": submission_file.get("row_count"),
            "expected_submission_columns": submission_file.get("columns"),
            "candidate_risk_level": candidate_risk.get("risk_level", "unknown"),
            "source": "manual",
            "notes": "post_submit_feedback",
        }
        report = {
            "competition_name": competition_name,
            "status": status,
            "decision": "manual_submit_then_record_feedback" if status == "ready_for_manual_submit" else "fix_submit_readiness",
            "submission_target": submission_target,
            "candidate": candidate,
            "submission_path": str(submission_path),
            "submission_file": submission_file,
            "candidate_risk": candidate_risk,
            "readiness_path": str(readiness_result.report_path),
            "manual_submission_ready": readiness.get("manual_submission_ready"),
            "api_submission_review_ready": readiness.get("api_submission_review_ready"),
            "confirmed_submit_ready": readiness.get("confirmed_submit_ready"),
            "feedback_loop_command_template": feedback_loop_command,
            "record_feedback_template": record_feedback_template,
            "feedback_input_template_path": str(self.competition_dir / "leaderboard_feedback_input_template.json"),
            "feedback_validation_rules": [
                "Replace every angle-bracket placeholder before recording feedback.",
                "Set unavailable optional fields to null, not placeholder text.",
                "leaderboard_rank must be a positive integer when provided.",
                "submission_target must match the uploaded submission target.",
                "candidate_task_id must match the uploaded candidate.",
                "expected_submission_sha256, rows, and columns must match the packaged upload file.",
                "candidate_risk_level must match the submit decision candidate risk.",
            ],
            "required_after_upload_fields": [
                "public_score or leaderboard_rank or submission_id",
                "submission_target",
                "candidate_task_id",
            ],
            "expected_feedback_artifacts": [
                "leaderboard_feedback.json",
                "leaderboard_gap_audit.json",
                "leaderboard_feedback_loop.json",
                "remote_brain_reply.md",
                "brain_review.json",
            ],
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "next_action": (
                "Upload the listed submission file, then run the feedback loop command with the observed public score."
                if status == "ready_for_manual_submit"
                else "Fix readiness issues, rerun --post-submit-workflow, then upload."
            ),
        }

        report_path = self.competition_dir / "post_submit_workflow.json"
        checklist_path = self.competition_dir / "post_submit_workflow.md"
        feedback_input_template_path = self.competition_dir / "leaderboard_feedback_input_template.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        checklist_path.write_text(self._render_checklist(report), encoding="utf-8")
        feedback_input_template_path.write_text(
            json.dumps(record_feedback_template, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ledger_entry = self.ledger.create_entry(
            task_id="post_submit_workflow",
            agent="post_submit_workflow",
            title="Prepare post-submit feedback workflow",
            status=status,
            input_payload=report,
            prompt=(
                "Create a human-readable handoff from manual Kaggle upload to leaderboard feedback loop, "
                "binding the feedback to the exact submitted target and candidate."
            ),
            scorecard={
                "agent": "post_submit_workflow",
                "task_id": "post_submit_workflow",
                "status": status,
                "scores": {
                    "submission_target": submission_target,
                    "candidate_task_id": candidate.get("task_id") or "n/a",
                    "manual_submission_ready": readiness.get("manual_submission_ready"),
                    "api_submission_review_ready": readiness.get("api_submission_review_ready"),
                    "confirmed_submit_ready": readiness.get("confirmed_submit_ready"),
                },
                "metric_name": candidate.get("metric_name"),
                "local_score": candidate.get("local_score"),
                "issues": issues + list(dict.fromkeys(warnings)),
                "recommended_human_action": "continue" if status == "ready_for_manual_submit" else "patch_prompt",
            },
            artifacts={
                "post_submit_workflow": report_path,
                "post_submit_checklist": checklist_path,
                "leaderboard_feedback_input_template": feedback_input_template_path,
                "manual_submit_readiness": readiness_result.report_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=competition_name,
                profile_name="tabular_classic",
                task_id="post_submit_workflow",
                status=status,
                metric_name=candidate.get("metric_name"),
                local_score=candidate.get("local_score"),
                submission_path=str(submission_path) if submission_path.exists() else None,
                brain_review_path=str(report_path),
                artifacts=[
                    str(report_path),
                    str(checklist_path),
                    str(feedback_input_template_path),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=report["next_action"],
            )
        )
        return PostSubmitWorkflowResult(
            status=status,
            report_path=report_path,
            checklist_path=checklist_path,
            feedback_input_template_path=feedback_input_template_path,
        )

    @staticmethod
    def _feedback_loop_command(*, competition_name: str, submission_target: str) -> str:
        return (
            "python framework.py "
            f"--competition {competition_name} "
            "--leaderboard-feedback-from-template"
        )

    def _resolve_competition_path(self, value: Optional[str]) -> Path:
        if not value:
            return Path("")
        path = Path(value)
        if path.exists():
            return path
        local_path = self.competition_dir / path.name
        if local_path.exists():
            return local_path
        return path

    def _candidate_risk_summary(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        risk_audit = self._read_risk_audit(candidate)
        if risk_audit:
            return {
                "source": "risk_audit",
                "path": risk_audit.get("_path"),
                "status": risk_audit.get("status"),
                "risk_level": risk_audit.get("risk_level", "unknown"),
                "issues": risk_audit.get("issues") or [],
            }
        return {
            "source": "candidate",
            "path": candidate.get("risk_audit_path"),
            "status": None,
            "risk_level": candidate.get("risk_level", "unknown"),
            "issues": candidate.get("risk_issues") or [],
        }

    def _read_risk_audit(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        paths = []
        raw_path = candidate.get("risk_audit_path")
        if raw_path:
            paths.append(Path(str(raw_path)))
        task_id = candidate.get("task_id")
        if task_id:
            paths.append(self.competition_dir / "experiments" / str(task_id) / "risk_audit.json")
        for path in paths:
            if path.exists() and path.is_file():
                audit = self._read_json(path)
                audit["_path"] = str(path)
                return audit
        return {}

    @staticmethod
    def _file_summary(path: Path) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists() or not path.is_file():
            return summary
        content = path.read_bytes()
        summary.update({"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            header = rows[0] if rows else []
            summary.update(
                {
                    "columns": header,
                    "column_count": len(header),
                    "row_count": max(len(rows) - 1, 0),
                }
            )
        return summary

    @staticmethod
    def _render_checklist(report: Dict[str, Any]) -> str:
        candidate = report.get("candidate") or {}
        submission_file = report.get("submission_file") or {}
        candidate_risk = report.get("candidate_risk") or {}
        command = report.get("feedback_loop_command_template", "")
        return f"""# Post-Submit Workflow

Competition: {report.get("competition_name")}
Status: {report.get("status")}
Submission target: {report.get("submission_target")}
Candidate task: {candidate.get("task_id", "unknown")}
Metric: {candidate.get("metric_name", "unknown")}
Local score: {candidate.get("local_score", "n/a")}
Submission SHA-256: {submission_file.get("sha256", "unknown")}
Candidate risk: {candidate_risk.get("risk_level", "unknown")}

## Submit

1. Upload this file to Kaggle:
   `{report.get("submission_path")}`
2. Confirm the uploaded file matches the submission target above.
3. Copy the public score, rank if visible, and submission id if visible.

## Record Feedback

Fill this JSON template first:

`{report.get("feedback_input_template_path")}`

Run this command from the AutoKaggle project root after replacing placeholders in the JSON template:

```bash
{command}
```

If rank or submission id is unavailable, set that JSON value to `null`. The public score alone is enough to close the feedback loop.

Do not submit angle-bracket placeholders such as `<PUBLIC_SCORE>` or `<SUBMISSION_ID_OR_NULL>`.

## Required Fields

- public_score or leaderboard_rank or submission_id
- submission_target
- candidate_task_id

## Expected Artifacts

- leaderboard_feedback.json
- leaderboard_gap_audit.json
- leaderboard_feedback_loop.json
- remote_brain_reply.md
- brain_review.json

Next action: {report.get("next_action")}
"""

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
