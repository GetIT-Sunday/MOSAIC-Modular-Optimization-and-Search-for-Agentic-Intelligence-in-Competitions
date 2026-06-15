from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class ManualSubmissionPackageResult:
    status: str
    manifest_path: Path
    checklist_path: Path
    package_dir: Path


class ManualSubmissionPackage:
    """Package the exact file and instructions needed for a manual Kaggle upload."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def build(self, *, submission_target: str = "recommended") -> ManualSubmissionPackageResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")

        pipeline = self._read_json(self.competition_dir / "post_experiment_pipeline.json")
        handoff = self._read_json(self.competition_dir / "submit_decision_handoff.json")
        workflow = self._read_json(self.competition_dir / "post_submit_workflow.json")
        feedback_template_path = self.competition_dir / "leaderboard_feedback_input_template.json"
        feedback_template = self._read_json(feedback_template_path)
        promoted = pipeline.get("promoted_candidate") or handoff.get("candidate") or workflow.get("candidate") or {}
        submission_path = self._resolve_competition_path(
            handoff.get("submission_path") or workflow.get("submission_path")
        )
        package_dir = self.competition_dir / "manual_submission_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        packaged_submission = package_dir / "submission.csv"
        packaged_feedback_template = package_dir / "leaderboard_feedback_input_template.json"
        manifest_path = package_dir / "manifest.json"
        checklist_path = package_dir / "README.md"

        issues = []
        warnings = []
        warnings.extend(pipeline.get("warnings") or [])
        warnings.extend(handoff.get("warnings") or [])
        warnings.extend(workflow.get("warnings") or [])

        if pipeline.get("status") != "pass":
            issues.append("post_experiment_pipeline.json is not pass.")
        if handoff.get("status") != "ready_for_human_submit_decision":
            issues.append("submit_decision_handoff.json is not ready_for_human_submit_decision.")
        if workflow.get("status") != "ready_for_manual_submit":
            issues.append("post_submit_workflow.json is not ready_for_manual_submit.")
        if handoff.get("submission_target") != submission_target:
            issues.append("submit_decision_handoff target does not match requested target.")
        if workflow.get("submission_target") != submission_target:
            issues.append("post_submit_workflow target does not match requested target.")
        submission_is_file = submission_path.exists() and submission_path.is_file()
        if not submission_is_file:
            issues.append("Submission file is missing.")
        if not feedback_template:
            issues.append("leaderboard_feedback_input_template.json is missing.")
        candidate_task_id = promoted.get("task_id")
        if feedback_template.get("candidate_task_id") != candidate_task_id:
            issues.append("Feedback template candidate_task_id does not match promoted candidate.")

        if submission_is_file:
            shutil.copyfile(submission_path, packaged_submission)
        if feedback_template_path.exists():
            shutil.copyfile(feedback_template_path, packaged_feedback_template)

        candidate_risk_summary = self._candidate_risk_summary(promoted)
        submission_file_summary = self._file_summary(packaged_submission)
        feedback_template_file_summary = self._file_summary(packaged_feedback_template)
        status = "ready_for_manual_upload" if not issues else "needs_review"
        package_feedback_template_arg = str(packaged_feedback_template.relative_to(self.competition_dir))
        feedback_loop_command = self._feedback_loop_command(
            competition_name=pipeline.get("competition_name", self.competition_dir.name),
            feedback_template=package_feedback_template_arg,
        )
        feedback_fill_command = self._feedback_fill_command(
            competition_name=pipeline.get("competition_name", self.competition_dir.name),
            feedback_template=package_feedback_template_arg,
        )
        verify_package_command = self._verify_package_command(
            competition_name=pipeline.get("competition_name", self.competition_dir.name),
        )
        manifest = {
            "competition_name": pipeline.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": "manual_upload_package_ready" if status == "ready_for_manual_upload" else "manual_upload_package_blocked",
            "submission_target": submission_target,
            "candidate": promoted,
            "source_submission_path": str(submission_path),
            "packaged_submission_path": str(packaged_submission),
            "packaged_submission_relative_path": str(packaged_submission.relative_to(self.competition_dir)),
            "feedback_template_path": str(packaged_feedback_template),
            "feedback_template_relative_path": str(packaged_feedback_template.relative_to(self.competition_dir)),
            "candidate_risk": candidate_risk_summary,
            "submission_file": submission_file_summary,
            "feedback_template_file": feedback_template_file_summary,
            "upload_file_checks": [
                "Confirm manual_submission_package/submission.csv SHA-256 before upload.",
                "Confirm row_count and columns match the Kaggle sample submission contract.",
                "Fill the packaged feedback template only after Kaggle returns a real public score, rank, or submission_id.",
            ],
            "post_submit_workflow_path": str(self.competition_dir / "post_submit_workflow.md"),
            "submit_decision_handoff_path": str(self.competition_dir / "submit_decision_handoff.md"),
            "required_after_upload": [
                "public_score or leaderboard_rank or submission_id",
                "candidate_task_id",
                "submission_target",
            ],
            "feedback_loop_command": feedback_loop_command,
            "feedback_fill_command": feedback_fill_command,
            "verify_package_command": verify_package_command,
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "next_action": (
                "Upload manual_submission_package/submission.csv, then fill the packaged feedback template and run the feedback loop command."
                if status == "ready_for_manual_upload"
                else "Fix package issues, then rebuild the manual submission package."
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        checklist_path.write_text(self._render_readme(manifest), encoding="utf-8")

        ledger_entry = self.ledger.create_entry(
            task_id="manual_submission_package",
            agent="manual_submission_package",
            title="Package manual Kaggle upload handoff",
            status=status,
            input_payload=manifest,
            prompt="Package the exact submission file, feedback template, and upload checklist for a human Kaggle upload.",
            scorecard={
                "agent": "manual_submission_package",
                "task_id": "manual_submission_package",
                "status": status,
                "scores": {
                    "submission_target": submission_target,
                    "candidate_task_id": candidate_task_id or "n/a",
                    "submission_file_exists": submission_is_file,
                    "submission_sha256": submission_file_summary.get("sha256", "n/a"),
                    "submission_row_count": submission_file_summary.get("row_count"),
                    "candidate_risk_level": candidate_risk_summary.get("risk_level", "unknown"),
                    "feedback_template_ready": bool(feedback_template),
                },
                "metric_name": promoted.get("metric_name"),
                "local_score": promoted.get("local_score"),
                "issues": issues + list(dict.fromkeys(warnings)),
                "recommended_human_action": "continue" if status == "ready_for_manual_upload" else "patch_prompt",
            },
            artifacts={
                "manual_submission_manifest": manifest_path,
                "manual_submission_readme": checklist_path,
                "submission": packaged_submission,
                "feedback_template": packaged_feedback_template,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=manifest["competition_name"],
                profile_name="tabular_classic",
                task_id="manual_submission_package",
                status=status,
                metric_name=promoted.get("metric_name"),
                local_score=promoted.get("local_score"),
                submission_path=str(packaged_submission) if packaged_submission.exists() else None,
                brain_review_path=str(manifest_path),
                artifacts=[str(manifest_path), str(checklist_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=manifest["next_action"],
            )
        )
        return ManualSubmissionPackageResult(
            status=status,
            manifest_path=manifest_path,
            checklist_path=checklist_path,
            package_dir=package_dir,
        )

    @staticmethod
    def _feedback_loop_command(*, competition_name: str, feedback_template: str) -> str:
        return (
            "python framework.py "
            f"--competition {competition_name} "
            "--leaderboard-feedback-from-template "
            f"--feedback-template {feedback_template}"
        )

    @staticmethod
    def _feedback_fill_command(*, competition_name: str, feedback_template: str) -> str:
        return (
            "python framework.py "
            f"--competition {competition_name} "
            "--fill-leaderboard-feedback-template "
            "--public-score <PUBLIC_SCORE> "
            f"--feedback-template {feedback_template} "
            "--run-filled-feedback-loop"
        )

    @staticmethod
    def _verify_package_command(*, competition_name: str) -> str:
        return (
            "python framework.py "
            f"--competition {competition_name} "
            "--verify-manual-submission-package"
        )

    @staticmethod
    def _render_readme(manifest: Dict[str, Any]) -> str:
        candidate = manifest.get("candidate") or {}
        submission_file = manifest.get("submission_file") or {}
        feedback_template_file = manifest.get("feedback_template_file") or {}
        candidate_risk = manifest.get("candidate_risk") or {}
        columns = ", ".join(submission_file.get("columns") or [])
        return f"""# Manual Submission Package

Status: {manifest.get("status")}
Submission target: {manifest.get("submission_target")}
Candidate task: {candidate.get("task_id", "unknown")}
Metric: {candidate.get("metric_name", "unknown")}
Local score: {candidate.get("local_score", "n/a")}

## Before Upload

Run this verifier and continue only if it reports `pass`:

```bash
{manifest.get("verify_package_command") or "python framework.py --competition <competition> --verify-manual-submission-package"}
```

## Upload

Upload this file to Kaggle:

`{manifest.get("packaged_submission_relative_path") or "manual_submission_package/submission.csv"}`

Absolute path recorded when the package was built:

`{manifest.get("packaged_submission_path")}`

## File Checks

Submission SHA-256: `{submission_file.get("sha256", "missing")}`
Submission rows: {submission_file.get("row_count", "unknown")}
Submission columns: {columns or "unknown"}
Submission size bytes: {submission_file.get("size_bytes", "unknown")}

Feedback template SHA-256: `{feedback_template_file.get("sha256", "missing")}`

## Candidate Risk

Risk level: {candidate_risk.get("risk_level", "unknown")}
Risk source: {candidate_risk.get("source", "unknown")}
Risk issues: {", ".join(candidate_risk.get("issues") or []) or "None"}

## After Upload

Preferred command after Kaggle returns a public score:

```bash
{manifest.get("feedback_fill_command") or "python framework.py --competition <competition> --fill-leaderboard-feedback-template --public-score <PUBLIC_SCORE> --run-filled-feedback-loop"}
```

Optional additions when Kaggle shows them:

```bash
--leaderboard-rank <LEADERBOARD_RANK> --submission-id <SUBMISSION_ID>
```

Or fill this feedback template manually:

`{manifest.get("feedback_template_relative_path") or "manual_submission_package/leaderboard_feedback_input_template.json"}`

Then run:

```bash
{manifest.get("feedback_loop_command") or "python framework.py --competition <competition> --leaderboard-feedback-from-template"}
```

Required fields:

- public_score or leaderboard_rank or submission_id
- candidate_task_id
- submission_target

Next action: {manifest.get("next_action")}
"""

    def _candidate_risk_summary(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        risk_audit = self._read_risk_audit(candidate)
        if risk_audit:
            return {
                "source": "risk_audit",
                "path": risk_audit.get("_path"),
                "status": risk_audit.get("status"),
                "risk_level": risk_audit.get("risk_level", "unknown"),
                "risk_points": risk_audit.get("risk_points"),
                "fold_stability_score": risk_audit.get("fold_stability_score"),
                "max_fold_std": risk_audit.get("max_fold_std"),
                "max_model_correlation": risk_audit.get("max_model_correlation"),
                "issues": risk_audit.get("issues") or [],
                "recommendation": risk_audit.get("recommendation"),
            }
        return {
            "source": "candidate",
            "path": candidate.get("risk_audit_path"),
            "status": None,
            "risk_level": candidate.get("risk_level", "unknown"),
            "risk_points": None,
            "fold_stability_score": None,
            "max_fold_std": None,
            "max_model_correlation": None,
            "issues": candidate.get("risk_issues") or [],
            "recommendation": None,
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
        summary: Dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            return summary
        content = path.read_bytes()
        summary.update(
            {
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
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

    def _resolve_competition_path(self, value: Optional[str]) -> Path:
        if not value:
            return self.competition_dir / "__missing_submission__"
        path = Path(value)
        if path.exists() and path.is_file():
            return path
        local_path = self.competition_dir / path.name
        if local_path.exists() and local_path.is_file():
            return local_path
        return path

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
