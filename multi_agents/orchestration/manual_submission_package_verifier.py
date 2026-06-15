from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .manual_submission_package import ManualSubmissionPackage
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class ManualSubmissionPackageVerificationResult:
    status: str
    report_path: Path
    experiment_roadmap_path: Optional[Path] = None


class ManualSubmissionPackageVerifier:
    """Verify the existing manual upload package without rebuilding it."""

    def __init__(
        self,
        competition_dir: Path,
        memory: Optional[CompetitionMemory] = None,
    ):
        self.competition_dir = competition_dir.resolve()
        self.memory = memory or CompetitionMemory()
        self.ledger = RunLedger(self.competition_dir)

    def verify(self) -> ManualSubmissionPackageVerificationResult:
        package_dir = self.competition_dir / "manual_submission_package"
        manifest_path = package_dir / "manifest.json"
        template_path = package_dir / "leaderboard_feedback_input_template.json"
        submission_path = package_dir / "submission.csv"
        manifest = self._read_json(manifest_path)
        template = self._read_json(template_path)
        actual_submission = ManualSubmissionPackage._file_summary(submission_path)
        actual_template = ManualSubmissionPackage._file_summary(template_path)
        issues: list[str] = []
        warnings: list[str] = []

        if not manifest:
            issues.append("manual_submission_package/manifest.json is missing or empty.")
        if not template:
            issues.append("manual_submission_package/leaderboard_feedback_input_template.json is missing or empty.")
        if not actual_submission.get("exists"):
            issues.append("manual_submission_package/submission.csv is missing.")

        manifest_submission = manifest.get("submission_file") or {}
        manifest_template = manifest.get("feedback_template_file") or {}
        self._compare_field(
            "submission SHA-256",
            manifest_submission.get("sha256"),
            actual_submission.get("sha256"),
            issues,
        )
        self._compare_field(
            "submission row count",
            manifest_submission.get("row_count"),
            actual_submission.get("row_count"),
            issues,
        )
        self._compare_field(
            "submission columns",
            manifest_submission.get("columns"),
            actual_submission.get("columns"),
            issues,
        )
        self._compare_field(
            "feedback template SHA-256",
            manifest_template.get("sha256"),
            actual_template.get("sha256"),
            issues,
        )
        if template:
            self._compare_field(
                "template expected submission SHA-256",
                template.get("expected_submission_sha256"),
                actual_submission.get("sha256"),
                issues,
            )
            self._compare_field(
                "template expected submission row count",
                template.get("expected_submission_rows"),
                actual_submission.get("row_count"),
                issues,
            )
            self._compare_field(
                "template expected submission columns",
                template.get("expected_submission_columns"),
                actual_submission.get("columns"),
                issues,
            )
            candidate_task_id = (manifest.get("candidate") or {}).get("task_id")
            self._compare_field(
                "template candidate_task_id",
                candidate_task_id,
                template.get("candidate_task_id"),
                issues,
            )
            self._compare_field(
                "template submission_target",
                manifest.get("submission_target"),
                template.get("submission_target"),
                issues,
            )
        if manifest.get("status") != "ready_for_manual_upload":
            issues.append("manual_submission_package manifest is not ready_for_manual_upload.")

        status = "pass" if not issues else "needs_review"
        report = {
            "competition_name": manifest.get("competition_name", self.competition_dir.name),
            "status": status,
            "decision": "package_verified_for_upload" if status == "pass" else "package_verification_blocked",
            "package_dir": str(package_dir),
            "manifest_path": str(manifest_path),
            "template_path": str(template_path),
            "submission_path": str(submission_path),
            "manifest_status": manifest.get("status"),
            "submission_target": manifest.get("submission_target"),
            "candidate_task_id": (manifest.get("candidate") or {}).get("task_id"),
            "actual_submission_file": actual_submission,
            "manifest_submission_file": manifest_submission,
            "actual_feedback_template_file": actual_template,
            "manifest_feedback_template_file": manifest_template,
            "experiment_roadmap_path": None,
            "issues": issues,
            "warnings": warnings,
            "next_action": (
                "Upload manual_submission_package/submission.csv, then fill leaderboard feedback."
                if status == "pass"
                else "Rebuild the manual submission package before uploading."
            ),
        }
        report_path = self.competition_dir / "manual_submission_package_verification.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        from .experiment_roadmap import ExperimentRoadmapBuilder

        roadmap_result = ExperimentRoadmapBuilder(self.competition_dir, memory=self.memory).build()
        report["experiment_roadmap_path"] = str(roadmap_result.roadmap_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ledger_entry = self.ledger.create_entry(
            task_id="manual_submission_package_verification",
            agent="manual_submission_package_verifier",
            title="Verify manual submission package",
            status=status,
            input_payload=report,
            prompt="Verify that the packaged submission, manifest, and feedback template still point to the same upload file.",
            scorecard={
                "agent": "manual_submission_package_verifier",
                "task_id": "manual_submission_package_verification",
                "status": status,
                "scores": {
                    "submission_target": manifest.get("submission_target", "n/a"),
                    "candidate_task_id": (manifest.get("candidate") or {}).get("task_id") or "n/a",
                    "submission_sha256": actual_submission.get("sha256") or "n/a",
                    "submission_rows": actual_submission.get("row_count", "n/a"),
                },
                "metric_name": (manifest.get("candidate") or {}).get("metric_name"),
                "local_score": (manifest.get("candidate") or {}).get("local_score"),
                "issues": issues + warnings,
                "recommended_human_action": "continue" if status == "pass" else "patch_prompt",
            },
            artifacts={
                "manual_submission_package_verification": report_path,
                "experiment_roadmap": roadmap_result.roadmap_path,
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=report["competition_name"],
                profile_name="tabular_classic",
                task_id="manual_submission_package_verification",
                status=status,
                metric_name=(manifest.get("candidate") or {}).get("metric_name"),
                local_score=(manifest.get("candidate") or {}).get("local_score"),
                brain_review_path=str(report_path),
                artifacts=[str(report_path), str(self.competition_dir / ledger_entry.html_report_path)],
                notes=report["next_action"],
            )
        )
        return ManualSubmissionPackageVerificationResult(
            status=status,
            report_path=report_path,
            experiment_roadmap_path=roadmap_result.roadmap_path,
        )

    @staticmethod
    def _compare_field(label: str, expected: Any, actual: Any, issues: list[str]) -> None:
        if expected is None and actual is None:
            return
        if expected != actual:
            issues.append(f"{label} mismatch: expected={expected}, actual={actual}.")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
