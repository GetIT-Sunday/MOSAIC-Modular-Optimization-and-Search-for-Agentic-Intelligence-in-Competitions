from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class LeaderboardFeedbackFreshnessAuditor:
    """Check whether leaderboard feedback belongs to the current packaged submission."""

    def __init__(self, competition_dir: Path):
        self.competition_dir = competition_dir.resolve()

    def audit(
        self,
        *,
        leaderboard_feedback: Optional[Dict[str, Any]] = None,
        manual_package: Optional[Dict[str, Any]] = None,
        post_submit_workflow: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        feedback = leaderboard_feedback if isinstance(leaderboard_feedback, dict) else self._read_json(
            self.competition_dir / "leaderboard_feedback.json"
        )
        package = manual_package if isinstance(manual_package, dict) else self._read_json(
            self.competition_dir / "manual_submission_package" / "manifest.json"
        )
        workflow = post_submit_workflow if isinstance(post_submit_workflow, dict) else self._read_json(
            self.competition_dir / "post_submit_workflow.json"
        )
        expected = self._expected_binding(package, workflow)
        actual = self._actual_binding(feedback)
        issues: list[str] = []
        warnings: list[str] = []

        if not feedback:
            return self._result(
                status="missing",
                is_current=False,
                expected=expected,
                actual=actual,
                issues=["leaderboard_feedback.json is missing."],
                warnings=[],
            )
        if not expected.get("candidate_task_id"):
            warnings.append("Current packaged candidate task_id is unavailable; feedback freshness is weak.")
        if expected.get("candidate_task_id") and actual.get("candidate_task_id") != expected.get("candidate_task_id"):
            issues.append(
                "candidate_task_id mismatch: "
                f"feedback={actual.get('candidate_task_id')}, current={expected.get('candidate_task_id')}"
            )
        if expected.get("submission_target") and actual.get("submission_target") != expected.get("submission_target"):
            issues.append(
                "submission_target mismatch: "
                f"feedback={actual.get('submission_target')}, current={expected.get('submission_target')}"
            )

        binding_fields = [
            ("expected_submission_sha256", "sha256"),
            ("expected_submission_rows", "rows"),
            ("expected_submission_columns", "columns"),
        ]
        saw_file_binding = False
        for field, label in binding_fields:
            expected_value = expected.get(field)
            actual_value = actual.get(field)
            if expected_value is None:
                continue
            saw_file_binding = True
            if actual_value is None:
                issues.append(f"{field} is missing from leaderboard feedback.")
            elif actual_value != expected_value:
                issues.append(
                    f"{label} mismatch: feedback={actual_value}, current={expected_value}"
                )
        if expected.get("candidate_risk_level") and actual.get("candidate_risk_level") is None:
            warnings.append("candidate_risk_level is missing from leaderboard feedback.")

        if issues:
            status = "stale"
            is_current = False
        elif saw_file_binding:
            status = "fresh"
            is_current = True
        else:
            status = "weak_match"
            is_current = True
            warnings.append("Feedback matches candidate metadata but no file hash/row/column binding is available.")
        return self._result(status, is_current, expected, actual, issues, warnings)

    @staticmethod
    def _expected_binding(package: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        candidate = package.get("candidate") or workflow.get("candidate") or {}
        submission_file = package.get("submission_file") or workflow.get("submission_file") or {}
        candidate_risk = package.get("candidate_risk") or workflow.get("candidate_risk") or {}
        return {
            "submission_target": package.get("submission_target") or workflow.get("submission_target"),
            "candidate_task_id": candidate.get("task_id"),
            "expected_submission_sha256": submission_file.get("sha256"),
            "expected_submission_rows": submission_file.get("row_count"),
            "expected_submission_columns": submission_file.get("columns"),
            "candidate_risk_level": candidate_risk.get("risk_level"),
        }

    @staticmethod
    def _actual_binding(feedback: Dict[str, Any]) -> Dict[str, Any]:
        binding = feedback.get("submission_binding") if isinstance(feedback.get("submission_binding"), dict) else {}
        return {
            "submission_target": feedback.get("submission_target"),
            "candidate_task_id": feedback.get("candidate_task_id"),
            "expected_submission_sha256": feedback.get("expected_submission_sha256")
            or binding.get("expected_submission_sha256"),
            "expected_submission_rows": feedback.get("expected_submission_rows")
            or binding.get("expected_submission_rows"),
            "expected_submission_columns": feedback.get("expected_submission_columns")
            or binding.get("expected_submission_columns"),
            "candidate_risk_level": feedback.get("candidate_risk_level")
            or binding.get("candidate_risk_level"),
        }

    @staticmethod
    def _result(
        status: str,
        is_current: bool,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        issues: list[str],
        warnings: list[str],
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "is_current": is_current,
            "expected": expected,
            "actual": actual,
            "issues": issues,
            "warnings": warnings,
        }

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
