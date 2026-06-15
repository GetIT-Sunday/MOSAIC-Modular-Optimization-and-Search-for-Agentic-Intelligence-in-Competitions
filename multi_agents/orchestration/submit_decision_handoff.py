from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .manual_submit_readiness import ManualSubmitReadinessChecker
from .memory import CompetitionMemory, ExperimentRecord
from .run_ledger import RunLedger


@dataclass(frozen=True)
class SubmitDecisionHandoffResult:
    status: str
    report_path: Path
    markdown_path: Path


class SubmitDecisionHandoff:
    """Create an auditable handoff before any leaderboard submission."""

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
        submission_target: str = "recommended",
        refresh: bool = False,
    ) -> SubmitDecisionHandoffResult:
        if submission_target not in {"champion", "recommended"}:
            raise ValueError(f"Unsupported submission_target: {submission_target}")

        if refresh or not (self.competition_dir / "manual_submit_readiness.json").exists():
            readiness_result = ManualSubmitReadinessChecker(
                self.competition_dir,
                memory=self.memory,
            ).run(refresh=refresh, submission_target=submission_target)
            readiness_path = readiness_result.report_path
        else:
            readiness_path = self.competition_dir / "manual_submit_readiness.json"

        readiness = self._read_json(readiness_path)
        submission_policy = self._read_json(self.competition_dir / "submission_policy.json")
        promotion_gate = self._read_json(self.competition_dir / "promotion_gate_review.json")
        submission_gate = self._read_json(self.competition_dir / "submission_gate.json")
        post_workflow = self._read_json(self.competition_dir / "post_submit_workflow.json")

        candidate = readiness.get("candidate") or submission_gate.get("candidate") or {}
        submission_path = self._resolve_competition_path(readiness.get("submission_path"))
        issues = []
        warnings = []
        warnings.extend(readiness.get("warnings") or [])
        warnings.extend(submission_policy.get("warnings") or [])
        warnings.extend(promotion_gate.get("warnings") or [])
        warnings.extend(submission_gate.get("warnings") or [])

        if readiness.get("submission_target") != submission_target:
            issues.append("manual_submit_readiness.json target does not match requested submission target.")
        if not readiness.get("manual_submission_ready"):
            issues.append("manual_submit_readiness.json is not ready for manual submission.")
        if submission_gate.get("status") != "pass":
            issues.append("submission_gate.json is not pass.")
        if submission_gate.get("submission_target") != submission_target:
            issues.append("submission_gate.json target does not match requested submission target.")
        if submission_target == "recommended":
            if submission_policy.get("decision") != "recommended_submission_selected":
                issues.append("submission_policy.json has not selected a recommended submission.")
            policy_source = (submission_policy.get("policy") or {}).get("source")
            if policy_source != "promotion_gate":
                warnings.append("Recommended submission is not sourced from promotion_gate.")
            if policy_source == "promotion_gate" and promotion_gate.get("decision") != "promote_candidate":
                issues.append("promotion_gate_review.json has not promoted a candidate.")
        if not candidate.get("task_id"):
            issues.append("Submission candidate task_id is unavailable.")
        if not submission_path.exists():
            issues.append("Submission file is missing.")

        status = "ready_for_human_submit_decision" if not issues else "needs_review"
        decision = "await_human_submit_decision" if status == "ready_for_human_submit_decision" else "fix_submit_handoff"
        competition_name = readiness.get("competition_name", self.competition_dir.name)
        post_submit_command = (
            "python framework.py "
            f"--competition {competition_name} "
            f"--post-submit-workflow --submission-target {submission_target}"
        )
        report = {
            "competition_name": competition_name,
            "status": status,
            "decision": decision,
            "submission_target": submission_target,
            "candidate": candidate,
            "submission_path": str(submission_path),
            "manual_submit_readiness_path": str(readiness_path),
            "manual_submission_ready": readiness.get("manual_submission_ready"),
            "api_submission_review_ready": readiness.get("api_submission_review_ready"),
            "confirmed_submit_ready": readiness.get("confirmed_submit_ready"),
            "submission_policy": {
                "status": submission_policy.get("status"),
                "decision": submission_policy.get("decision"),
                "source": (submission_policy.get("policy") or {}).get("source"),
                "recommended_submission_path": submission_policy.get("recommended_submission_path"),
            },
            "promotion_gate": {
                "status": promotion_gate.get("status"),
                "decision": promotion_gate.get("decision"),
                "promoted_task_id": (promotion_gate.get("promoted_candidate") or {}).get("task_id"),
                "promoted_submission_path": promotion_gate.get("promoted_submission_path"),
            },
            "submission_gate": {
                "status": submission_gate.get("status"),
                "decision": submission_gate.get("decision"),
                "target": submission_gate.get("submission_target"),
            },
            "post_submit_workflow_status": post_workflow.get("status"),
            "evidence_summary": self._evidence_summary(candidate, promotion_gate),
            "post_submit_workflow_command": post_submit_command,
            "required_human_decision": {
                "file_to_upload": str(submission_path),
                "decision_options": ["submit_now", "defer", "request_more_evidence", "patch_prompt"],
                "after_submit": [
                    "Run the post-submit workflow command.",
                    "Fill leaderboard_feedback_input_template.json with the observed public score, rank, or submission id.",
                    "Run --leaderboard-feedback-from-template to close the feedback loop.",
                ],
            },
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "next_action": (
                "Human should decide whether to upload the listed submission, then run post-submit workflow after upload."
                if status == "ready_for_human_submit_decision"
                else "Fix handoff blockers, rerun readiness and submit decision handoff."
            ),
        }

        report_path = self.competition_dir / "submit_decision_handoff.json"
        markdown_path = self.competition_dir / "submit_decision_handoff.md"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(self._render_markdown(report), encoding="utf-8")

        ledger_entry = self.ledger.create_entry(
            task_id="submit_decision_handoff",
            agent="submit_decision_handoff",
            title="Prepare human submit decision handoff",
            status=status,
            input_payload=report,
            prompt=(
                "Create an auditable pre-submit handoff that binds the human decision to the exact "
                "candidate, submission file, readiness evidence, and post-submit feedback workflow."
            ),
            scorecard={
                "agent": "submit_decision_handoff",
                "task_id": "submit_decision_handoff",
                "status": status,
                "scores": {
                    "submission_target": submission_target,
                    "manual_submission_ready": readiness.get("manual_submission_ready"),
                    "policy_source": (submission_policy.get("policy") or {}).get("source"),
                    "promotion_gate_decision": promotion_gate.get("decision"),
                    "post_submit_workflow_ready": post_workflow.get("status") == "ready_for_manual_submit",
                },
                "metric_name": candidate.get("metric_name"),
                "local_score": candidate.get("local_score"),
                "issues": issues + list(dict.fromkeys(warnings)),
                "recommended_human_action": "continue" if status == "ready_for_human_submit_decision" else "patch_prompt",
            },
            artifacts={
                "submit_decision_handoff": report_path,
                "submit_decision_handoff_markdown": markdown_path,
                "manual_submit_readiness": readiness_path,
                "submission_policy": self.competition_dir / "submission_policy.json",
                "promotion_gate_review": self.competition_dir / "promotion_gate_review.json",
                "submission_gate": self.competition_dir / "submission_gate.json",
            },
        )
        self.memory.append(
            ExperimentRecord(
                competition_name=competition_name,
                profile_name="tabular_classic",
                task_id="submit_decision_handoff",
                status=status,
                metric_name=candidate.get("metric_name"),
                local_score=candidate.get("local_score"),
                submission_path=str(submission_path) if submission_path.exists() else None,
                brain_review_path=str(report_path),
                artifacts=[
                    str(report_path),
                    str(markdown_path),
                    str(self.competition_dir / ledger_entry.html_report_path),
                ],
                notes=report["next_action"],
            )
        )
        return SubmitDecisionHandoffResult(status=status, report_path=report_path, markdown_path=markdown_path)

    @staticmethod
    def _evidence_summary(candidate: Dict[str, Any], promotion_gate: Dict[str, Any]) -> Dict[str, Any]:
        promoted = promotion_gate.get("promoted_candidate") or {}
        merged = dict(promoted)
        merged.update({key: value for key, value in candidate.items() if value is not None})
        evidence_keys = [
            "task_id",
            "metric_name",
            "local_score",
            "seed_mean",
            "seed_std",
            "fold_std",
            "train_valid_gap",
            "seed_ci95",
            "max_model_correlation",
        ]
        return {key: merged.get(key) for key in evidence_keys if key in merged}

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

    @staticmethod
    def _render_markdown(report: Dict[str, Any]) -> str:
        candidate = report.get("candidate") or {}
        issues = "\n".join(f"- {issue}" for issue in report.get("issues", [])) or "- None"
        warnings = "\n".join(f"- {warning}" for warning in report.get("warnings", [])) or "- None"
        evidence = report.get("evidence_summary") or {}
        evidence_lines = "\n".join(f"- {key}: {value}" for key, value in evidence.items()) or "- None"
        return f"""# Submit Decision Handoff

Competition: {report.get("competition_name")}
Status: {report.get("status")}
Decision: {report.get("decision")}
Submission target: {report.get("submission_target")}

## Candidate

- Task: {candidate.get("task_id", "unknown")}
- Metric: {candidate.get("metric_name", "unknown")}
- Local score: {candidate.get("local_score", "n/a")}
- Submission file: `{report.get("submission_path")}`

## Evidence

{evidence_lines}

## Human Decision

Choose one of:
- submit_now
- defer
- request_more_evidence
- patch_prompt

This handoff does not submit automatically.

## After Submit

Run:

```bash
{report.get("post_submit_workflow_command")}
```

Then fill `leaderboard_feedback_input_template.json` and run `--leaderboard-feedback-from-template`.

## Blocking Issues

{issues}

## Warnings

{warnings}

Next action: {report.get("next_action")}
"""

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
