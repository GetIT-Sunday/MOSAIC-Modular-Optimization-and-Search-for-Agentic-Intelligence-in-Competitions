import json
from pathlib import Path

from multi_agents.orchestration import (
    CompetitionIngestor,
    ExperimentChampionSelector,
    KaggleSubmitAdapter,
    ManualSubmitReadinessChecker,
    PostReselectionGate,
    SubmissionGate,
    SubmissionValidator,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ROOT = REPO_ROOT / "multi_agents" / "competition"


def _copy_titanic(tmp_path: Path) -> Path:
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    return competition_dir


def _write_candidate(competition_dir: Path, task_id: str, score: float, risk_level: str = "low") -> None:
    source = COMPETITION_ROOT / "titanic"
    exp_dir = competition_dir / "experiments" / task_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "submission.csv").write_text(
        (source / "sample_submission.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (exp_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "experiment": task_id,
                "status": "completed",
                "metric_name": "accuracy",
                "local_score": score,
            }
        ),
        encoding="utf-8",
    )
    (exp_dir / "validator_result.json").write_text(
        json.dumps({"ok": True, "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    (exp_dir / "risk_audit.json").write_text(
        json.dumps({"risk_level": risk_level, "issues": []}),
        encoding="utf-8",
    )


def test_smoke_ingest_validate_select_gate_and_dry_run(tmp_path: Path):
    competition_dir = _copy_titanic(tmp_path)
    manifest = CompetitionIngestor(competition_dir).build_manifest()
    validation = SubmissionValidator(manifest).validate(competition_dir / "sample_submission.csv")

    assert manifest.target_column == "Survived"
    assert validation.ok

    _write_candidate(competition_dir, "smoke_candidate", 0.80)
    selection = ExperimentChampionSelector(competition_dir).select()
    gate = SubmissionGate(competition_dir).run(dry_run=True)
    plan = KaggleSubmitAdapter(competition_dir).plan(dry_run=True)

    assert selection.status == "pass"
    assert gate.status == "pass"
    assert plan.status == "pass"
    assert (competition_dir / "champion_submission.csv").exists()
    assert "KAGGLE_KEY" not in (competition_dir / "kaggle_submit_plan.json").read_text(encoding="utf-8")


def test_smoke_post_reselection_refreshes_current_champion(tmp_path: Path):
    competition_dir = _copy_titanic(tmp_path)
    _write_candidate(competition_dir, "old_high_gap", 0.84)
    _write_candidate(competition_dir, "stability_first_search_v1", 0.83)
    (competition_dir / "leaderboard_gap_audit.json").write_text(
        json.dumps(
            {
                "risk_level": "high",
                "champion": {"task_id": "old_high_gap", "source_id": "experiment:old_high_gap"},
                "score_gap": {"materially_worse": True, "gap": -0.04},
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "stability_first_review.json").write_text(
        json.dumps(
            {
                "task_id": "stability_first_search_v1",
                "status": "validated",
                "risk_level": "low",
                "local_score": 0.83,
            }
        ),
        encoding="utf-8",
    )

    ExperimentChampionSelector(competition_dir).select()
    result = PostReselectionGate(competition_dir).run()
    readiness = ManualSubmitReadinessChecker(competition_dir).run(refresh=False)
    report = json.loads((competition_dir / "post_reselection_gate.json").read_text(encoding="utf-8"))
    gate = json.loads((competition_dir / "submission_gate.json").read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert readiness.status == "manual_submit_ready"
    assert report["champion"]["task_id"] == "stability_first_search_v1"
    assert gate["champion"]["task_id"] == "stability_first_search_v1"


def test_sync_from_dev_lite_includes_feedback_loop_artifacts():
    script = (REPO_ROOT / "scripts" / "sync_from_dev.sh").read_text(encoding="utf-8")

    assert 'CONFIG_JSON="${AUTOKAGGLE_CONFIG:-autokaggle_config.json}"' in script
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in script
    assert 'get("ssh_alias")) or "dev"' in script
    assert 'get("workspace")) or "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac"' in script
    assert "Refusing to sync outside hard remote workspace" in script

    for artifact in [
        "leaderboard_feedback_template_fill.json",
        "leaderboard_feedback_input_validation.json",
        "leaderboard_feedback_loop.json",
        "leaderboard_target.json",
        "leaderboard_target_raw.csv",
        "manual_submission_package_verification.json",
        "competition_intake.json",
        "experiment_roadmap.json",
        "runs/index.html",
    ]:
        assert artifact in script


def test_sync_to_dev_preserves_remote_competition_runtime_data():
    script = (REPO_ROOT / "scripts" / "sync_to_dev.sh").read_text(encoding="utf-8")

    for pattern in [
        'multi_agents/competition/*/*.zip',
        'multi_agents/competition/*/*.csv',
        'multi_agents/competition/*/*.parquet',
        'multi_agents/competition/*/overview.txt',
        'multi_agents/competition/*/competition_intake.json',
        'multi_agents/competition/*/runs/',
        'multi_agents/competition/*/experiments/',
    ]:
        assert f'--exclude "{pattern}"' in script
