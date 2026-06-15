import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from multi_agents.orchestration import (
    AgentLoopController,
    BrainOrchestrator,
    CompetitionIngestor,
    CompetitionIntakeAgent,
    CompetitionMemory,
    EnhancementRunner,
    ExperimentQueueBuilder,
    ExperimentRoadmapBuilder,
    ExperimentChampionSelector,
    ExperimentRecord,
    HumanGate,
    HarnessRegistry,
    IterationOrchestrator,
    KaggleDiscoveryAgent,
    KaggleSubmitAdapter,
    LeaderboardFeedbackInputRunner,
    LeaderboardFeedbackRecorder,
    LeaderboardFeedbackLoop,
    LeaderboardFeedbackTemplateFiller,
    LeaderboardGapAuditor,
    LeaderboardTargetAgent,
    ManualSubmissionPackage,
    ManualSubmissionPackageVerifier,
    ManualSubmitReadinessChecker,
    PostReselectionGate,
    PostExperimentPipeline,
    PostSubmitWorkflow,
    ProjectConfigAgent,
    ProjectConsoleAgent,
    ProjectControlPanel,
    PromotionGateEvaluator,
    RemoteBrainReviewer,
    RemoteHealthCheckAgent,
    RunLedger,
    SkillRegistry,
    SubmissionGate,
    SubmissionDecisionReviewer,
    SubmitDecisionHandoff,
    SubmissionPolicy,
    StabilityFirstRunner,
    SubmissionValidator,
    TabularBaselineRunner,
    TabularFeatureLeakageAuditor,
    TabularFeaturePruner,
    TabularRiskAuditor,
    TabularSearchRunner,
)
from multi_agents.orchestration.profile import load_profile, profile_file_match_score
from multi_agents.orchestration.task_identifier import CompetitionSignal, identify_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ROOT = REPO_ROOT / "multi_agents" / "competition"


def test_kaggle_discovery_agent_caches_competition_pool_and_renders_panel(tmp_path: Path):
    commands = []

    def fake_runner(command):
        import subprocess

        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "ref,title,deadline,category,reward,teamCount,userHasEntered\n"
                "cafa-6-protein-function-prediction,CAFA 6,2026-09-01,research,$50000,123,false\n"
                "playground-series-s6e6,Bank Churn Playground,2026-07-01,playground,Swag,456,true\n"
            ),
            stderr="",
        )

    competition_root = tmp_path / "competition"
    result = KaggleDiscoveryAgent(competition_root, runner=fake_runner).discover(
        category="research",
        sort_by="recentlyCreated",
    )

    assert result.status == "pass"
    assert result.competitions[0].ref == "cafa-6-protein-function-prediction"
    assert result.competitions[0].team_count == "123"
    assert "--category" in commands[0]
    cache = json.loads((competition_root / "kaggle_competitions_cache.json").read_text(encoding="utf-8"))
    assert cache["competitions"][1]["ref"] == "playground-series-s6e6"

    competition_dir = competition_root / "bank_churn"
    competition_dir.mkdir()
    project_html = (ProjectControlPanel(competition_root).write_html()).read_text(encoding="utf-8")
    pool_html = (competition_root / "console" / "pool.html").read_text(encoding="utf-8")
    assert "ProjectConsoleAgent" in project_html
    assert "Kaggle 题目池" in pool_html
    assert "cafa-6-protein-function-prediction" in pool_html
    assert 'href="console/pool.html"' in project_html
    assert 'href="#pool"' not in project_html
    assert 'href="../console/workspaces.html"' in pool_html
    assert "python framework.py --kaggle-select cafa-6-protein-function-prediction --kaggle-download" in project_html

    competition_dir = competition_root / "bank_churn"
    competition_dir.mkdir(exist_ok=True)
    competition_html = (RunLedger(competition_dir).write_html()).read_text(encoding="utf-8")
    assert "返回 AutoKaggle 总控制台" in competition_html
    assert "Kaggle 题目池" not in competition_html
    assert 'href="console/data.html"' in competition_html
    assert 'href="#data"' not in competition_html
    assert (competition_dir / "runs" / "console" / "data.html").exists()
    assert (competition_dir / "runs" / "console" / "brain.html").exists()
    assert (competition_dir / "runs" / "console" / "experiments.html").exists()
    assert (competition_dir / "runs" / "console" / "submission.html").exists()


def test_kaggle_select_writes_intake_without_secrets_and_renders_panel(tmp_path: Path):
    def fake_runner(command):
        import subprocess

        if command[:3] == ["kaggle", "competitions", "files"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="name,size,creationDate\ntrain.csv,10MB,2026-06-01\ntest.csv,4MB,2026-06-01\nsample_submission.csv,1MB,2026-06-01\n",
                stderr="",
            )
        raise AssertionError(command)

    competition_root = tmp_path / "competition"
    cache = {
        "status": "pass",
        "competitions": [
            {
                "ref": "playground-series-s6e6",
                "title": "Bank Churn Playground",
                "category": "playground",
            }
        ],
    }
    competition_root.mkdir()
    (competition_root / "kaggle_competitions_cache.json").write_text(
        json.dumps(cache),
        encoding="utf-8",
    )

    result = KaggleDiscoveryAgent(competition_root, runner=fake_runner).select(
        "https://www.kaggle.com/competitions/playground-series-s6e6"
    )
    intake_text = result.intake_path.read_text(encoding="utf-8")
    intake = json.loads(intake_text)
    html = (result.competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert result.status == "selected"
    assert intake["competition_slug"] == "playground-series-s6e6"
    assert intake["cached_competition"]["title"] == "Bank Churn Playground"
    assert "KAGGLE_KEY" not in intake_text
    assert "竞赛 Intake" in html
    assert "返回 AutoKaggle 总控制台" in html
    assert "Kaggle 题目池" not in html
    assert "train.csv" in html
    assert "python framework.py --competition playground-series-s6e6 --run-baselines" in html

    project_html = (ProjectControlPanel(competition_root).write_html()).read_text(encoding="utf-8")
    workspaces_html = (competition_root / "console" / "workspaces.html").read_text(encoding="utf-8")
    snapshot = json.loads((competition_root / "console" / "snapshot.json").read_text(encoding="utf-8"))
    assert "Competition Workspaces" in workspaces_html
    assert "playground-series-s6e6" in workspaces_html
    assert "../playground-series-s6e6/runs/index.html" in workspaces_html
    assert snapshot["generated_by"] == "ProjectConsoleAgent"


def test_kaggle_select_download_extracts_archives(tmp_path: Path):
    competition_root = tmp_path / "competition"

    def fake_runner(command):
        if command[:3] == ["kaggle", "competitions", "files"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="name,size,creationDate\ntrain.csv,10,2026-06-01\ntest.csv,4,2026-06-01\nsample_submission.csv,1,2026-06-01\n",
                stderr="",
            )
        if command[:3] == ["kaggle", "competitions", "download"]:
            out_dir = Path(command[command.index("-p") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out_dir / "demo.zip", "w") as archive:
                archive.writestr("train.csv", "id,target\n1,0\n")
                archive.writestr("test.csv", "id\n2\n")
                archive.writestr("sample_submission.csv", "id,target\n2,0\n")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    result = KaggleDiscoveryAgent(competition_root, runner=fake_runner).select("demo", download=True)

    assert result.status == "selected"
    assert (result.competition_dir / "train.csv").exists()
    assert (result.competition_dir / "test.csv").exists()
    assert (result.competition_dir / "sample_submission.csv").exists()


def test_project_console_agent_writes_real_navigation_pages(tmp_path: Path):
    competition_root = tmp_path / "competition"
    workspace = competition_root / "demo_competition"
    workspace.mkdir(parents=True)
    (workspace / "baseline_review.json").write_text("{}", encoding="utf-8")
    (workspace / "runs").mkdir()
    (workspace / "runs" / "index.html").write_text("<html></html>", encoding="utf-8")
    (competition_root / "kaggle_competitions_cache.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "competitions": [
                    {
                        "ref": "demo-new-competition",
                        "title": "Demo New Competition",
                        "category": "playground",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    index_path = ProjectConsoleAgent(competition_root).write_html()
    index_html = index_path.read_text(encoding="utf-8")
    pool_html = (competition_root / "console" / "pool.html").read_text(encoding="utf-8")
    workspaces_html = (competition_root / "console" / "workspaces.html").read_text(encoding="utf-8")

    assert (competition_root / "console" / "config.html").exists()
    assert (competition_root / "console" / "roadmap.html").exists()
    assert 'href="console/pool.html"' in index_html
    assert 'href="../index.html"' in pool_html
    assert 'href="#pool"' not in index_html
    assert "demo-new-competition" in pool_html
    assert "../demo_competition/runs/index.html" in workspaces_html


def test_project_config_agent_checks_private_config_without_rendering_secrets(tmp_path: Path, monkeypatch):
    example = {
        "remote": {
            "ssh_alias": "friend_gpu",
            "workspace": "/home/dataset-local/data_local/friend/AutoKaggle",
            "project_subdir": "workspaces/AutoKaggle",
            "conda_env": "mac",
        },
        "llm": {
            "openai_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
            "api_key_env": "AUTOKAGGLE_API_KEY",
            "api_key_file": "api_key.txt",
            "planner_model": "mimo-v2.5-pro",
            "coding_model": "mimo-v2.5",
            "cheap_model": "mimo-v2.5",
        },
        "safety": {
            "allow_paths": ["/home/dataset-local/data_local/friend/AutoKaggle"],
            "never_render_secret_values": True,
        },
        "kaggle": {"remote_config_dir": ".kaggle"},
    }
    (tmp_path / "autokaggle_config.example.json").write_text(json.dumps(example), encoding="utf-8")
    (tmp_path / "autokaggle_config.json").write_text(
        json.dumps({"llm": {"api_key": "sk-should-not-render"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTOKAGGLE_API_KEY", "sk-env-secret")

    agent = ProjectConfigAgent(tmp_path)
    status_path = agent.write_status(tmp_path / "config_status.json")
    payload = status_path.read_text(encoding="utf-8")
    snapshot = agent.snapshot()

    assert snapshot.using_private_config is True
    assert "sk-should-not-render" not in payload
    assert "sk-env-secret" not in payload
    assert "friend_gpu" in payload
    assert any(check.key == "llm_api_key" and check.status == "pass" for check in snapshot.checks)


def test_project_console_config_page_shows_setup_and_checks(tmp_path: Path):
    project_root = tmp_path
    competition_root = project_root / "multi_agents" / "competition"
    competition_root.mkdir(parents=True)
    (project_root / "autokaggle_config.example.json").write_text(
        json.dumps(
            {
                "remote": {
                    "ssh_alias": "friend_gpu",
                    "workspace": "/home/dataset-local/data_local/friend/AutoKaggle",
                    "project_subdir": "workspaces/AutoKaggle",
                    "conda_env": "mac",
                },
                "llm": {
                    "openai_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
                    "api_key_env": "AUTOKAGGLE_API_KEY",
                    "api_key_file": "api_key.txt",
                    "planner_model": "mimo-v2.5-pro",
                    "coding_model": "mimo-v2.5",
                },
                "safety": {"allow_paths": ["/home/dataset-local/data_local/friend/AutoKaggle"]},
                "kaggle": {"remote_config_dir": ".kaggle"},
            }
        ),
        encoding="utf-8",
    )

    ProjectConsoleAgent(competition_root).write_html()
    config_html = (competition_root / "console" / "config.html").read_text(encoding="utf-8")
    status_payload = (competition_root / "console" / "config_status.json").read_text(encoding="utf-8")

    assert "用户配置方式" in config_html
    assert "python framework.py --config-check" in config_html
    assert "friend_gpu" in config_html
    assert "config_status.json" in config_html
    assert "sk-" not in config_html
    assert "friend_gpu" in status_payload


def test_remote_health_check_agent_reports_conda_blocked_without_secrets(tmp_path: Path):
    (tmp_path / "autokaggle_config.example.json").write_text(
        json.dumps(
            {
                "remote": {
                    "ssh_alias": "dev",
                    "workspace": "/safe/workspace",
                    "project_subdir": "workspaces/AutoKaggle",
                    "conda_env": "mac",
                }
            }
        ),
        encoding="utf-8",
    )
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        if "conda run" in command[-1]:
            return subprocess.CompletedProcess(command, 137, stdout="", stderr="Killed")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "AUTOKAGGLE_HEALTH_BEGIN\n"
                "workspace=pass:/safe/workspace\n"
                "boundary=pass:/safe/workspace/workspaces/AutoKaggle\n"
                "project=pass:/safe/workspace/workspaces/AutoKaggle\n"
                "conda_cli=pass:/opt/conda/bin/conda\n"
                "disk=pass:1000000KB_available\n"
                "gpu=warn:nvidia-smi not found\n"
                "kaggle_config=pass:/safe/workspace/.kaggle\n"
                "AUTOKAGGLE_HEALTH_END\n"
            ),
            stderr="",
        )

    result = RemoteHealthCheckAgent(tmp_path, runner=fake_runner).run(tmp_path / "health.json")
    payload = (tmp_path / "health.json").read_text(encoding="utf-8")

    assert result.status == "blocked"
    assert any(check.key == "conda_env" and check.status == "blocked" for check in result.checks)
    assert "Killed" in payload
    assert "KAGGLE_KEY" not in payload
    assert commands[0][:2] == ["ssh", "dev"]


def test_project_console_config_page_renders_remote_health_report(tmp_path: Path):
    project_root = tmp_path
    competition_root = project_root / "multi_agents" / "competition"
    console_dir = competition_root / "console"
    console_dir.mkdir(parents=True)
    (project_root / "autokaggle_config.example.json").write_text(
        json.dumps({"remote": {"ssh_alias": "dev", "workspace": "/safe/workspace", "conda_env": "mac"}}),
        encoding="utf-8",
    )
    (console_dir / "remote_health_check.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-12T00:00:00+00:00",
                "status": "blocked",
                "checks": [
                    {"key": "ssh", "label": "SSH 连接", "status": "pass", "detail": "dev"},
                    {"key": "conda_env", "label": "Conda 环境运行", "status": "blocked", "detail": "Killed"},
                ],
            }
        ),
        encoding="utf-8",
    )

    ProjectConsoleAgent(competition_root).write_html()
    config_html = (console_dir / "config.html").read_text(encoding="utf-8")

    assert "远端健康检查" in config_html
    assert "remote_health_check.json" in config_html
    assert "Conda 环境运行" in config_html
    assert "Killed" in config_html


def test_kaggle_discovery_classifies_auth_and_rules_errors(tmp_path: Path):
    import subprocess

    def auth_runner(command):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="Authentication required to call the Kaggle API.",
            stderr="",
        )

    auth = KaggleDiscoveryAgent(tmp_path / "competition_auth", runner=auth_runner).discover()
    assert auth.status == "auth_missing"

    def rules_runner(command):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="You must accept this competition's rules before downloading files.",
            stderr="",
        )

    selected = KaggleDiscoveryAgent(tmp_path / "competition_rules", runner=rules_runner).select("demo")
    assert selected.status == "rules_not_accepted"

    download_root = tmp_path / "competition_download_rules"
    intermediate_statuses = []

    def forbidden_download_runner(command):
        if command[:3] == ["kaggle", "competitions", "files"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="name,size,creationDate\ntrain.csv,10,2026-06-01\n",
                stderr="",
            )
        if command[:3] == ["kaggle", "competitions", "download"]:
            intake_path = download_root / "playground-series-s6e6" / "competition_intake.json"
            intermediate_statuses.append(json.loads(intake_path.read_text(encoding="utf-8"))["status"])
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="403 Client Error: Forbidden for url: https://api.kaggle.com/v1/competitions/download",
            )
        raise AssertionError(command)

    download = KaggleDiscoveryAgent(download_root, runner=forbidden_download_runner).select(
        "playground-series-s6e6",
        download=True,
    )
    intake = json.loads(download.intake_path.read_text(encoding="utf-8"))
    assert download.status == "rules_not_accepted"
    assert intermediate_statuses == ["downloading"]
    assert intake["next_step"] == "accept_rules_on_kaggle"
    assert "open https://www.kaggle.com/competitions/playground-series-s6e6" in intake["recommended_commands"]


def test_competition_intake_agent_generates_artifacts_and_control_panel(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (competition_dir / "competition_intake.json").write_text(
        json.dumps(
            {
                "status": "selected",
                "competition_slug": "titanic_copy",
                "files": [{"name": "train.csv"}, {"name": "test.csv"}, {"name": "sample_submission.csv"}],
            }
        ),
        encoding="utf-8",
    )

    result = CompetitionIntakeAgent(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run()
    intake = json.loads(result.intake_path.read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert result.status == "ready_for_baseline"
    assert result.unknown_fields == []
    assert (competition_dir / "data_manifest.json").exists()
    assert (competition_dir / "task_card.md").exists()
    assert intake["intake_agent"]["next_command"].endswith("--agent-baseline-start")
    assert "竞赛 Intake" in html
    assert "unknown 字段" in html
    assert "--agent-baseline-start" in html


def test_competition_intake_agent_blocks_missing_required_files(tmp_path: Path):
    competition_dir = tmp_path / "missing_files"
    competition_dir.mkdir()
    (competition_dir / "competition_intake.json").write_text(
        json.dumps({"status": "selected", "competition_slug": "missing_files"}),
        encoding="utf-8",
    )

    result = CompetitionIntakeAgent(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run()
    intake = json.loads(result.intake_path.read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert result.status == "needs_data_or_review"
    assert "missing train.csv" in result.blocking_items
    assert "id_column" in result.unknown_fields
    assert intake["next_step"] == "download_or_fix_unknown_fields"
    assert "当前阻塞项" in html
    assert "missing train.csv" in html


def test_profile_file_match_and_task_identifier():
    tabular = load_profile("tabular_classic")
    score = profile_file_match_score(
        tabular,
        ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"],
    )
    assert score == 1.0

    decision = identify_profile(
        CompetitionSignal(
            competition_name="demo",
            files=["train.csv", "test.csv", "sample_submission.csv", "overview.txt"],
            overview_text="Binary classification tabular CSV competition.",
        )
    )
    assert decision.profile_name == "tabular_classic"
    assert decision.confidence > 0.5

    playground_decision = identify_profile(
        CompetitionSignal(
            competition_name="playground-series-s6e6",
            files=["train.csv", "test.csv", "sample_submission.csv"],
            overview_text="",
        )
    )
    assert playground_decision.profile_name == "tabular_classic"
    assert "multiple_csv_files_present" in playground_decision.reasons


def test_capability_registries_load_autokaggle_skills_and_harnesses():
    skill_summary = SkillRegistry().summary()
    harness_summary = HarnessRegistry().summary()
    skill_names = {item["name"] for item in skill_summary["skills"]}
    harness_names = {item["name"] for item in harness_summary["harnesses"]}

    assert {
        "leaderboard_target",
        "tabular_optimization",
        "validation_risk",
        "ensemble_strategy",
        "tabular_nn",
        "class_specialist",
        "clean_ensemble",
    }.issubset(skill_names)
    assert "stratified_gbdt_oof_harness" in harness_names
    assert "regularized_oof_blend_harness" in harness_names
    assert "tabular_nn_oof_harness" in harness_names
    assert "class_specialist_oof_harness" in harness_names
    assert "clean_oof_blend_harness" in harness_names
    assert HarnessRegistry().default_for_runner("tabular_mlp") == "tabular_nn_oof_harness"
    assert HarnessRegistry().default_for_runner("star_specialist_lgbm") == "class_specialist_oof_harness"
    assert HarnessRegistry().default_for_runner("star_specialist_threshold_tuning") == "class_specialist_oof_harness"
    assert HarnessRegistry().default_for_runner("clean_oof_blend") == "clean_oof_blend_harness"
    assert all(item["word_count"] > 100 for item in skill_summary["skills"])


def test_leaderboard_target_agent_parses_cli_snapshot(tmp_path: Path):
    competition_dir = tmp_path / "demo_competition"
    competition_dir.mkdir()
    (competition_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "competition_name": "demo_competition",
                "metric_candidates": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "metric_spec.json").write_text(
        json.dumps({"metric_name": "accuracy"}),
        encoding="utf-8",
    )
    (competition_dir / "baseline_review.json").write_text(
        json.dumps(
            {
                "best_baseline": {
                    "task_id": "baseline_sklearn_linear_baseline",
                    "metric_name": "accuracy",
                    "local_score": 0.927,
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command):
        assert command[:3] == ["kaggle", "competitions", "leaderboard"]
        rows = ["Next Page Token = token-demo", "teamId,teamName,submissionDate,score"]
        for index in range(1, 21):
            rows.append(f"{index},team_{index},2026-06-01,{1.0 - index * 0.001:.6f}")
        return subprocess.CompletedProcess(command, 0, stdout="\n".join(rows) + "\n", stderr="")

    result = LeaderboardTargetAgent(competition_dir, runner=fake_runner).run(page_size=20)
    target = json.loads(result.target_path.read_text(encoding="utf-8"))

    assert result.status == "completed"
    assert target["top_score"] == 0.999
    assert target["top_10_score"] == 0.99
    assert target["estimated_silver_score"] == 0.999
    assert round(target["gap_to_top"], 6) == 0.072
    assert target["next_decision"] == "gap_closing"
    assert "leaderboard_target_raw.csv" in [path.name for path in competition_dir.iterdir()]


def test_bio_profile_identifier():
    decision = identify_profile(
        CompetitionSignal(
            competition_name="sequence_demo",
            files=["train_sequences.fasta", "test_sequences.fasta", "ontology.obo"],
            overview_text="Protein sequence multi-label ontology prediction with GO terms.",
        )
    )
    assert decision.profile_name == "bio_sequence_multilabel"
    assert decision.confidence > 0.5


def test_brain_orchestrator_plans_coding_tasks(tmp_path: Path):
    competition_dir = tmp_path / "demo_competition"
    competition_dir.mkdir()
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        (competition_dir / name).write_text("id,value\n1,0\n", encoding="utf-8")
    (competition_dir / "overview.txt").write_text(
        "CSV binary classification competition.",
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    memory.append(
        ExperimentRecord(
            competition_name="old_demo",
            profile_name="tabular_classic",
            task_id="baseline",
            status="submitted",
            public_score=0.8,
            leaderboard_rank=100,
        )
    )

    decision = BrainOrchestrator(memory=memory).decide(competition_dir)

    assert decision.profile.name == "tabular_classic"
    assert decision.memory_summary["best_public_score"] == 0.8
    assert decision.data_manifest.target_column == "value"
    assert decision.coding_tasks[0].task_id == "competition_audit"
    assert "schema_report.json" in decision.coding_tasks[0].expected_outputs
    assert len(decision.coding_tasks) == 4


def test_ingestor_reads_titanic_manifest():
    manifest = CompetitionIngestor(COMPETITION_ROOT / "titanic").build_manifest()

    assert manifest.competition_name == "titanic"
    assert manifest.id_column == "PassengerId"
    assert manifest.target_column == "Survived"
    assert manifest.task_type == "classification"
    assert manifest.tables["train.csv"].row_count == 891
    assert manifest.tables["test.csv"].row_count == 418
    assert manifest.submission_columns == ["PassengerId", "Survived"]


def test_ingestor_ignores_runtime_artifacts(tmp_path: Path):
    competition_dir = tmp_path / "runtime_artifact_competition"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (competition_dir / ".pycache").mkdir()
    (competition_dir / ".pycache" / "junk.pyc").write_text("cached", encoding="utf-8")
    (competition_dir / "runs" / "0001").mkdir(parents=True)
    (competition_dir / "runs" / "0001" / "index.html").write_text("<html></html>", encoding="utf-8")
    (competition_dir / "experiments" / "exp1").mkdir(parents=True)
    (competition_dir / "experiments" / "exp1" / "submission.csv").write_text("PassengerId,Survived\n892,0\n", encoding="utf-8")

    manifest = CompetitionIngestor(competition_dir).build_manifest()

    assert "train.csv" in manifest.files
    assert "test.csv" in manifest.files
    assert "sample_submission.csv" in manifest.files
    assert all(".pycache" not in path for path in manifest.files)
    assert all(not path.startswith("runs/") for path in manifest.files)
    assert all(not path.startswith("experiments/") for path in manifest.files)


def test_submission_validator_accepts_and_rejects_titanic(tmp_path: Path):
    manifest = CompetitionIngestor(COMPETITION_ROOT / "titanic").build_manifest()
    validator = SubmissionValidator(manifest)

    valid = validator.validate(COMPETITION_ROOT / "titanic" / "sample_submission.csv")
    assert valid.ok

    bad_submission = tmp_path / "bad_submission.csv"
    bad_submission.write_text("PassengerId,Wrong\n892,\n", encoding="utf-8")
    invalid = validator.validate(bad_submission)
    assert not invalid.ok
    assert any("columns mismatch" in error for error in invalid.errors)


def test_brain_dry_loop_writes_artifacts_and_memory(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    decision = BrainOrchestrator(memory=memory).run_dry_loop(competition_dir)

    assert decision.profile.name == "tabular_classic"
    assert (competition_dir / "task_card.md").exists()
    assert (competition_dir / "metric_spec.json").exists()
    assert (competition_dir / "data_manifest.json").exists()
    assert (competition_dir / "experiment_plan.json").exists()
    assert (competition_dir / "brain_review.json").exists()
    assert (competition_dir / "experiments" / "competition_audit" / "coding_prompt.md").exists()
    assert (competition_dir / "runs" / "ledger.jsonl").exists()
    assert (competition_dir / "runs" / "index.html").exists()
    assert (competition_dir / "runs" / "0001_brain_plan" / "human_review.md").exists()
    assert (competition_dir / "runs" / "0002_sample_submission_validation" / "scorecard.json").exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    experiments_html = (competition_dir / "runs" / "console" / "experiments.html").read_text(encoding="utf-8")
    data_html = (competition_dir / "runs" / "console" / "data.html").read_text(encoding="utf-8")
    assert "Competition 控制台：titanic_copy" in html
    assert "返回 AutoKaggle 总控制台" in html
    assert 'href="console/experiments.html"' in html
    assert 'href="../index.html"' in experiments_html
    assert "Validate sample submission" in experiments_html
    assert "数据与 Intake" in data_html
    assert memory.query(competition_name="titanic_copy")[0].status == "validated"


def test_bank_churn_tabular_manifest_and_validator():
    manifest = CompetitionIngestor(COMPETITION_ROOT / "bank_churn").build_manifest()
    validation = SubmissionValidator(manifest).validate(
        COMPETITION_ROOT / "bank_churn" / "sample_submission.csv"
    )

    assert manifest.id_column == "id"
    assert manifest.target_column == "Exited"
    assert manifest.task_type == "classification"
    assert validation.ok


def test_tabular_baseline_runner_writes_ledger_entries(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    results = TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )

    assert [result.status for result in results] == ["validated", "validated"]
    assert (competition_dir / "experiments" / "sample_submission_baseline" / "run.py").exists()
    assert (competition_dir / "experiments" / "target_frequency_or_mean_baseline" / "submission.csv").exists()
    assert (competition_dir / "baseline_review.json").exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    experiments_html = (competition_dir / "runs" / "console" / "experiments.html").read_text(encoding="utf-8")
    assert "Competition 控制台：titanic_copy" in html
    assert "返回 AutoKaggle 总控制台" in html
    assert "Run sample_submission_baseline" in experiments_html
    assert "Run target_frequency_or_mean_baseline" in experiments_html
    assert "Select best baseline" in experiments_html
    assert "实验排行榜" in experiments_html
    assert "审计日志" in experiments_html
    assert "baseline_sample_submission_baseline" in experiments_html
    assert "<details class=\"audit-group\" open>" in experiments_html
    records = memory.query(competition_name="titanic_copy")
    assert len(records) == 2
    assert records[1].local_score is not None


def test_human_gate_defaults_to_transparent_continue(tmp_path: Path):
    missing = HumanGate.parse(tmp_path / "missing.md")
    assert missing.decision == "continue"
    assert not missing.is_intervention

    review = tmp_path / "human_review.md"
    review.write_text(
        "# Human Review\n\ndecision: patch_prompt\n\nnotes:\nUse 5-fold CV.\nPrefer LightGBM.\n",
        encoding="utf-8",
    )
    parsed = HumanGate.parse(review)
    assert parsed.decision == "patch_prompt"
    assert parsed.is_intervention
    assert "LightGBM" in parsed.notes


def test_remote_brain_reviewer_fallback_writes_plan_and_ledger(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    (competition_dir / "leaderboard_target.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "metric_name": "accuracy",
                "top_score": 0.99,
                "estimated_silver_score": 0.95,
                "gap_to_silver": 0.15,
                "target_policy": "silver_or_better",
            }
        ),
        encoding="utf-8",
    )
    latest_review = sorted((competition_dir / "runs").glob("*/human_review.md"))[-1]
    latest_review.write_text(
        "# Human Review\n\ndecision: patch_prompt\n\nnotes:\nTry LightGBM with 5-fold CV first.\n",
        encoding="utf-8",
    )

    result = RemoteBrainReviewer(competition_dir, memory=memory, use_llm=False).review()

    assert result.json_path.exists()
    assert result.markdown_path.exists()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))
    experiment = plan["recommended_experiments"][0]
    assert experiment["task_id"] in {"enhance_lightgbm_5fold_v1", "patched_lightgbm_5fold_v1"}
    assert experiment["skill_used"] == "tabular_optimization"
    assert experiment["harness"] == "stratified_gbdt_oof_harness"
    assert experiment["hypothesis"]
    assert experiment["validation_plan"]
    assert plan["leaderboard_target"]["estimated_silver_score"] == 0.95
    assert plan["skill_registry"]["skill_count"] >= 4
    assert plan["harness_registry"]["harness_count"] >= 4
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Review baselines and plan next LLM-guided experiments" in html
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "Remote Brain Experiment Plan" in markdown
    assert "Leaderboard Target" in markdown


def test_remote_brain_context_includes_mac_brain_mission(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    mission = {
        "phase": "execution_fidelity_and_diverse_portfolio",
        "do_not_repeat": ["No generic GBDT baseline."],
        "experiment_portfolio": [{"branch": "validation", "task_id_hint": "audit_plan_execution_fidelity_v1"}],
    }
    (competition_dir / "remote_brain_mission.json").write_text(json.dumps(mission), encoding="utf-8")
    (competition_dir / "mac_brain_diagnosis.json").write_text(
        json.dumps({"diagnoses": [{"id": "execution_fidelity_gap"}]}),
        encoding="utf-8",
    )

    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)
    context = reviewer._build_context()
    prompt = reviewer._build_prompt(context)

    assert context["remote_brain_mission"]["phase"] == "execution_fidelity_and_diverse_portfolio"
    assert context["mac_brain_diagnosis"]["diagnoses"][0]["id"] == "execution_fidelity_gap"
    assert "Mac Brain strategy contract" in prompt
    assert "do_not_repeat" in prompt
    assert "tabular_mlp" in prompt
    assert "star_specialist_lgbm" in prompt
    assert "star_specialist_threshold_tuning" in prompt
    assert "clean_oof_blend" in prompt


def test_remote_brain_fills_empty_coding_agent_task(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)

    experiment = reviewer._normalize_experiment(
        {
            "task_id": "audit_execution_fidelity_v1",
            "title": "Audit execution fidelity",
            "runner_kind": "cv_stability_audit",
            "hypothesis": "Planned features may not be implemented.",
            "validation_plan": "Compare plan against run.py and feature artifacts.",
            "evidence_needed": ["plan_vs_execution_diff.json", "actual_feature_list.json"],
            "coding_agent_task": "",
        },
        1,
    )

    assert experiment["coding_agent_task"]
    assert "audit_execution_fidelity_v1" in experiment["coding_agent_task"]

    nn_experiment = reviewer._normalize_experiment(
        {"task_id": "tabular_mlp_oof_v1", "title": "Train tabular MLP", "runner_kind": "tabular_mlp"},
        index=2,
    )
    specialist_experiment = reviewer._normalize_experiment(
        {"task_id": "star_specialist_lgbm_v1", "title": "STAR specialist", "runner_kind": "star_specialist_lgbm"},
        index=3,
    )
    threshold_experiment = reviewer._normalize_experiment(
        {
            "task_id": "star_specialist_threshold_tuning_v1",
            "title": "STAR specialist threshold tuning",
            "runner_kind": "star_specialist_threshold_tuning",
        },
        index=4,
    )
    blend_experiment = reviewer._normalize_experiment(
        {"task_id": "clean_diversity_blend_v1", "title": "Clean OOF blend", "runner_kind": "clean_oof_blend"},
        index=5,
    )
    assert nn_experiment["runner_kind"] == "tabular_mlp"
    assert nn_experiment["skill_used"] == "tabular_nn"
    assert specialist_experiment["runner_kind"] == "star_specialist_lgbm"
    assert specialist_experiment["skill_used"] == "class_specialist"
    assert threshold_experiment["runner_kind"] == "star_specialist_threshold_tuning"
    assert threshold_experiment["skill_used"] == "class_specialist"
    assert blend_experiment["runner_kind"] == "clean_oof_blend"
    assert blend_experiment["skill_used"] == "clean_ensemble"
    assert "plan_vs_execution_diff.json" in experiment["coding_agent_task"]
    assert "generic template" in experiment["coding_agent_task"]


def test_remote_brain_reviewer_uses_leaderboard_feedback_gap(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    LeaderboardFeedbackRecorder(competition_dir, memory=memory).record(
        public_score=0.40,
        leaderboard_rank=9999,
        submission_id="gap-demo",
        source="manual",
    )

    result = RemoteBrainReviewer(competition_dir, memory=memory, use_llm=False).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert plan["leaderboard_feedback"]["public_score"] == 0.40
    assert plan["leaderboard_diagnosis"]["risk_level"] == "high"
    assert plan["recommended_experiments"][0]["task_id"] == "enhance_leaderboard_gap_audit_v1"
    assert plan["recommended_experiments"][0]["runner_kind"] == "distribution_shift_audit"
    assert plan["recommended_experiments"][0]["skill_used"] == "validation_risk"
    assert plan["recommended_experiments"][0]["harness"]
    assert "distribution_shift_audit.json" in plan["recommended_experiments"][0]["evidence_needed"]
    assert plan["recommended_experiments"][0]["promotion_gate"]["manual_submit_allowed"] is False
    assert "Leaderboard Feedback" in result.markdown_path.read_text(encoding="utf-8")


def test_remote_brain_reviewer_uses_gap_audit_as_stronger_evidence(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    LeaderboardFeedbackRecorder(competition_dir, memory=memory).record(
        public_score=0.40,
        leaderboard_rank=9999,
        submission_id="gap-demo",
        source="manual",
    )
    LeaderboardGapAuditor(competition_dir, memory=memory).audit()

    result = RemoteBrainReviewer(competition_dir, memory=memory, use_llm=False).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))
    markdown = result.markdown_path.read_text(encoding="utf-8")

    assert plan["leaderboard_gap_audit"]["risk_level"] == "high"
    assert plan["leaderboard_diagnosis"]["risk_level"] == "high"
    assert plan["leaderboard_diagnosis"]["risks"]
    assert plan["recommended_experiments"][0]["runner_kind"] == "distribution_shift_audit"
    assert "Leaderboard Gap Audit" in markdown


def test_remote_brain_fallback_avoids_completed_recommendations(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    completed_dir = competition_dir / "experiments" / "enhance_lightgbm_5fold_v1"
    completed_dir.mkdir(parents=True)
    (completed_dir / "validation_report.json").write_text(
        json.dumps({"status": "completed", "metric_name": "accuracy", "local_score": 0.81}),
        encoding="utf-8",
    )

    result = RemoteBrainReviewer(competition_dir, memory=memory, use_llm=False).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert plan["recommended_experiments"][0]["task_id"] == "post_feedback_catboost_5fold_v1"
    assert plan["recommended_experiments"][0]["runner_kind"] == "catboost"


def test_remote_brain_fallback_uses_best_completed_validation_report(tmp_path: Path):
    competition_dir = tmp_path / "bank_churn_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "bank_churn"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    exp_dir = competition_dir / "experiments" / "enhance_lightgbm_5fold_v1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "metric_name": "roc_auc",
                "local_score": 0.8894,
                "runner_kind": "lightgbm",
            }
        ),
        encoding="utf-8",
    )
    (exp_dir / "validator_result.json").write_text(
        json.dumps({"ok": True, "errors": [], "warnings": []}),
        encoding="utf-8",
    )

    result = RemoteBrainReviewer(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
        use_llm=False,
    ).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert plan["current_best_baseline"]["task_id"] == "enhance_lightgbm_5fold_v1"
    assert plan["current_best_baseline"]["local_score"] == 0.8894
    assert plan["current_best_baseline"]["source"] == "validation_report"
    assert plan["recommended_experiments"][0]["task_id"] == "post_feedback_catboost_5fold_v1"


def test_remote_brain_pause_replan_avoids_completed_recommendations(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "champion": {
                    "task_id": "champion_blend",
                    "metric_name": "accuracy",
                    "local_score": 0.832,
                }
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "submission_decision_review.json").write_text(
        json.dumps(
            {
                "decision": "pause_manual_submit",
                "issues": ["stability evidence missing"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    for task_id in ["stability_replan_after_pause_v1", "post_pause_tuned_random_forest_v1"]:
        experiment_dir = competition_dir / "experiments" / task_id
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "validation_report.json").write_text(
            json.dumps({"status": "completed", "metric_name": "accuracy", "local_score": 0.81}),
            encoding="utf-8",
        )

    result = RemoteBrainReviewer(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
        use_llm=False,
    ).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert plan["recommended_experiments"][0]["task_id"] == "post_pause_cv_stability_audit_v2"
    assert plan["recommended_experiments"][0]["runner_kind"] == "cv_stability_audit"


def test_remote_brain_regularized_blend_plan_requires_stability_evidence(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "champion": {
                    "task_id": "champion_blend",
                    "metric_name": "accuracy",
                    "local_score": 0.82,
                }
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "submission_decision_review.json").write_text(
        json.dumps({"decision": "pause_manual_submit", "issues": ["manual submit paused"]}),
        encoding="utf-8",
    )
    for task_id in [
        "stability_replan_after_pause_v1",
        "post_pause_tuned_random_forest_v1",
        "post_pause_cv_stability_audit_v2",
    ]:
        experiment_dir = competition_dir / "experiments" / task_id
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "validation_report.json").write_text(
            json.dumps({"status": "completed", "metric_name": "accuracy", "local_score": 0.81}),
            encoding="utf-8",
        )

    result = RemoteBrainReviewer(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
        use_llm=False,
    ).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))
    experiment = plan["recommended_experiments"][0]

    assert experiment["task_id"] == "post_pause_regularized_blend_v1"
    assert experiment["runner_kind"] == "regularized_blend"
    assert experiment["skill_used"] == "validation_risk"
    assert experiment["harness"] == "regularized_oof_blend_harness"
    assert "regularized_blend_report.json" in experiment["evidence_needed"]
    assert "oof_predictions.csv" in experiment["evidence_needed"]
    assert "seed_std" in experiment["evidence_needed"]
    assert experiment["promotion_gate"]["seed_std"] == "<= 0.010"
    assert experiment["promotion_gate"]["fold_std"] == "<= 0.030"
    assert experiment["promotion_gate"]["train_valid_gap"] == "<= 0.040"
    assert experiment["promotion_gate"]["max_model_correlation"] == "<= 0.995"
    assert experiment["promotion_gate"]["max_local_score_drop"] == 0.03
    assert "min_local_score_delta" not in experiment["promotion_gate"]


def test_remote_brain_reviewer_replans_after_paused_submission_decision(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "champion": {
                    "task_id": "champion_blend",
                    "status": "champion_selected",
                    "metric_name": "accuracy",
                    "local_score": 0.832,
                }
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_feedback.json").write_text(
        json.dumps(
            {
                "submission_target": "recommended",
                "candidate_task_id": "stability_first_search_v1",
                "metric_name": "accuracy",
                "public_score": 0.81234,
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "experiment_queue.json").write_text(
        json.dumps(
            {
                "queue": [
                    {
                        "task_id": "champion_blend_lb_submit",
                        "status": "blocked",
                        "action_type": "manual_submit",
                    }
                ],
                "next_runnable": None,
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "submission_decision_review.json").write_text(
        json.dumps(
            {
                "status": "needs_review",
                "queue_task_id": "champion_blend_lb_submit",
                "decision": "pause_manual_submit",
                "issues": ["CV stability audit risk is medium."],
                "warnings": ["Champion local CV is above recommended, but public feedback belongs to recommended."],
            }
        ),
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    result = RemoteBrainReviewer(competition_dir, memory=memory, use_llm=False).review()
    plan = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert plan["submission_decision_review"]["decision"] == "pause_manual_submit"
    assert plan["recommended_experiments"][0]["task_id"] == "stability_replan_after_pause_v1"
    assert plan["recommended_experiments"][0]["runner_kind"] == "cv_stability_audit"
    assert plan["recommended_experiments"][0]["promotion_gate"]["manual_submit_allowed"] is False
    assert "CV stability audit risk is medium" in plan["recommended_experiments"][0]["coding_agent_task"]
    assert "Submission Decision Review" in result.markdown_path.read_text(encoding="utf-8")

    queue_result = ExperimentQueueBuilder(competition_dir, memory=memory).build()
    queue = json.loads(queue_result.queue_path.read_text(encoding="utf-8"))
    assert queue["next_runnable"]["task_id"] == "stability_replan_after_pause_v1"
    assert queue["next_runnable"]["action_type"] == "audit"
    assert queue["next_runnable"]["runner_kind"] == "cv_stability_audit"

    completed_dir = competition_dir / "experiments" / "stability_replan_after_pause_v1"
    completed_dir.mkdir(parents=True)
    (completed_dir / "validation_report.json").write_text(
        json.dumps({"experiment": "stability_replan_after_pause_v1", "status": "completed"}),
        encoding="utf-8",
    )
    next_result = RemoteBrainReviewer(competition_dir, memory=memory, use_llm=False).review()
    next_plan = json.loads(next_result.json_path.read_text(encoding="utf-8"))
    assert next_plan["recommended_experiments"][0]["task_id"] == "post_pause_tuned_random_forest_v1"
    assert next_plan["recommended_experiments"][0]["runner_kind"] == "tuned_random_forest"


def test_remote_brain_normalize_preserves_submission_decision_context(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)
    context = {
        "baseline_review": {},
        "leaderboard_feedback": {},
        "submission_decision_review": {
            "decision": "pause_manual_submit",
            "issues": ["Public score is outside the seed-level confidence interval."],
        },
        "active_human_interventions": [],
    }

    normalized = reviewer._normalize_plan(
        {
            "submission_decision_review": None,
            "recommended_experiments": [
                {
                    "task_id": "distribution_shift_audit_v1",
                    "title": "Distribution shift audit",
                    "coding_agent_task": "Audit drift.",
                }
            ],
        },
        context,
    )

    assert normalized["submission_decision_review"]["decision"] == "pause_manual_submit"


def test_remote_brain_normalizes_structured_experiment_fields(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)
    normalized = reviewer._normalize_plan(
        {
            "recommended_experiments": [
                {
                    "task_id": "ambiguous_task_v1",
                    "title": "Try a careful classifier",
                    "runner_kind": "drift_audit",
                    "evidence_needed": "distribution_shift_audit.json",
                    "promotion_gate": {"validator_must_pass": True},
                    "coding_agent_task": "Use the explicit runner kind, not this vague text.",
                }
            ],
        },
        {"baseline_review": {}, "leaderboard_feedback": {}, "active_human_interventions": []},
    )
    item = normalized["recommended_experiments"][0]
    assert item["runner_kind"] == "distribution_shift_audit"
    assert item["evidence_needed"] == ["distribution_shift_audit.json"]
    assert item["promotion_gate"]["validator_must_pass"] is True


def test_remote_brain_renames_recommended_completed_task_ids(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    completed_dir = competition_dir / "experiments" / "experiment_1"
    completed_dir.mkdir(parents=True)
    (completed_dir / "validation_report.json").write_text(
        json.dumps({"experiment": "experiment_1", "status": "completed", "metric_name": "accuracy", "local_score": 0.9}),
        encoding="utf-8",
    )
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)
    context = reviewer._build_context()

    normalized = reviewer._normalize_plan(
        {
            "recommended_experiments": [
                {
                    "task_id": "experiment_1",
                    "title": "Feature engineering LightGBM",
                    "runner_kind": "lightgbm",
                    "hypothesis": "Add color indices and redshift interactions.",
                }
            ]
        },
        context,
    )
    item = normalized["recommended_experiments"][0]

    assert item["task_id"] != "experiment_1"
    assert item["original_task_id"] == "experiment_1"
    assert "lightgbm" in item["task_id"]


def test_remote_brain_recomputes_leaderboard_target_gaps(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)

    normalized = reviewer._normalize_plan(
        {
            "current_best_baseline": {"task_id": "experiment_1", "metric_name": "accuracy", "local_score": 0.9639},
            "leaderboard_target": {"top_score": 0.97173, "silver_score": 0.97148, "gap_to_silver": 0.04446},
            "recommended_experiments": [{"task_id": "next_lightgbm", "runner_kind": "lightgbm"}],
        },
        {"baseline_review": {}, "leaderboard_feedback": {}, "active_human_interventions": []},
    )

    assert normalized["leaderboard_target"]["gap_to_silver"] == pytest.approx(0.00758)
    assert normalized["leaderboard_target"]["gap_to_top"] == pytest.approx(0.00783)


def test_remote_brain_strategy_guardrail_prioritizes_blend_when_oof_pool_ready(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)
    context = {
        "baseline_review": {},
        "leaderboard_feedback": {},
        "active_human_interventions": [],
        "candidate_pool": {
            "ensemble_candidates": [
                {"task_id": "lightgbm_oof", "has_oof": True, "validator_ok": True},
                {"task_id": "catboost_oof", "has_oof": True, "validator_ok": True},
            ]
        },
    }

    normalized = reviewer._normalize_plan(
        {
            "recommended_experiments": [
                {
                    "task_id": "another_lightgbm",
                    "title": "Another LightGBM",
                    "runner_kind": "lightgbm",
                }
            ]
        },
        context,
    )
    first = normalized["recommended_experiments"][0]

    assert first["runner_kind"] == "regularized_blend"
    assert first["skill_used"] == "ensemble_strategy"
    assert first["strategy_guardrail"] == "candidate_pool_has_multiple_oof_models"
    assert first["input_candidates"] == ["lightgbm_oof", "catboost_oof"]


def test_remote_brain_fallback_uses_strategy_guardrail_for_oof_pool(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    reviewer = RemoteBrainReviewer(competition_dir, memory=CompetitionMemory(tmp_path / "memory"), use_llm=False)

    plan = reviewer._fallback_plan(
        {
            "baseline_review": {},
            "leaderboard_feedback": {},
            "leaderboard_feedback_freshness": {},
            "active_human_interventions": [],
            "candidate_pool": {
                "ensemble_candidates": [
                    {"task_id": "lightgbm_oof", "has_oof": True, "validator_ok": True},
                    {"task_id": "catboost_oof", "has_oof": True, "validator_ok": True},
                ]
            },
        }
    )
    first = plan["recommended_experiments"][0]

    assert first["runner_kind"] == "regularized_blend"
    assert first["strategy_guardrail"] == "candidate_pool_has_multiple_oof_models"


def test_enhancement_runner_writes_skip_or_validated_ledger_entry(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        """{
  "current_best_baseline": {"task_id": "baseline", "metric_name": "accuracy", "local_score": 0.7},
  "next_action": "generate_enhancement_tasks",
  "recommended_experiments": [
    {
      "task_id": "random_forest_baseline_1",
      "title": "RandomForest test",
      "coding_agent_task": "Run a random forest enhancement."
    }
  ]
}
""",
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    result = EnhancementRunner(competition_dir, memory=memory).run_first_recommendation()

    assert result.status in {"validated", "skipped"}
    assert (competition_dir / "experiments" / "random_forest_baseline_1" / "run.py").exists()
    assert (competition_dir / "enhancement_review.json").exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Run enhancement" in html
    assert "Review latest enhancement experiment" in html
    assert memory.query(competition_name="titanic_copy")


def test_enhancement_runner_consumes_experiment_queue_next_runnable(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "current_best_baseline": {"task_id": "baseline", "metric_name": "accuracy", "local_score": 0.7},
                "next_action": "recommend_experiments",
                "recommended_experiments": [
                    {
                        "task_id": "manual_submit_first",
                        "title": "Submit current champion to Kaggle leaderboard",
                        "coding_agent_task": "Submit current champion and record public score.",
                    },
                    {
                        "task_id": "queued_random_forest_v1",
                        "title": "Queued RandomForest test",
                        "coding_agent_task": "Run a random forest enhancement.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    memory = CompetitionMemory(tmp_path / "memory")
    ExperimentQueueBuilder(competition_dir, memory=memory).build()
    result = EnhancementRunner(competition_dir, memory=memory).run_next_recommendation()
    queue = json.loads((competition_dir / "experiment_queue.json").read_text(encoding="utf-8"))

    assert result.task_id == "queued_random_forest_v1"
    assert result.status in {"validated", "skipped"}
    queued = {item["task_id"]: item for item in queue["queue"]}
    assert queued["queued_random_forest_v1"]["status"] == (
        "completed" if result.status == "validated" else "blocked"
    )
    assert queued["manual_submit_first"]["status"] == "manual_gate"
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "下一批实验队列" in html
    assert "queued_random_forest_v1" in html


def test_enhancement_runner_runs_cv_stability_audit_queue_item(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (competition_dir / "recommended_submission.csv").write_text(
        (source / "sample_submission.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    stability_dir = competition_dir / "experiments" / "stability_first_search_v1"
    stability_dir.mkdir(parents=True)
    (stability_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "experiment": "stability_first_search_v1",
                "status": "completed",
                "metric_name": "accuracy",
                "local_score": 0.837,
                "best_model": {
                    "model": "xgboost",
                    "score": 0.837,
                    "fold_scores": [0.84, 0.83, 0.82, 0.85, 0.84, 0.83],
                    "seed_scores": [
                        {"seed": 42, "score": 0.836, "fold_scores": [0.84, 0.83, 0.82]},
                        {"seed": 123, "score": 0.84, "fold_scores": [0.85, 0.84, 0.83]},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_feedback.json").write_text(
        json.dumps(
            {
                "submission_target": "recommended",
                "candidate_task_id": "stability_first_search_v1",
                "public_score": 0.834,
                "leaderboard_rank": 1000,
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "next_action": "recommend_experiments",
                "recommended_experiments": [
                    {
                        "task_id": "cv_stability_audit_v1",
                        "title": "CV stability audit",
                        "coding_agent_task": "Re-run CV stability audit.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    ExperimentQueueBuilder(competition_dir, memory=memory).build()
    result = EnhancementRunner(competition_dir, memory=memory).run_next_recommendation()
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    audit = json.loads((result.experiment_dir / "cv_stability_audit.json").read_text(encoding="utf-8"))
    queue = json.loads((competition_dir / "experiment_queue.json").read_text(encoding="utf-8"))

    assert result.task_id == "cv_stability_audit_v1"
    assert result.status == "validated"
    assert report["runner_kind"] == "cv_stability_audit"
    assert report["risk_level"] in {"low", "medium", "high"}
    assert audit["seed_mean"] is not None
    assert audit["leaderboard_feedback"]["public_score"] == 0.834
    assert result.validator_result.ok is True
    assert queue["queue"][0]["status"] == "completed"


def test_enhancement_runner_routes_model_specific_templates(tmp_path: Path):
    runner = EnhancementRunner(tmp_path)

    assert runner._runner_kind({"runner_kind": "distribution_shift_audit", "title": "Vague task"}) == "distribution_shift_audit"
    assert runner._runner_kind({"runner_kind": "regularized_blend", "title": "Vague task"}) == "regularized_blend"
    assert runner._runner_kind({"title": "Train LightGBM with 5-fold CV"}) == "lightgbm"
    assert runner._runner_kind({"task_id": "cv_stability_audit_v1"}) == "cv_stability_audit"
    assert runner._runner_kind({"task_id": "distribution_shift_audit_v1"}) == "distribution_shift_audit"
    assert runner._runner_kind({"task_id": "overfitting_detection_experiment_v1"}) == "overfitting_audit"
    assert runner._runner_kind({"task_id": "champion_blend_with_regularization_v1"}) == "regularized_blend"
    assert runner._runner_kind({"description": "Use CatBoost for categorical features"}) == "catboost"
    assert runner._runner_kind({"coding_agent_task": "Implement XGBoost classifier"}) == "xgboost"
    assert runner._runner_kind({"title": "RandomForest with grid search tuning"}) == "tuned_random_forest"
    assert runner._runner_kind({"runner_kind": "tabular_mlp"}) == "tabular_mlp"
    assert runner._runner_kind({"title": "Train a tabular MLP"}) == "tabular_mlp"
    assert runner._runner_kind({"runner_kind": "star_specialist_lgbm"}) == "star_specialist_lgbm"
    assert runner._runner_kind({"runner_kind": "star_specialist_threshold_tuning"}) == "star_specialist_threshold_tuning"
    assert runner._runner_kind({"title": "STAR specialist threshold tuning"}) == "star_specialist_threshold_tuning"
    assert runner._runner_kind({"title": "Clean OOF blend"}) == "clean_oof_blend"

    script = runner._script_for(
        {
            "task_id": "xgboost_test",
            "title": "Implement XGBoost classifier",
            "coding_agent_task": "Use XGBoost with the current preprocessing.",
        }
    )
    assert "XGBClassifier" in script
    assert 'model_kind = "xgboost"' in script
    assert "pipe.fit(X, y_model)" in script
    assert "pipe.fit(X, y)\n" not in script

    lightgbm_script = runner._script_for({"title": "Try a LightGBM model"})
    catboost_script = runner._script_for({"title": "Try a CatBoost model"})
    tuned_rf_script = runner._script_for({"title": "Tune RandomForest with grid search"})
    assert "LGBMClassifier" in lightgbm_script
    assert 'model_kind = "lightgbm"' in lightgbm_script
    assert "actual_feature_list.json" in lightgbm_script
    assert "plan_vs_execution_diff.json" in lightgbm_script
    assert "u-g" in lightgbm_script
    assert "log1p_redshift" in lightgbm_script
    assert "_x_redshift" in lightgbm_script
    assert "CatBoostClassifier" in catboost_script
    assert 'model_kind = "catboost"' in catboost_script
    assert "n_estimators=500" in tuned_rf_script
    assert "max_depth=5" in tuned_rf_script

    assert "distribution_shift_audit.json" in runner._script_for({"task_id": "distribution_shift_audit_v1"})
    assert "overfitting_audit.json" in runner._script_for({"task_id": "overfitting_detection_experiment_v1"})
    blend_script = runner._script_for({"task_id": "champion_blend_with_regularization_v1"})
    assert "regularized_blend_report.json" in blend_script
    assert "existing_oof_and_submission_artifacts" in blend_script
    assert "weighted_submission_blend" in blend_script
    assert "plan_vs_execution_diff.json" in runner._script_for({"task_id": "audit_execution_fidelity_v1"})
    assert "per_class_oof_report.json" in runner._script_for({"task_id": "per_class_oof_audit_v1"})
    assert "oof_diversity_report.json" in runner._script_for({"task_id": "oof_diversity_matrix_v1"})
    assert "per_class_oof_report.json" in runner._script_for(
        {
            "task_id": "per_class_oof_audit_v1",
            "runner_kind": "cv_stability_audit",
            "coding_agent_task": "Compute per_class_oof_report.json and also write plan_vs_execution_diff.json if needed.",
        }
    )
    assert "oof_diversity_report.json" in runner._script_for(
        {
            "task_id": "oof_diversity_matrix_v1",
            "runner_kind": "regularized_blend",
            "coding_agent_task": "Compute oof_diversity_report.json and write plan_vs_execution_diff.json if needed.",
        }
    )
    assert "nn_training_report.json" in runner._script_for({"runner_kind": "tabular_mlp"})
    assert "model_config.json" in runner._script_for({"runner_kind": "tabular_resnet"})
    assert "specialist_report.json" in runner._script_for({"runner_kind": "star_specialist_lgbm"})
    threshold_script = runner._script_for({"runner_kind": "star_specialist_threshold_tuning"})
    assert "specialist_report.json" in threshold_script
    assert "threshold_frontier" in threshold_script
    assert "promotion_min_score" in threshold_script
    assert "clean_blend_report.json" in runner._script_for({"runner_kind": "clean_oof_blend"})
    assert "skipped_candidates.json" in runner._script_for({"runner_kind": "classwise_blend"})


def test_enhancement_runner_records_timeout_as_failed_experiment(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    manifest = CompetitionIngestor(competition_dir).build_manifest()

    def fake_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 1800), output="started", stderr="still running")

    monkeypatch.setattr("multi_agents.orchestration.enhancement_runner.subprocess.run", fake_timeout)

    result = EnhancementRunner(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run_experiment(
        {"task_id": "timeout_runner_v1", "runner_kind": "lightgbm"},
        manifest,
        {"next_action": "recommend_experiments"},
    )
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    log_text = result.run_log.read_text(encoding="utf-8")

    assert result.status == "failed"
    assert report["status"] == "failed"
    assert "timed out" in report["issues"][0]
    assert "returncode=124" in log_text
    assert result.validator_result.ok is False


def _write_synthetic_stellar_competition(competition_dir: Path) -> None:
    competition_dir.mkdir(parents=True, exist_ok=True)
    rows = ["id,alpha,delta,u,g,r,i,z,redshift,spectral_type,class"]
    labels = ["GALAXY", "QSO", "STAR"] * 5
    for index, label in enumerate(labels, start=1):
        offset = {"GALAXY": 0.0, "QSO": 1.0, "STAR": 2.0}[label]
        rows.append(
            f"{index},{10+index},{-5+index},"
            f"{1.0+offset},{1.3+offset},{1.6+offset},{1.9+offset},{2.2+offset},"
            f"{0.1+offset/10},type_{label},{label}"
        )
    test_rows = [
        "id,alpha,delta,u,g,r,i,z,redshift,spectral_type",
        "101,1,1,1.0,1.3,1.6,1.9,2.2,0.1,type_GALAXY",
        "102,2,2,2.0,2.3,2.6,2.9,3.2,0.2,type_QSO",
        "103,3,3,3.0,3.3,3.6,3.9,4.2,0.3,type_STAR",
    ]
    (competition_dir / "overview.txt").write_text("Metric: accuracy\n", encoding="utf-8")
    (competition_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (competition_dir / "test.csv").write_text("\n".join(test_rows) + "\n", encoding="utf-8")
    (competition_dir / "sample_submission.csv").write_text("id,class\n101,GALAXY\n102,QSO\n103,STAR\n", encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(competition_dir / "data_manifest.json")


def test_tabular_mlp_runner_produces_oof_and_nn_reports(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    competition_dir = tmp_path / "stellar"
    _write_synthetic_stellar_competition(competition_dir)
    manifest = CompetitionIngestor(competition_dir).build_manifest()

    result = EnhancementRunner(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run_experiment(
        {"task_id": "tabular_mlp_oof_v1", "runner_kind": "tabular_mlp"},
        manifest,
        {"next_action": "run_experiments"},
    )
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))

    assert result.status == "validated"
    assert report["runner_kind"] == "tabular_mlp"
    assert (result.experiment_dir / "oof_predictions.csv").exists()
    assert (result.experiment_dir / "nn_training_report.json").exists()
    assert (result.experiment_dir / "model_config.json").exists()


def test_star_specialist_runner_produces_specialist_reports(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    competition_dir = tmp_path / "stellar"
    _write_synthetic_stellar_competition(competition_dir)
    manifest = CompetitionIngestor(competition_dir).build_manifest()

    result = EnhancementRunner(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run_experiment(
        {"task_id": "star_specialist_lgbm_v1", "runner_kind": "star_specialist_lgbm"},
        manifest,
        {"next_action": "run_experiments"},
    )
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    specialist = json.loads((result.experiment_dir / "specialist_report.json").read_text(encoding="utf-8"))

    assert result.status == "validated"
    assert report["runner_kind"] == "star_specialist_lgbm"
    assert specialist["target_class"] == "STAR"
    assert (result.experiment_dir / "per_class_oof_report.json").exists()
    assert (result.experiment_dir / "oof_predictions.csv").exists()


def test_star_specialist_threshold_tuning_runner_searches_threshold(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    competition_dir = tmp_path / "stellar"
    _write_synthetic_stellar_competition(competition_dir)
    manifest = CompetitionIngestor(competition_dir).build_manifest()

    result = EnhancementRunner(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run_experiment(
        {"task_id": "star_specialist_threshold_tuning_v1", "runner_kind": "star_specialist_threshold_tuning"},
        manifest,
        {"next_action": "run_experiments"},
    )
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    specialist = json.loads((result.experiment_dir / "specialist_report.json").read_text(encoding="utf-8"))

    assert result.status == "validated"
    assert report["runner_kind"] == "star_specialist_threshold_tuning"
    assert specialist["runner_kind"] == "star_specialist_threshold_tuning"
    assert specialist["target_class"] == "STAR"
    assert specialist["threshold_frontier"]
    assert specialist["threshold_search_floor"] <= specialist["base_score"]
    assert (result.experiment_dir / "per_class_oof_report.json").exists()
    assert (result.experiment_dir / "oof_predictions.csv").exists()


def test_clean_oof_blend_runner_filters_invalid_candidates(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    competition_dir = tmp_path / "stellar"
    _write_synthetic_stellar_competition(competition_dir)
    manifest = CompetitionIngestor(competition_dir).build_manifest()
    train = (competition_dir / "train.csv").read_text(encoding="utf-8").splitlines()
    train_labels = [line.rsplit(",", 1)[1] for line in train[1:]]

    for name, score, labels in [
        ("valid_a", 0.80, train_labels),
        ("valid_b", 0.70, list(reversed(train_labels))),
    ]:
        exp_dir = competition_dir / "experiments" / name
        exp_dir.mkdir(parents=True)
        (exp_dir / "validation_report.json").write_text(
            json.dumps({"experiment": name, "runner_kind": "lightgbm", "status": "completed", "metric_name": "accuracy", "local_score": score}),
            encoding="utf-8",
        )
        (exp_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
        (exp_dir / "oof_predictions.csv").write_text(
            "id,class,oof_selected\n" + "\n".join(f"{idx},{truth},{pred}" for idx, (truth, pred) in enumerate(zip(train_labels, labels), start=1)) + "\n",
            encoding="utf-8",
        )
        (exp_dir / "submission.csv").write_text("id,class\n101,GALAXY\n102,QSO\n103,STAR\n", encoding="utf-8")
    invalid_dir = competition_dir / "experiments" / "invalid_nan"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "validation_report.json").write_text(json.dumps({"status": "completed", "local_score": 0.99}), encoding="utf-8")
    (invalid_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (invalid_dir / "oof_predictions.csv").write_text("id,class,oof_selected\n1,GALAXY,\n", encoding="utf-8")
    (invalid_dir / "submission.csv").write_text("id,class\n101,GALAXY\n102,QSO\n103,STAR\n", encoding="utf-8")

    result = EnhancementRunner(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run_experiment(
        {"task_id": "clean_diversity_blend_v1", "runner_kind": "clean_oof_blend"},
        manifest,
        {"next_action": "run_experiments"},
    )
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    blend = json.loads((result.experiment_dir / "clean_blend_report.json").read_text(encoding="utf-8"))

    assert result.status == "validated"
    assert report["runner_kind"] == "clean_oof_blend"
    assert blend["valid_candidate_count"] == 2
    assert any(item["task_id"] == "invalid_nan" for item in blend["skipped_candidates"])
    assert (result.experiment_dir / "skipped_candidates.json").exists()


def test_enhancement_runner_executes_execution_fidelity_audit(tmp_path: Path):
    competition_dir = tmp_path / "mission_competition"
    competition_dir.mkdir()
    (competition_dir / "overview.txt").write_text("Metric: accuracy\n", encoding="utf-8")
    (competition_dir / "train.csv").write_text(
        "id,u,g,r,i,z,redshift,class\n"
        "1,1,2,3,4,5,0.1,GALAXY\n"
        "2,2,3,4,5,6,0.2,QSO\n"
        "3,3,4,5,6,7,0.3,STAR\n",
        encoding="utf-8",
    )
    (competition_dir / "test.csv").write_text(
        "id,u,g,r,i,z,redshift\n4,1,2,3,4,5,0.1\n",
        encoding="utf-8",
    )
    (competition_dir / "sample_submission.csv").write_text("id,class\n4,GALAXY\n", encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    prior_dir = competition_dir / "experiments" / "claimed_feature_lgbm"
    prior_dir.mkdir(parents=True)
    (prior_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "experiment": "claimed_feature_lgbm",
                "runner_kind": "lightgbm",
                "status": "completed",
                "metric_name": "accuracy",
                "local_score": 0.96,
                "feature_count": 10,
            }
        ),
        encoding="utf-8",
    )
    (prior_dir / "run.py").write_text("def add_features(df):\n    return df\n", encoding="utf-8")
    (prior_dir / "feature_importance.csv").write_text("feature,importance\nu,1.0\ng,0.5\n", encoding="utf-8")
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "next_action": "run_experiments",
                "recommended_experiments": [
                    {
                        "task_id": "audit_execution_fidelity_v1",
                        "title": "Audit execution fidelity",
                        "runner_kind": "cv_stability_audit",
                        "hypothesis": "Check whether u-g and redshift features were implemented.",
                        "validation_plan": "Compare plans with run.py and feature artifacts.",
                        "evidence_needed": ["plan_vs_execution_diff.json", "actual_feature_list.json"],
                        "coding_agent_task": "Compare planned color/redshift features against actual run.py implementation and write plan_vs_execution_diff.json.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "remote_brain_mission.json").write_text(
        json.dumps({"primary_question": "Did u-g, g-r, redshift interactions run?", "do_not_repeat": []}),
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    ExperimentQueueBuilder(competition_dir, memory=memory).build()
    result = EnhancementRunner(competition_dir, memory=memory).run_next_recommendation()
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    diff = json.loads((result.experiment_dir / "plan_vs_execution_diff.json").read_text(encoding="utf-8"))

    assert result.task_id == "audit_execution_fidelity_v1"
    assert result.status == "validated"
    assert result.validator_result.ok is True
    assert report["runner_kind"] == "execution_fidelity_audit"
    assert (result.experiment_dir / "actual_feature_list.json").exists()
    assert (result.experiment_dir / "feature_implementation_audit.json").exists()
    assert "redshift" in diff["planned_feature_terms"]
    assert diff["issues"]


def test_experiment_queue_preserves_structured_runner_fields(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "next_action": "recommend_experiments",
                "recommended_experiments": [
                    {
                        "task_id": "structured_runner_task",
                        "title": "Ambiguous title",
                        "skill_used": "validation_risk",
                        "harness": "stratified_cv_stability_harness",
                        "hypothesis": "Audit whether the current champion is overfit.",
                        "runner_kind": "overfitting_audit",
                        "expected_gain": "risk_reduction",
                        "risk": "low",
                        "compute_cost": "low",
                        "validation_plan": "Compare train and validation behavior under the audit harness.",
                        "evidence_needed": ["overfitting_audit.json", "validation_report.json"],
                        "promotion_gate": {"max_train_valid_gap": 0.05},
                        "coding_agent_task": "Use explicit runner kind.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ExperimentQueueBuilder(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).build()
    queue = json.loads(result.queue_path.read_text(encoding="utf-8"))
    item = queue["queue"][0]
    assert item["skill_used"] == "validation_risk"
    assert item["harness"] == "stratified_cv_stability_harness"
    assert item["hypothesis"] == "Audit whether the current champion is overfit."
    assert item["runner_kind"] == "overfitting_audit"
    assert item["validation_plan"] == "Compare train and validation behavior under the audit harness."
    assert item["action_type"] == "audit"
    assert item["evidence_needed"] == ["overfitting_audit.json", "validation_report.json"]
    assert item["promotion_gate"]["max_train_valid_gap"] == 0.05
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "Skill: validation_risk" in markdown
    assert "Harness: stratified_cv_stability_harness" in markdown
    assert "Hypothesis: Audit whether the current champion is overfit." in markdown
    assert "Runner kind: overfitting_audit" in markdown
    assert "overfitting_audit.json" in markdown


def test_promotion_gate_evaluator_promotes_candidate_that_satisfies_gate(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "champion": {
                    "task_id": "current_champion",
                    "metric_name": "accuracy",
                    "local_score": 0.80,
                }
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "structured_promote_v1",
                        "runner_kind": "regularized_blend",
                        "evidence_needed": ["validation_report.json", "validator_result.json", "submission.csv"],
                        "promotion_gate": {
                            "validator_must_pass": True,
                            "min_local_score_delta": 0.002,
                            "manual_submit_allowed": False,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_dir = competition_dir / "experiments" / "structured_promote_v1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "experiment": "structured_promote_v1",
                "runner_kind": "regularized_blend",
                "status": "completed",
                "metric_name": "accuracy",
                "local_score": 0.812,
            }
        ),
        encoding="utf-8",
    )
    (exp_dir / "validator_result.json").write_text(
        json.dumps({"ok": True, "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    (exp_dir / "submission.csv").write_text(
        (source / "sample_submission.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = PromotionGateEvaluator(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).evaluate()
    review = json.loads(result.review_path.read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert review["decision"] == "promote_candidate"
    assert review["promoted_candidate"]["task_id"] == "structured_promote_v1"
    assert result.promoted_submission_path and result.promoted_submission_path.exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "候选晋级审核" in html
    assert "structured_promote_v1" in html


def test_promotion_gate_evaluator_holds_candidate_when_gate_fails(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps({"champion": {"task_id": "current_champion", "metric_name": "accuracy", "local_score": 0.80}}),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "structured_hold_v1",
                        "runner_kind": "regularized_blend",
                        "evidence_needed": ["validation_report.json", "validator_result.json", "submission.csv"],
                        "promotion_gate": {
                            "validator_must_pass": True,
                            "min_local_score_delta": 0.02,
                        },
                    },
                    {
                        "task_id": "diagnostic_done_v1",
                        "runner_kind": "distribution_shift_audit",
                        "evidence_needed": ["distribution_shift_audit.json"],
                        "promotion_gate": {"max_drift_score_reviewed": True},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    promote_dir = competition_dir / "experiments" / "structured_hold_v1"
    promote_dir.mkdir(parents=True)
    (promote_dir / "validation_report.json").write_text(
        json.dumps({"experiment": "structured_hold_v1", "runner_kind": "regularized_blend", "status": "completed", "metric_name": "accuracy", "local_score": 0.805}),
        encoding="utf-8",
    )
    (promote_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (promote_dir / "submission.csv").write_text((source / "sample_submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
    diagnostic_dir = competition_dir / "experiments" / "diagnostic_done_v1"
    diagnostic_dir.mkdir(parents=True)
    (diagnostic_dir / "validation_report.json").write_text(
        json.dumps({"experiment": "diagnostic_done_v1", "runner_kind": "distribution_shift_audit", "status": "completed", "metric_name": "accuracy", "local_score": 0.79}),
        encoding="utf-8",
    )
    (diagnostic_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (diagnostic_dir / "submission.csv").write_text((source / "sample_submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (diagnostic_dir / "distribution_shift_audit.json").write_text(json.dumps({"max_drift_score": 1.2}), encoding="utf-8")

    result = PromotionGateEvaluator(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).evaluate()
    review = json.loads(result.review_path.read_text(encoding="utf-8"))
    decisions = {item["task_id"]: item["decision"] for item in review["evaluations"]}

    assert result.status == "needs_review"
    assert review["decision"] == "hold_all_candidates"
    assert decisions["structured_hold_v1"] == "hold_candidate"
    assert decisions["diagnostic_done_v1"] == "diagnostic_complete"
    assert any("local score delta" in issue for issue in review["issues"])


def test_promotion_gate_consumes_regularized_blend_evidence(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps({"champion": {"task_id": "current_champion", "metric_name": "accuracy", "local_score": 0.80}}),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "regularized_blend_evidence_test",
                        "runner_kind": "regularized_blend",
                        "evidence_needed": [
                            "regularized_blend_report.json",
                            "oof_predictions.csv",
                            "seed_mean",
                            "seed_std",
                            "fold_std",
                            "train_valid_gap",
                            "max_model_correlation",
                        ],
                        "promotion_gate": {
                            "validator_must_pass": True,
                            "seed_mean": ">= 0.75",
                            "seed_std": "<= 0.10",
                            "fold_std": "<= 0.20",
                            "train_valid_gap": "<= 0.50",
                            "max_model_correlation": "<= 1.0",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_dir = competition_dir / "experiments" / "regularized_blend_evidence_test"
    exp_dir.mkdir(parents=True)
    evidence_report = {
        "experiment": "regularized_blend_evidence_test",
        "runner_kind": "regularized_blend",
        "status": "completed",
        "metric_name": "accuracy",
        "local_score": 0.812,
        "seed_mean": 0.812,
        "seed_std": 0.01,
        "fold_std": 0.04,
        "train_valid_gap": 0.03,
        "max_model_correlation": 0.95,
        "issues": [],
    }
    (exp_dir / "validation_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "regularized_blend_report.json").write_text(
        json.dumps({**evidence_report, "seed_scores": [{"seed": 42, "score": 0.812}]}),
        encoding="utf-8",
    )
    (exp_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (exp_dir / "submission.csv").write_text((source / "sample_submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (exp_dir / "oof_predictions.csv").write_text("PassengerId,Survived,oof_selected\n1,0,0.1\n", encoding="utf-8")
    (exp_dir / "risk_audit.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "risk_level": "low",
                "risk_points": 0,
                "fold_stability_score": 0.95,
                "issues": [],
                "recommendation": "Proceed.",
            }
        ),
        encoding="utf-8",
    )

    gate_result = PromotionGateEvaluator(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).evaluate()
    review = json.loads(gate_result.review_path.read_text(encoding="utf-8"))
    evaluation = review["evaluations"][0]

    assert evaluation["decision"] == "promote_candidate"
    assert evaluation["issues"] == []
    assert gate_result.promoted_submission_path and gate_result.promoted_submission_path.exists()


def test_promotion_gate_allows_stable_candidate_with_bounded_score_drop(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps({"champion": {"task_id": "possibly_overfit_champion", "metric_name": "accuracy", "local_score": 0.846}}),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "stable_regularized_blend",
                        "runner_kind": "regularized_blend",
                        "evidence_needed": ["regularized_blend_report.json", "oof_predictions.csv", "seed_std", "fold_std"],
                        "promotion_gate": {
                            "validator_must_pass": True,
                            "max_local_score_drop": 0.03,
                            "seed_std": "<= 0.010",
                            "fold_std": "<= 0.030",
                            "train_valid_gap": "<= 0.040",
                            "max_model_correlation": "<= 0.995",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_dir = competition_dir / "experiments" / "stable_regularized_blend"
    exp_dir.mkdir(parents=True)
    evidence_report = {
        "experiment": "stable_regularized_blend",
        "runner_kind": "regularized_blend",
        "status": "completed",
        "metric_name": "accuracy",
        "local_score": 0.822,
        "seed_std": 0.004,
        "fold_std": 0.017,
        "train_valid_gap": 0.026,
        "max_model_correlation": 0.991,
    }
    (exp_dir / "validation_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "regularized_blend_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (exp_dir / "submission.csv").write_text((source / "sample_submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (exp_dir / "oof_predictions.csv").write_text("PassengerId,Survived,oof_selected\n1,0,0.1\n", encoding="utf-8")

    gate_result = PromotionGateEvaluator(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).evaluate()
    review = json.loads(gate_result.review_path.read_text(encoding="utf-8"))

    assert gate_result.status == "pass"
    assert review["decision"] == "promote_candidate"
    assert review["promoted_candidate"]["task_id"] == "stable_regularized_blend"


def test_post_experiment_pipeline_runs_downstream_gates_for_promoted_candidate(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "competition_name": "titanic_copy",
                "decision": "champion_selected",
                "champion": {
                    "task_id": "current_champion",
                    "metric_name": "accuracy",
                    "local_score": 0.80,
                    "risk_level": "low",
                    "submission_path": str(competition_dir / "champion_submission.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "champion_submission.csv").write_text(
        (source / "sample_submission.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "post_pause_regularized_blend_v1",
                        "runner_kind": "regularized_blend",
                        "evidence_needed": [
                            "validation_report.json",
                            "validator_result.json",
                            "submission.csv",
                            "regularized_blend_report.json",
                            "oof_predictions.csv",
                            "seed_std",
                            "fold_std",
                            "train_valid_gap",
                            "max_model_correlation",
                        ],
                        "promotion_gate": {
                            "validator_must_pass": True,
                            "manual_submit_allowed": False,
                            "max_local_score_drop": 0.03,
                            "seed_std": "<= 0.010",
                            "fold_std": "<= 0.030",
                            "train_valid_gap": "<= 0.040",
                            "max_model_correlation": "<= 0.995",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_dir = competition_dir / "experiments" / "post_pause_regularized_blend_v1"
    exp_dir.mkdir(parents=True)
    evidence_report = {
        "experiment": "post_pause_regularized_blend_v1",
        "runner_kind": "regularized_blend",
        "status": "completed",
        "metric_name": "accuracy",
        "local_score": 0.823,
        "seed_mean": 0.822,
        "seed_std": 0.004,
        "fold_std": 0.02,
        "train_valid_gap": 0.025,
        "max_model_correlation": 0.98,
        "risk_level": "low",
        "issues": [],
    }
    (exp_dir / "validation_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "regularized_blend_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (exp_dir / "submission.csv").write_text((source / "sample_submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (exp_dir / "oof_predictions.csv").write_text("PassengerId,Survived,oof_selected\n1,0,0.1\n", encoding="utf-8")

    result = PostExperimentPipeline(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run(
        submission_target="recommended"
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    refreshed_plan = json.loads((competition_dir / "llm_experiment_plan.json").read_text(encoding="utf-8"))
    policy = json.loads((competition_dir / "submission_policy.json").read_text(encoding="utf-8"))
    gate = json.loads((competition_dir / "submission_gate.json").read_text(encoding="utf-8"))
    readiness = json.loads((competition_dir / "manual_submit_readiness.json").read_text(encoding="utf-8"))
    handoff = json.loads((competition_dir / "submit_decision_handoff.json").read_text(encoding="utf-8"))
    workflow = json.loads((competition_dir / "post_submit_workflow.json").read_text(encoding="utf-8"))
    feedback_template = json.loads((competition_dir / "leaderboard_feedback_input_template.json").read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert result.status == "pass"
    assert report["decision"] == "ready_for_human_submit_decision"
    refreshed_gate = refreshed_plan["recommended_experiments"][0]["promotion_gate"]
    assert "min_local_score_delta" not in refreshed_gate
    assert refreshed_gate["max_local_score_drop"] == 0.03
    assert report["downstream"]["submission_policy_status"] == "pass"
    assert report["downstream"]["post_submit_workflow_status"] == "ready_for_manual_submit"
    assert policy["policy"]["source"] == "promotion_gate"
    assert gate["status"] == "pass"
    assert readiness["manual_submission_ready"] is True
    assert handoff["status"] == "ready_for_human_submit_decision"
    assert workflow["candidate"]["task_id"] == "post_pause_regularized_blend_v1"
    assert feedback_template["candidate_task_id"] == "post_pause_regularized_blend_v1"
    assert (competition_dir / "recommended_submission.csv").exists()
    assert "Run post-experiment promotion and submit gates" in html
    assert "Prepare post-submit feedback workflow" in html


def test_post_experiment_pipeline_falls_back_to_champion_policy_when_promotion_blocks(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    champion_submission = competition_dir / "champion_submission.csv"
    champion_submission.write_text(
        (source / "sample_submission.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    champion = {
        "source_id": "experiment:stable_champion",
        "task_id": "stable_champion",
        "metric_name": "accuracy",
        "local_score": 0.84,
        "risk_level": "low",
        "submission_valid": True,
        "submission_path": str(champion_submission),
    }
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "competition_name": "titanic_copy",
                "status": "pass",
                "decision": "champion_selected",
                "champion": champion,
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "champion_comparison.json").write_text(
        json.dumps(
            {
                "competition_name": "titanic_copy",
                "status": "completed",
                "top_candidates": [champion],
                "feature_control_candidates": [],
                "selection_context": {},
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "weak_candidate",
                        "runner_kind": "lightgbm",
                        "promotion_gate": {
                            "min_local_score_delta": 0.002,
                            "validator_must_pass": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    weak_dir = competition_dir / "experiments" / "weak_candidate"
    weak_dir.mkdir(parents=True)
    (weak_dir / "validation_report.json").write_text(
        json.dumps({"status": "completed", "metric_name": "accuracy", "local_score": 0.80}),
        encoding="utf-8",
    )
    (weak_dir / "validator_result.json").write_text(
        json.dumps({"ok": True, "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    (weak_dir / "submission.csv").write_text(champion_submission.read_text(encoding="utf-8"), encoding="utf-8")

    result = PostExperimentPipeline(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).run(submission_target="recommended")
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    policy = json.loads((competition_dir / "submission_policy.json").read_text(encoding="utf-8"))
    workflow = json.loads((competition_dir / "post_submit_workflow.json").read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert report["promotion_gate_decision"] == "hold_all_candidates"
    assert report["downstream"]["submission_policy_status"] == "pass"
    assert policy["policy"]["source"] == "champion_comparison"
    assert policy["recommended_submission_candidate"]["task_id"] == "stable_champion"
    assert workflow["candidate"]["task_id"] == "stable_champion"
    assert (competition_dir / "recommended_submission.csv").exists()


def test_manual_submission_package_bundles_submission_and_feedback_template(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "competition_name": "titanic_copy",
                "decision": "champion_selected",
                "champion": {
                    "task_id": "current_champion",
                    "metric_name": "accuracy",
                    "local_score": 0.846,
                    "risk_level": "low",
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "recommended_experiments": [
                    {
                        "task_id": "post_pause_regularized_blend_v1",
                        "runner_kind": "regularized_blend",
                        "evidence_needed": ["regularized_blend_report.json", "oof_predictions.csv"],
                        "promotion_gate": {
                            "validator_must_pass": True,
                            "max_local_score_drop": 0.03,
                            "seed_std": "<= 0.010",
                            "fold_std": "<= 0.030",
                            "train_valid_gap": "<= 0.040",
                            "max_model_correlation": "<= 0.995",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_dir = competition_dir / "experiments" / "post_pause_regularized_blend_v1"
    exp_dir.mkdir(parents=True)
    evidence_report = {
        "experiment": "post_pause_regularized_blend_v1",
        "runner_kind": "regularized_blend",
        "status": "completed",
        "metric_name": "accuracy",
        "local_score": 0.823,
        "seed_std": 0.004,
        "fold_std": 0.02,
        "train_valid_gap": 0.025,
        "max_model_correlation": 0.98,
        "risk_level": "low",
    }
    (exp_dir / "validation_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "regularized_blend_report.json").write_text(json.dumps(evidence_report), encoding="utf-8")
    (exp_dir / "validator_result.json").write_text(json.dumps({"ok": True, "errors": [], "warnings": []}), encoding="utf-8")
    (exp_dir / "submission.csv").write_text((source / "sample_submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (exp_dir / "oof_predictions.csv").write_text("PassengerId,Survived,oof_selected\n1,0,0.1\n", encoding="utf-8")
    (exp_dir / "risk_audit.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "risk_level": "low",
                "risk_points": 0,
                "fold_stability_score": 0.95,
                "issues": [],
                "recommendation": "Proceed.",
            }
        ),
        encoding="utf-8",
    )

    pipeline = PostExperimentPipeline(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run(
        submission_target="recommended"
    )
    assert pipeline.status == "pass"
    result = ManualSubmissionPackage(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).build(
        submission_target="recommended"
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    template = json.loads((result.package_dir / "leaderboard_feedback_input_template.json").read_text(encoding="utf-8"))
    readme = result.checklist_path.read_text(encoding="utf-8")
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert result.status == "ready_for_manual_upload"
    assert manifest["candidate"]["task_id"] == "post_pause_regularized_blend_v1"
    assert manifest["packaged_submission_relative_path"] == "manual_submission_package/submission.csv"
    assert manifest["feedback_template_relative_path"] == "manual_submission_package/leaderboard_feedback_input_template.json"
    assert template["candidate_task_id"] == "post_pause_regularized_blend_v1"
    assert (result.package_dir / "submission.csv").exists()
    assert manifest["submission_file"]["exists"] is True
    assert len(manifest["submission_file"]["sha256"]) == 64
    assert manifest["submission_file"]["row_count"] == 418
    assert manifest["submission_file"]["columns"] == ["PassengerId", "Survived"]
    assert manifest["feedback_template_file"]["exists"] is True
    assert len(manifest["feedback_template_file"]["sha256"]) == 64
    assert manifest["candidate_risk"]["source"] == "risk_audit"
    assert manifest["candidate_risk"]["risk_level"] == "low"
    assert manifest["candidate_risk"]["issues"] == []
    assert "Confirm manual_submission_package/submission.csv SHA-256 before upload." in manifest["upload_file_checks"]
    assert manifest["feedback_loop_command"].endswith(
        "--feedback-template manual_submission_package/leaderboard_feedback_input_template.json"
    )
    assert manifest["verify_package_command"].endswith("--verify-manual-submission-package")
    assert "--fill-leaderboard-feedback-template" in manifest["feedback_fill_command"]
    assert "--run-filled-feedback-loop" in manifest["feedback_fill_command"]
    assert "OR_OMIT" not in manifest["feedback_fill_command"]
    assert "--leaderboard-rank" not in manifest["feedback_fill_command"]
    assert "--submission-id" not in manifest["feedback_fill_command"]
    assert "Manual Submission Package" in readme
    assert "Before Upload" in readme
    assert "--verify-manual-submission-package" in readme
    assert "`manual_submission_package/submission.csv`" in readme
    assert "`manual_submission_package/leaderboard_feedback_input_template.json`" in readme
    assert "Submission SHA-256" in readme
    assert "Submission rows: 418" in readme
    assert "Submission columns: PassengerId, Survived" in readme
    assert "Risk level: low" in readme
    assert "--fill-leaderboard-feedback-template" in readme
    assert "--run-filled-feedback-loop" in readme
    assert "Optional additions when Kaggle shows them" in readme
    assert "--leaderboard-rank <LEADERBOARD_RANK> --submission-id <SUBMISSION_ID>" in readme
    assert "--feedback-template manual_submission_package/leaderboard_feedback_input_template.json" in readme
    assert "Package manual Kaggle upload handoff" in html

    verification = ManualSubmissionPackageVerifier(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).verify()
    assert verification.status == "pass"

    roadmap_result = ExperimentRoadmapBuilder(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).build()
    roadmap = json.loads(roadmap_result.roadmap_path.read_text(encoding="utf-8"))
    roadmap_md = roadmap_result.markdown_path.read_text(encoding="utf-8")
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert roadmap_result.status == "ready"
    assert roadmap["decision"] == "await_human_leaderboard_feedback"
    assert roadmap["top_action"]["action_id"] == "manual_upload_and_feedback_capture"
    assert roadmap["top_action"]["status"] == "waiting_for_human"
    assert any(item["action_id"] == "cross_competition_tabular_smoke" for item in roadmap["items"])
    assert "Experiment Roadmap" in roadmap_md
    assert "实验路线图" in html
    assert "manual_upload_and_feedback_capture" in html


def test_experiment_roadmap_flags_stale_leaderboard_feedback_binding(tmp_path: Path):
    competition_dir = tmp_path / "demo_competition"
    competition_dir.mkdir()
    (competition_dir / "data_manifest.json").write_text(
        json.dumps({"competition_name": "demo_competition"}),
        encoding="utf-8",
    )
    package_dir = competition_dir / "manual_submission_package"
    package_dir.mkdir()
    current_package = {
        "competition_name": "demo_competition",
        "status": "ready_for_manual_upload",
        "submission_target": "recommended",
        "candidate": {"task_id": "same_task_id", "metric_name": "roc_auc", "local_score": 0.88},
        "submission_file": {
            "sha256": "current-sha",
            "row_count": 100,
            "columns": ["id", "target"],
        },
        "candidate_risk": {"risk_level": "low"},
        "feedback_fill_command": "python framework.py --competition demo_competition --fill-leaderboard-feedback-template --public-score <PUBLIC_SCORE> --run-filled-feedback-loop",
        "feedback_loop_command": "python framework.py --competition demo_competition --leaderboard-feedback-from-template",
    }
    (package_dir / "manifest.json").write_text(json.dumps(current_package), encoding="utf-8")
    (competition_dir / "manual_submission_package_verification.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "package_verified_for_upload",
                "candidate_task_id": "same_task_id",
                "actual_submission_file": {
                    "sha256": "current-sha",
                    "row_count": 100,
                    "columns": ["id", "target"],
                },
                "issues": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_feedback.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "submission_target": "recommended",
                "candidate_task_id": "same_task_id",
                "public_score": 0.81,
                "expected_submission_sha256": "old-sha",
                "expected_submission_rows": 100,
                "expected_submission_columns": ["id", "target"],
                "candidate_risk_level": "low",
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_feedback_loop.json").write_text(
        json.dumps({"status": "pass"}),
        encoding="utf-8",
    )

    roadmap_result = ExperimentRoadmapBuilder(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).build()
    roadmap = json.loads(roadmap_result.roadmap_path.read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert roadmap["leaderboard_feedback_freshness"]["status"] == "stale"
    assert roadmap["leaderboard_feedback_freshness"]["is_current"] is False
    assert any("sha256" in issue or "sha mismatch" in issue for issue in roadmap["leaderboard_feedback_freshness"]["issues"])
    assert roadmap["top_action"]["action_id"] == "manual_upload_and_feedback_capture"
    assert "--fill-leaderboard-feedback-template" in roadmap["top_action"]["next_command"]
    assert "--run-filled-feedback-loop" in roadmap["top_action"]["next_command"]
    assert "Existing feedback is not current" in roadmap["top_action"]["rationale"]
    assert roadmap["manual_submission_package_verification"]["status"] == "pass"
    assert "手动提交包校验" in html
    assert "package_verified_for_upload" in html
    assert "Leaderboard 反馈新鲜度" in html
    assert "--fill-leaderboard-feedback-template" in html
    assert "current-sha" in html
    assert "old-sha" in html

    brain_result = RemoteBrainReviewer(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
        use_llm=False,
    ).review()
    brain_plan = json.loads(brain_result.json_path.read_text(encoding="utf-8"))

    assert brain_plan["leaderboard_feedback_freshness"]["status"] == "stale"
    assert any("not bound to the current packaged submission" in risk for risk in brain_plan["risks"])


def test_experiment_roadmap_requires_package_verification_before_manual_upload(tmp_path: Path):
    competition_dir = tmp_path / "demo_competition"
    competition_dir.mkdir()
    (competition_dir / "data_manifest.json").write_text(
        json.dumps({"competition_name": "demo_competition"}),
        encoding="utf-8",
    )
    package_dir = competition_dir / "manual_submission_package"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "competition_name": "demo_competition",
                "status": "ready_for_manual_upload",
                "submission_target": "recommended",
                "candidate": {"task_id": "same_task_id", "metric_name": "roc_auc", "local_score": 0.88},
                "submission_file": {
                    "sha256": "current-sha",
                    "row_count": 100,
                    "columns": ["id", "target"],
                },
                "feedback_fill_command": "python framework.py --competition demo_competition --fill-leaderboard-feedback-template --public-score <PUBLIC_SCORE> --run-filled-feedback-loop",
            }
        ),
        encoding="utf-8",
    )

    roadmap_result = ExperimentRoadmapBuilder(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).build()
    roadmap = json.loads(roadmap_result.roadmap_path.read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert roadmap["top_action"]["action_id"] == "verify_manual_submission_package"
    assert roadmap["top_action"]["status"] == "ready"
    assert "--verify-manual-submission-package" in roadmap["top_action"]["next_command"]
    assert "--verify-manual-submission-package" in html


def test_experiment_roadmap_resolves_stale_queue_paths_by_task_id(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    task_id = "completed_remote_path_task"
    (competition_dir / "experiment_queue.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "next_runnable": {
                    "task_id": task_id,
                    "status": "pending",
                    "action_type": "coding_experiment",
                    "experiment_dir": "/home/old/workspace/experiments/completed_remote_path_task",
                    "next_command": "python framework.py --competition {competition} --run-enhancement",
                },
            }
        ),
        encoding="utf-8",
    )
    exp_dir = competition_dir / "experiments" / task_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "validation_report.json").write_text(
        json.dumps({"status": "completed", "local_score": 0.8}),
        encoding="utf-8",
    )

    result = ExperimentRoadmapBuilder(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).build()
    roadmap = json.loads(result.roadmap_path.read_text(encoding="utf-8"))

    assert not any(item["action_id"] == f"execute_queue_{task_id}" for item in roadmap["items"])
    assert any(item["action_id"] == "refresh_remote_brain_queue" for item in roadmap["items"])


def test_manual_submission_package_handles_missing_submission_without_crashing(tmp_path: Path):
    competition_dir = tmp_path / "blocked_competition"
    competition_dir.mkdir()
    (competition_dir / "post_experiment_pipeline.json").write_text(
        json.dumps(
            {
                "competition_name": "blocked_competition",
                "status": "needs_review",
                "promoted_candidate": {"task_id": "missing_submission_candidate"},
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "submit_decision_handoff.json").write_text(
        json.dumps({"status": "needs_review", "submission_target": "recommended"}),
        encoding="utf-8",
    )
    (competition_dir / "post_submit_workflow.json").write_text(
        json.dumps({"status": "needs_review", "submission_target": "recommended"}),
        encoding="utf-8",
    )

    result = ManualSubmissionPackage(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).build(submission_target="recommended")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "needs_review"
    assert manifest["submission_file"]["exists"] is False
    assert "Submission file is missing." in manifest["issues"]
    assert not (result.package_dir / "submission.csv").exists()


def test_enhancement_runner_executes_specialized_audit_templates(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    CompetitionIngestor(competition_dir).build_manifest().write_json(
        competition_dir / "data_manifest.json"
    )
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "next_action": "recommend_experiments",
                "recommended_experiments": [
                    {
                        "task_id": "distribution_shift_audit_v1",
                        "title": "Distribution shift audit",
                        "coding_agent_task": "Audit train/test drift.",
                    },
                    {
                        "task_id": "overfitting_detection_experiment_v1",
                        "title": "Overfitting detection experiment",
                        "coding_agent_task": "Detect overfitting.",
                    },
                    {
                        "task_id": "champion_blend_with_regularization_v1",
                        "title": "Champion blend with regularization",
                        "coding_agent_task": "Build a regularized blend.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    ExperimentQueueBuilder(competition_dir, memory=memory).build()
    expected_artifacts = {
        "distribution_shift_audit_v1": "distribution_shift_audit.json",
        "overfitting_detection_experiment_v1": "overfitting_audit.json",
        "champion_blend_with_regularization_v1": "regularized_blend_report.json",
    }
    for task_id, artifact_name in expected_artifacts.items():
        result = EnhancementRunner(competition_dir, memory=memory).run_next_recommendation()
        report = json.loads(result.validation_report.read_text(encoding="utf-8"))
        assert result.task_id == task_id
        assert result.status in {"validated", "skipped"}
        if report["status"] == "completed":
            assert (result.experiment_dir / artifact_name).exists()
            assert report["runner_kind"] in {
                "distribution_shift_audit",
                "overfitting_audit",
                "regularized_blend",
            }
            if report["runner_kind"] == "regularized_blend":
                blend_report = json.loads((result.experiment_dir / "regularized_blend_report.json").read_text(encoding="utf-8"))
                assert "seed_mean" in report
                assert "seed_std" in report
                assert "fold_std" in report
                assert "train_valid_gap" in report
                assert "max_model_correlation" in report
                assert (result.experiment_dir / "oof_predictions.csv").exists()
                assert blend_report["seed_scores"]
            assert result.validator_result.ok is True


def test_tabular_search_runner_writes_search_artifacts(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    feature_dir = competition_dir / "experiments" / "tabular_feature_prune_v1"
    feature_dir.mkdir(parents=True)
    (feature_dir / "feature_report.json").write_text(
        '{"kept_features": ["Sex", "Title", "FamilySize", "Age", "Pclass"]}',
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    result = TabularSearchRunner(competition_dir, memory=memory).run(
        "tabular_search_test",
        cv_seeds=[11, 42],
        feature_set="pruned",
    )

    assert result.status in {"validated", "skipped"}
    assert (competition_dir / "experiments" / "tabular_search_test" / "run.py").exists()
    assert (competition_dir / "experiments" / "tabular_search_test" / "validation_report.json").exists()
    assert (competition_dir / "experiments" / "tabular_search_test" / "model_report.json").exists()
    assert (competition_dir / "experiments" / "tabular_search_test" / "ensemble_report.json").exists()
    script = (competition_dir / "experiments" / "tabular_search_test" / "run.py").read_text(encoding="utf-8")
    assert "cv_seeds = [11, 42]" in script
    assert 'requested_feature_set = "pruned"' in script
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Run tabular multi-model search and blend" in html
    assert memory.query(competition_name="titanic_copy")


def test_tabular_search_runner_consumes_leakage_safe_feature_set(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (competition_dir / "tabular_feature_leakage_audit.json").write_text(
        json.dumps({"recommended_drop_features": ["Title"]}),
        encoding="utf-8",
    )

    result = TabularSearchRunner(competition_dir).run(
        "leakage_safe_search_test",
        cv_seeds=[42],
        feature_set="leakage_safe",
    )

    assert result.status in {"validated", "skipped"}
    script = (competition_dir / "experiments" / "leakage_safe_search_test" / "run.py").read_text(encoding="utf-8")
    assert 'requested_feature_set = "leakage_safe"' in script
    assert 'leakage_audit_path = root / "tabular_feature_leakage_audit.json"' in script
    report = json.loads((competition_dir / "experiments" / "leakage_safe_search_test" / "validation_report.json").read_text(encoding="utf-8"))
    assert report["feature_set"] == "leakage_safe"
    if report["status"] == "completed":
        assert report["requested_drop_features"] == ["Title"]
        assert report["feature_set_source"].endswith("tabular_feature_leakage_audit.json")


def test_tabular_risk_auditor_writes_audit_and_ledger(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    search = TabularSearchRunner(competition_dir, memory=memory).run("tabular_search_test")
    result = TabularRiskAuditor(competition_dir, memory=memory).audit(search.task_id)

    assert result.status in {"pass", "needs_review"}
    assert (competition_dir / "experiments" / "tabular_search_test" / "risk_audit.json").exists()
    audit = (competition_dir / "experiments" / "tabular_search_test" / "risk_audit.json").read_text(encoding="utf-8")
    assert "risk_level" in audit
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Audit tabular CV and leaderboard risk" in html
    assert memory.query(competition_name="titanic_copy")


def test_tabular_risk_auditor_uses_single_model_cv_scores(tmp_path: Path):
    competition_dir = tmp_path / "bank_churn_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "bank_churn"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    task_id = "single_model_lightgbm"
    exp_dir = competition_dir / "experiments" / task_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "experiment": task_id,
                "runner_kind": "lightgbm",
                "status": "completed",
                "metric_name": "roc_auc",
                "local_score": 0.889,
                "cv_scores": [0.889, 0.890, 0.888, 0.891, 0.889],
            }
        ),
        encoding="utf-8",
    )

    result = TabularRiskAuditor(
        competition_dir,
        memory=CompetitionMemory(tmp_path / "memory"),
    ).audit(task_id)
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert audit["risk_level"] == "low"
    assert audit["max_fold_std"] is not None
    assert not any("OOF predictions are missing" in issue for issue in audit["issues"])


def test_tabular_feature_pruner_writes_feature_report_and_ledger(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    result = TabularFeaturePruner(competition_dir, memory=memory).run("feature_prune_test")

    assert result.status in {"validated", "skipped"}
    assert (competition_dir / "experiments" / "feature_prune_test" / "run.py").exists()
    assert (competition_dir / "experiments" / "feature_prune_test" / "validation_report.json").exists()
    assert (competition_dir / "experiments" / "feature_prune_test" / "feature_report.json").exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Run tabular feature importance and pruning check" in html
    assert memory.query(competition_name="titanic_copy")


def test_tabular_feature_leakage_auditor_flags_transform_scope_and_drift(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    result = TabularFeatureLeakageAuditor(competition_dir, memory=memory).audit()

    assert result.status in {"pass", "needs_review"}
    assert result.audit_path.exists()
    report = json.loads((competition_dir / "tabular_feature_leakage_audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["engineered_train_test_drift"]["status"] == "completed"
    assert "Title" in report["leakage_checks"]["derived_features_present"]
    assert any(item["feature"] == "Title" for item in report["leakage_checks"]["transform_scope_risks"])
    assert "Title" in report["recommended_drop_features"]
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Audit tabular feature leakage and drift" in html


def test_champion_selector_copies_best_valid_submission(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    result = ExperimentChampionSelector(competition_dir, memory=memory).select()

    assert result.status == "pass"
    assert (competition_dir / "champion_selection.json").exists()
    assert (competition_dir / "champion_comparison.json").exists()
    assert (competition_dir / "champion_submission.csv").exists()
    selection = (competition_dir / "champion_selection.json").read_text(encoding="utf-8")
    assert "champion_selected" in selection
    comparison = json.loads((competition_dir / "champion_comparison.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "completed"
    assert comparison["top_candidates"]
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Select current champion submission" in html
    assert memory.query(competition_name="titanic_copy")


def test_champion_selector_prefers_stability_first_when_public_gap_is_high(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    def write_candidate(task_id: str, score: float):
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
            json.dumps({"risk_level": "low", "issues": []}),
            encoding="utf-8",
        )

    write_candidate("tabular_model_search_v1", 0.846)
    write_candidate("stability_first_search_v1", 0.837)
    write_candidate("leakage_safe_search_v1", 0.834)
    leakage_validation = competition_dir / "experiments" / "leakage_safe_search_v1" / "validation_report.json"
    leakage_payload = json.loads(leakage_validation.read_text(encoding="utf-8"))
    leakage_payload["feature_set"] = "leakage_safe"
    leakage_payload["requested_drop_features"] = ["Title"]
    leakage_validation.write_text(json.dumps(leakage_payload), encoding="utf-8")
    write_candidate("unknown_risk_candidate", 0.839)
    (competition_dir / "experiments" / "unknown_risk_candidate" / "risk_audit.json").unlink()
    (competition_dir / "leaderboard_gap_audit.json").write_text(
        json.dumps(
            {
                "risk_level": "high",
                "champion": {
                    "task_id": "tabular_model_search_v1",
                    "source_id": "experiment:tabular_model_search_v1",
                },
                "score_gap": {"materially_worse": True, "gap": -0.03},
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
                "local_score": 0.837,
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "tabular_feature_leakage_audit.json").write_text(
        json.dumps(
            {
                "risk_level": "medium",
                "recommended_drop_features": ["Title"],
                "issues": ["Raw train/test drift is high."],
                "warnings": ["Rare-title bucketing risk."],
            }
        ),
        encoding="utf-8",
    )

    result = ExperimentChampionSelector(competition_dir).select()
    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    comparison = json.loads((competition_dir / "champion_comparison.json").read_text(encoding="utf-8"))

    assert selection["champion"]["task_id"] == "stability_first_search_v1"
    assert "stability_first_candidate_bonus" in selection["champion"]["selection_context_notes"]
    assert comparison["selection_context"]["feature_leakage"]["recommended_drop_features"] == ["Title"]
    leakage_rows = [
        item for item in comparison["feature_control_candidates"]
        if item["task_id"] == "leakage_safe_search_v1"
    ]
    assert leakage_rows
    assert leakage_rows[0]["feature_control"]["uses_leakage_recommended_drops"] is True
    original = next(item for item in selection["candidates"] if item["task_id"] == "tabular_model_search_v1")
    assert original["contextual_penalty"] == 0.04
    unknown = next(item for item in selection["candidates"] if item["task_id"] == "unknown_risk_candidate")
    assert "penalized_unknown_risk_under_public_gap" in unknown["selection_context_notes"]


def test_submission_policy_recommends_safer_candidate_under_leakage_context(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    high_dir = competition_dir / "experiments" / "high_cv"
    safe_dir = competition_dir / "experiments" / "safe_cv"
    for exp_dir, score, feature_set, drops in [
        (high_dir, 0.846, None, []),
        (safe_dir, 0.837, "leakage_safe", ["Title"]),
    ]:
        exp_dir.mkdir(parents=True)
        (exp_dir / "submission.csv").write_text(
            (source / "sample_submission.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (exp_dir / "validation_report.json").write_text(
            json.dumps(
                {
                    "experiment": exp_dir.name,
                    "status": "completed",
                    "metric_name": "accuracy",
                    "local_score": score,
                    "feature_set": feature_set,
                    "requested_drop_features": drops,
                }
            ),
            encoding="utf-8",
        )
        (exp_dir / "validator_result.json").write_text(
            json.dumps({"ok": True, "errors": [], "warnings": []}),
            encoding="utf-8",
        )
        (exp_dir / "risk_audit.json").write_text(
            json.dumps({"risk_level": "low", "issues": []}),
            encoding="utf-8",
        )
    (competition_dir / "tabular_feature_leakage_audit.json").write_text(
        json.dumps({"risk_level": "medium", "recommended_drop_features": ["Title"]}),
        encoding="utf-8",
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    result = SubmissionPolicy(competition_dir, memory=memory).run()
    policy = json.loads((competition_dir / "submission_policy.json").read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert policy["cv_champion"]["task_id"] == "high_cv"
    assert policy["recommended_submission_candidate"]["task_id"] == "safe_cv"
    assert policy["policy"]["changed_from_cv_champion"] is True
    assert (competition_dir / "recommended_submission.csv").exists()

    gate_result = SubmissionGate(competition_dir, memory=memory).run(
        dry_run=True,
        submission_target="recommended",
    )
    plan_result = KaggleSubmitAdapter(competition_dir, memory=memory).plan(
        dry_run=True,
        submission_target="recommended",
    )
    gate = json.loads(gate_result.gate_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_result.plan_path.read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["submission_target"] == "recommended"
    assert gate["candidate"]["task_id"] == "safe_cv"
    assert plan["status"] == "pass"
    assert plan["submission_target"] == "recommended"
    assert plan["submission_path"].endswith("recommended_submission.csv")


def test_submission_policy_prefers_promoted_submission(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    promoted_submission = competition_dir / "promoted_submission.csv"
    promoted_submission.write_text(
        (source / "sample_submission.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "competition_name": "titanic_copy",
                "decision": "champion_selected",
                "champion": {
                    "task_id": "old_champion",
                    "source_id": "experiment:old_champion",
                    "metric_name": "accuracy",
                    "local_score": 0.846,
                    "submission_path": str(competition_dir / "missing_old.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "promotion_gate_review.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "promote_candidate",
                "promoted_submission_path": str(promoted_submission),
                "promoted_candidate": {
                    "task_id": "regularized_blend_evidence_v2",
                    "source_id": "promotion:regularized_blend_evidence_v2",
                    "runner_kind": "regularized_blend",
                    "metric_name": "accuracy",
                    "local_score": 0.8215,
                    "submission_path": str(promoted_submission),
                },
            }
        ),
        encoding="utf-8",
    )

    result = SubmissionPolicy(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run()
    policy = json.loads(result.policy_path.read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert policy["policy"]["source"] == "promotion_gate"
    assert policy["promotion_gate"]["used_promoted_candidate"] is True
    assert policy["recommended_submission_candidate"]["task_id"] == "regularized_blend_evidence_v2"
    assert result.recommended_submission_path == competition_dir / "recommended_submission.csv"
    assert (competition_dir / "recommended_submission.csv").read_text(encoding="utf-8") == promoted_submission.read_text(encoding="utf-8")


def test_submission_policy_does_not_use_failed_promotion_gate_without_comparison(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    (competition_dir / "promotion_gate_review.json").write_text(
        json.dumps({"status": "needs_review", "decision": "hold_all_candidates"}),
        encoding="utf-8",
    )

    result = SubmissionPolicy(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run()
    policy = json.loads(result.policy_path.read_text(encoding="utf-8"))

    assert result.status == "needs_review"
    assert policy["decision"] == "policy_blocked"
    assert "champion_comparison.json is missing or incomplete; run --select-champion first." in policy["issues"]
    assert "No eligible top candidates are available." in policy["issues"]


def test_recommended_leaderboard_feedback_uses_policy_candidate(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    high_dir = competition_dir / "experiments" / "high_cv"
    safe_dir = competition_dir / "experiments" / "safe_cv"
    for exp_dir, score, feature_set, drops in [
        (high_dir, 0.846, None, []),
        (safe_dir, 0.837, "leakage_safe", ["Title"]),
    ]:
        exp_dir.mkdir(parents=True)
        (exp_dir / "submission.csv").write_text(
            (source / "sample_submission.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (exp_dir / "validation_report.json").write_text(
            json.dumps(
                {
                    "experiment": exp_dir.name,
                    "status": "completed",
                    "metric_name": "accuracy",
                    "local_score": score,
                    "feature_set": feature_set,
                    "requested_drop_features": drops,
                }
            ),
            encoding="utf-8",
        )
        (exp_dir / "validator_result.json").write_text(
            json.dumps({"ok": True, "errors": [], "warnings": []}),
            encoding="utf-8",
        )
        (exp_dir / "risk_audit.json").write_text(
            json.dumps({"risk_level": "low", "issues": []}),
            encoding="utf-8",
        )
    (competition_dir / "tabular_feature_leakage_audit.json").write_text(
        json.dumps({"risk_level": "medium", "recommended_drop_features": ["Title"]}),
        encoding="utf-8",
    )

    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionPolicy(competition_dir, memory=memory).run()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True, submission_target="recommended")
    result = LeaderboardFeedbackRecorder(competition_dir, memory=memory).record(
        public_score=0.81234,
        leaderboard_rank=1234,
        submission_id="recommended-demo",
        submission_target="recommended",
    )
    gap_result = LeaderboardGapAuditor(competition_dir, memory=memory).audit()

    assert result.status == "pass"
    feedback = json.loads((competition_dir / "leaderboard_feedback.json").read_text(encoding="utf-8"))
    gap = json.loads(gap_result.audit_path.read_text(encoding="utf-8"))
    assert feedback["submission_target"] == "recommended"
    assert feedback["candidate_task_id"] == "safe_cv"
    assert feedback["champion_task_id"] == "high_cv"
    assert feedback["local_score"] == 0.837
    assert feedback["submission_path"].endswith("recommended_submission.csv")
    assert gap["leaderboard_feedback"]["submission_target"] == "recommended"
    assert gap["candidate"]["task_id"] == "safe_cv"
    assert gap["champion"]["task_id"] == "high_cv"
    assert gap["score_gap"]["local_score"] == 0.837


def test_submission_gate_passes_selected_champion_dry_run(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    result = SubmissionGate(competition_dir, memory=memory).run(dry_run=True)

    assert result.status == "pass"
    assert (competition_dir / "submission_gate.json").exists()
    gate = (competition_dir / "submission_gate.json").read_text(encoding="utf-8")
    assert "ready_for_manual_or_api_submission" in gate
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Dry-run final submission gate" in html


def test_post_reselection_gate_refreshes_gate_and_submit_plan(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    result = PostReselectionGate(competition_dir, memory=memory).run()

    assert result.status == "pass"
    report = json.loads((competition_dir / "post_reselection_gate.json").read_text(encoding="utf-8"))
    gate = json.loads((competition_dir / "submission_gate.json").read_text(encoding="utf-8"))
    plan = json.loads((competition_dir / "kaggle_submit_plan.json").read_text(encoding="utf-8"))
    assert report["submission_gate_status"] == "pass"
    assert report["kaggle_submit_plan_status"] == "pass"
    assert gate["champion"]["task_id"] == report["champion"]["task_id"]
    assert plan["submission_gate_status"] == "pass"
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Refresh post-reselection submission gate" in html


def test_manual_submit_readiness_separates_manual_and_api_readiness(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    result = ManualSubmitReadinessChecker(competition_dir, memory=memory).run()

    assert result.status == "manual_submit_ready"
    report_text = (competition_dir / "manual_submit_readiness.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["manual_submission_ready"] is True
    assert report["confirmed_submit_ready"] is False
    assert report["credentials_available"] is False
    assert "KAGGLE_KEY" not in report_text
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Review manual and API submit readiness" in html


def test_manual_submit_readiness_supports_recommended_target(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionPolicy(competition_dir, memory=memory).run()
    result = ManualSubmitReadinessChecker(competition_dir, memory=memory).run(
        submission_target="recommended"
    )

    assert result.status == "manual_submit_ready"
    report = json.loads((competition_dir / "manual_submit_readiness.json").read_text(encoding="utf-8"))
    gate = json.loads((competition_dir / "submission_gate.json").read_text(encoding="utf-8"))
    plan = json.loads((competition_dir / "kaggle_submit_plan.json").read_text(encoding="utf-8"))
    assert report["submission_target"] == "recommended"
    assert report["submission_path"].endswith("recommended_submission.csv")
    assert report["candidate"]["task_id"] == gate["candidate"]["task_id"]
    assert gate["submission_target"] == "recommended"
    assert plan["submission_target"] == "recommended"


def test_post_submit_workflow_builds_feedback_handoff(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionPolicy(competition_dir, memory=memory).run()
    result = PostSubmitWorkflow(competition_dir, memory=memory).run(
        submission_target="recommended"
    )

    assert result.status == "ready_for_manual_submit"
    report_text = (competition_dir / "post_submit_workflow.json").read_text(encoding="utf-8")
    checklist = result.checklist_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["submission_target"] == "recommended"
    assert report["candidate"]["task_id"]
    assert report["submission_path"].endswith("recommended_submission.csv")
    assert report["submission_file"]["sha256"]
    assert report["submission_file"]["row_count"] == 418
    assert report["candidate_risk"]["risk_level"]
    assert "--leaderboard-feedback-from-template" in report["feedback_loop_command_template"]
    assert "KAGGLE_KEY" not in report_text
    assert "Post-Submit Workflow" in checklist
    assert "recommended_submission.csv" in checklist
    feedback_template = json.loads(result.feedback_input_template_path.read_text(encoding="utf-8"))
    assert feedback_template["submission_target"] == "recommended"
    assert feedback_template["candidate_task_id"] == report["candidate"]["task_id"]
    assert feedback_template["expected_submission_sha256"] == report["submission_file"]["sha256"]
    assert feedback_template["expected_submission_rows"] == 418
    assert feedback_template["expected_submission_columns"] == ["PassengerId", "Survived"]
    assert feedback_template["candidate_risk_level"] == report["candidate_risk"]["risk_level"]
    assert feedback_template["public_score"] == "<PUBLIC_SCORE>"
    assert feedback_template["leaderboard_rank"] is None
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Prepare post-submit feedback workflow" in html
    assert "Current Submit Handoff" in html
    assert "人工上传到反馈闭环" in html
    assert report["candidate"]["task_id"] in html
    assert "--leaderboard-feedback-from-template" in html
    assert "反馈输入模板" in html


def test_submit_decision_handoff_ready_for_promoted_recommended_submission(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    champion = json.loads((competition_dir / "champion_selection.json").read_text(encoding="utf-8"))["champion"]
    promoted_path = competition_dir / "promoted_submission.csv"
    promoted_path.write_text(Path(champion["submission_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    promoted = dict(champion)
    promoted.update(
        {
            "task_id": "regularized_blend_evidence_v2",
            "submission_path": str(promoted_path),
            "seed_mean": 0.82,
            "seed_std": 0.001,
            "fold_std": 0.017,
            "train_valid_gap": 0.026,
            "max_model_correlation": 0.99,
        }
    )
    (competition_dir / "promotion_gate_review.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "promote_candidate",
                "promoted_candidate": promoted,
                "promoted_submission_path": str(promoted_path),
                "warnings": [],
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    SubmissionPolicy(competition_dir, memory=memory).run()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True, submission_target="recommended")
    ManualSubmitReadinessChecker(competition_dir, memory=memory).run(
        submission_target="recommended"
    )
    result = SubmitDecisionHandoff(competition_dir, memory=memory).run(
        submission_target="recommended"
    )

    assert result.status == "ready_for_human_submit_decision"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    markdown = result.markdown_path.read_text(encoding="utf-8")
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert report["decision"] == "await_human_submit_decision"
    assert report["submission_target"] == "recommended"
    assert report["submission_policy"]["source"] == "promotion_gate"
    assert report["promotion_gate"]["decision"] == "promote_candidate"
    assert report["submission_path"].endswith("recommended_submission.csv")
    assert report["evidence_summary"]["seed_std"] == 0.001
    assert "--post-submit-workflow --submission-target recommended" in report["post_submit_workflow_command"]
    assert "This handoff does not submit automatically." in markdown
    assert "Submit Decision Handoff" in html
    assert "人工榜单提交决策" in html
    assert "regularized_blend_evidence_v2" in html


def test_submit_decision_handoff_blocks_unready_submission(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    (competition_dir / "manual_submit_readiness.json").write_text(
        json.dumps(
            {
                "competition_name": "titanic_copy",
                "status": "needs_review",
                "submission_target": "recommended",
                "manual_submission_ready": False,
                "candidate": {},
                "submission_path": str(competition_dir / "recommended_submission.csv"),
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    result = SubmitDecisionHandoff(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).run(
        submission_target="recommended"
    )

    assert result.status == "needs_review"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["decision"] == "fix_submit_handoff"
    assert "manual_submit_readiness.json is not ready for manual submission." in report["issues"]
    assert "Submission file is missing." in report["issues"]


def test_leaderboard_feedback_rejects_placeholder_submission_id(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    result = LeaderboardFeedbackRecorder(competition_dir, memory=memory).record(
        public_score=0.81234,
        submission_id="<SUBMISSION_ID>",
    )

    assert result.status == "needs_review"
    payload = json.loads((competition_dir / "leaderboard_feedback.json").read_text(encoding="utf-8"))
    assert "submission_id still contains an unreplaced placeholder." in payload["issues"]


def test_leaderboard_feedback_input_runner_validates_and_runs_loop(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionPolicy(competition_dir, memory=memory).run()
    workflow = PostSubmitWorkflow(competition_dir, memory=memory).run(
        submission_target="recommended"
    )
    template_path = workflow.feedback_input_template_path
    template = json.loads(template_path.read_text(encoding="utf-8"))
    blocked = LeaderboardFeedbackInputRunner(competition_dir, memory=memory).run(
        input_path=template_path,
        brain_review=False,
    )
    blocked_report = json.loads(blocked.report_path.read_text(encoding="utf-8"))
    assert blocked.status == "needs_review"
    assert any("public_score still contains" in issue for issue in blocked_report["issues"])

    tampered = dict(template)
    tampered["public_score"] = 0.81234
    tampered["submission_id"] = "demo-submission"
    tampered["expected_submission_sha256"] = "bad-sha"
    template_path.write_text(json.dumps(tampered), encoding="utf-8")
    hash_blocked = LeaderboardFeedbackInputRunner(competition_dir, memory=memory).run(
        input_path=template_path,
        brain_review=False,
    )
    hash_blocked_report = json.loads(hash_blocked.report_path.read_text(encoding="utf-8"))
    assert hash_blocked.status == "needs_review"
    assert "expected_submission_sha256 does not match post_submit_workflow template." in hash_blocked_report["issues"]

    template["public_score"] = 0.81234
    template["submission_id"] = "demo-submission"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    result = LeaderboardFeedbackInputRunner(competition_dir, memory=memory).run(
        input_path=template_path,
        brain_review=False,
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    loop = json.loads((competition_dir / "leaderboard_feedback_loop.json").read_text(encoding="utf-8"))
    feedback = json.loads((competition_dir / "leaderboard_feedback.json").read_text(encoding="utf-8"))

    assert result.status in {"pass", "needs_review"}
    assert result.feedback_loop_report_path is not None
    assert result.experiment_roadmap_path is not None
    assert report["experiment_roadmap_path"] == str(result.experiment_roadmap_path)
    assert report["submission_target"] == "recommended"
    assert report["candidate_task_id"] == template["candidate_task_id"]
    assert report["public_score"] == 0.81234
    assert feedback["expected_submission_sha256"] == template["expected_submission_sha256"]
    assert feedback["expected_submission_rows"] == template["expected_submission_rows"]
    assert feedback["expected_submission_columns"] == template["expected_submission_columns"]
    assert feedback["candidate_risk_level"] == template["candidate_risk_level"]
    assert feedback["submission_binding"]["expected_submission_sha256"] == template["expected_submission_sha256"]
    assert loop["submission_target"] == "recommended"
    assert loop["leaderboard_feedback_status"] == "pass"
    assert loop["expected_submission_sha256"] == template["expected_submission_sha256"]
    assert loop["candidate_risk_level"] == template["candidate_risk_level"]
    assert loop["experiment_roadmap_path"] == str(result.experiment_roadmap_path)
    assert result.experiment_roadmap_path.exists()


def _build_titanic_recommended_manual_package(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionPolicy(competition_dir, memory=memory).run()
    workflow = PostSubmitWorkflow(competition_dir, memory=memory).run(
        submission_target="recommended"
    )
    package = ManualSubmissionPackage(competition_dir, memory=memory).build(
        submission_target="recommended"
    )
    return competition_dir, memory, package, workflow


def test_leaderboard_feedback_template_filler_updates_template_and_can_run_loop(tmp_path: Path, monkeypatch):
    competition_dir, memory, package, workflow = _build_titanic_recommended_manual_package(tmp_path, monkeypatch)
    template_path = package.package_dir / "leaderboard_feedback_input_template.json"
    template_before = json.loads(template_path.read_text(encoding="utf-8"))

    fill_result = LeaderboardFeedbackTemplateFiller(competition_dir, memory=memory).fill(
        template_path=Path("manual_submission_package/leaderboard_feedback_input_template.json"),
        public_score=0.81234,
        leaderboard_rank=1234,
        submission_id="demo-submission",
        source="manual",
        notes="uploaded via browser",
        run_feedback_loop=True,
        brain_use_llm=False,
    )
    filled = json.loads(template_path.read_text(encoding="utf-8"))
    report = json.loads(fill_result.report_path.read_text(encoding="utf-8"))
    feedback = json.loads((competition_dir / "leaderboard_feedback.json").read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert workflow.status == "ready_for_manual_submit"
    assert fill_result.status in {"pass", "needs_review"}
    assert fill_result.feedback_loop_report_path is not None
    assert fill_result.experiment_roadmap_path is not None
    assert report["experiment_roadmap_path"] == str(fill_result.experiment_roadmap_path)
    assert filled["public_score"] == 0.81234
    assert filled["leaderboard_rank"] == 1234
    assert filled["submission_id"] == "demo-submission"
    assert filled["notes"] == "uploaded via browser"
    assert filled["expected_submission_sha256"] == template_before["expected_submission_sha256"]
    assert report["expected_submission_sha256"] == template_before["expected_submission_sha256"]
    assert feedback["expected_submission_sha256"] == template_before["expected_submission_sha256"]
    assert feedback["candidate_task_id"] == template_before["candidate_task_id"]
    assert "榜单分数回填" in html
    assert "0.81234" in html
    assert "demo-submission" in html
    assert template_before["expected_submission_sha256"] in html


def test_leaderboard_feedback_template_filler_rejects_mutated_packaged_submission(tmp_path: Path, monkeypatch):
    competition_dir, memory, package, _workflow = _build_titanic_recommended_manual_package(tmp_path, monkeypatch)
    template_path = package.package_dir / "leaderboard_feedback_input_template.json"
    template_before = json.loads(template_path.read_text(encoding="utf-8"))
    (package.package_dir / "submission.csv").write_text(
        "PassengerId,Survived\n892,1\n893,1\n",
        encoding="utf-8",
    )

    fill_result = LeaderboardFeedbackTemplateFiller(competition_dir, memory=memory).fill(
        template_path=Path("manual_submission_package/leaderboard_feedback_input_template.json"),
        public_score=0.81234,
        run_feedback_loop=True,
        brain_use_llm=False,
    )
    filled = json.loads(template_path.read_text(encoding="utf-8"))
    report = json.loads(fill_result.report_path.read_text(encoding="utf-8"))

    assert fill_result.status == "needs_review"
    assert fill_result.feedback_loop_report_path is None
    assert filled["public_score"] == template_before["public_score"]
    assert any("submission SHA-256 mismatch" in issue for issue in report["issues"])
    assert any("submission row count mismatch" in issue for issue in report["issues"])
    assert report["packaged_submission_file"]["row_count"] == 2


def test_manual_submission_package_verifier_detects_file_drift(tmp_path: Path, monkeypatch):
    competition_dir, memory, package, _workflow = _build_titanic_recommended_manual_package(tmp_path, monkeypatch)
    manifest_path = package.package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "ready_for_manual_upload"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ok = ManualSubmissionPackageVerifier(competition_dir, memory=memory).verify()
    ok_report = json.loads(ok.report_path.read_text(encoding="utf-8"))
    assert ok.status == "pass"
    assert ok.experiment_roadmap_path is not None
    assert ok_report["experiment_roadmap_path"] == str(ok.experiment_roadmap_path)
    assert ok_report["decision"] == "package_verified_for_upload"
    assert ok_report["actual_submission_file"]["sha256"] == ok_report["manifest_submission_file"]["sha256"]
    roadmap = json.loads(ok.experiment_roadmap_path.read_text(encoding="utf-8"))
    assert roadmap["top_action"]["action_id"] == "manual_upload_and_feedback_capture"
    ledger_entries = [
        json.loads(line)
        for line in (competition_dir / "runs" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verifier_entry = next(
        entry for entry in reversed(ledger_entries) if entry["task_id"] == "manual_submission_package_verification"
    )
    verifier_artifact = competition_dir / verifier_entry["artifacts_dir"] / "manual_submission_package_verification.json"
    verifier_artifact_payload = json.loads(verifier_artifact.read_text(encoding="utf-8"))
    assert verifier_artifact_payload["experiment_roadmap_path"] == str(ok.experiment_roadmap_path)
    assert (competition_dir / verifier_entry["artifacts_dir"] / "experiment_roadmap.json").exists()

    (package.package_dir / "submission.csv").write_text(
        "PassengerId,Survived\n892,1\n893,1\n",
        encoding="utf-8",
    )
    blocked = ManualSubmissionPackageVerifier(competition_dir, memory=memory).verify()
    blocked_report = json.loads(blocked.report_path.read_text(encoding="utf-8"))

    assert blocked.status == "needs_review"
    assert blocked.experiment_roadmap_path is not None
    assert blocked_report["decision"] == "package_verification_blocked"
    assert any("submission SHA-256 mismatch" in issue for issue in blocked_report["issues"])
    assert any("submission row count mismatch" in issue for issue in blocked_report["issues"])
    assert any("template expected submission SHA-256 mismatch" in issue for issue in blocked_report["issues"])


def test_experiment_queue_builds_from_remote_brain_plan(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    (competition_dir / "llm_experiment_plan.json").write_text(
        json.dumps(
            {
                "next_action": "recommend_experiments",
                "leaderboard_diagnosis": {"risk_level": "medium"},
                "recommended_experiments": [
                    {
                        "task_id": "safe_engineered_features_v1",
                        "title": "Run safe engineered features",
                        "expected_gain": "medium",
                        "risk": "low",
                        "compute_cost": "medium",
                        "coding_agent_task": "Train a safer feature engineered model.",
                    },
                    {
                        "task_id": "champion_blend_lb_submit",
                        "title": "Submit champion blend to Kaggle leaderboard",
                        "coding_agent_task": "Submit current champion blend and record public leaderboard score.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ExperimentQueueBuilder(competition_dir, memory=CompetitionMemory(tmp_path / "memory")).build()
    queue = json.loads(result.queue_path.read_text(encoding="utf-8"))
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")

    assert result.status == "ready"
    assert queue["next_runnable"]["task_id"] == "safe_engineered_features_v1"
    assert queue["queue"][0]["action_type"] == "coding_experiment"
    assert queue["queue"][1]["action_type"] == "manual_submit"
    assert queue["queue"][1]["status"] == "manual_gate"
    assert "下一批实验队列" in html
    assert "safe_engineered_features_v1" in html
    assert "experiment_queue.json" in html


def test_submission_decision_review_blocks_champion_submit_after_stability_gap(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "decision": "champion_selected",
                "champion": {
                    "task_id": "tabular_model_search_v1",
                    "metric_name": "accuracy",
                    "local_score": 0.846,
                    "risk_level": "low",
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "submission_policy.json").write_text(
        json.dumps(
            {
                "decision": "recommended_submission_selected",
                "recommended_submission_candidate": {
                    "task_id": "stability_first_search_v1",
                    "metric_name": "accuracy",
                    "local_score": 0.837,
                    "risk_level": "low",
                    "feature_set": "stable",
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_feedback.json").write_text(
        json.dumps(
            {
                "submission_target": "recommended",
                "candidate_task_id": "stability_first_search_v1",
                "public_score": 0.81234,
                "leaderboard_rank": 1234,
            }
        ),
        encoding="utf-8",
    )
    audit_dir = competition_dir / "experiments" / "cv_stability_audit_v1"
    audit_dir.mkdir(parents=True)
    (audit_dir / "cv_stability_audit.json").write_text(
        json.dumps(
            {
                "risk_level": "medium",
                "seed_mean": 0.829,
                "seed_std": 0.007,
                "fold_std": 0.023,
                "public_gap_vs_seed_mean": -0.017,
                "public_within_seed_ci": False,
                "issues": ["Public score is outside the seed-level 95% confidence interval."],
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "experiment_queue.json").write_text(
        json.dumps(
            {
                "queue": [
                    {
                        "task_id": "champion_blend_lb_submit",
                        "status": "manual_gate",
                        "action_type": "manual_submit",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    memory = CompetitionMemory(tmp_path / "memory")
    result = SubmissionDecisionReviewer(competition_dir, memory=memory).review(
        queue_task_id="champion_blend_lb_submit",
        submission_target="champion",
    )
    ExperimentQueueBuilder(competition_dir, memory=memory).build()
    review = json.loads(result.review_path.read_text(encoding="utf-8"))
    queue = json.loads((competition_dir / "experiment_queue.json").read_text(encoding="utf-8"))

    assert result.status == "needs_review"
    assert review["decision"] == "pause_manual_submit"
    assert "Public score is outside the seed-level confidence interval." in review["issues"]
    assert queue["queue"][0]["status"] == "blocked"
    assert queue["queue"][0]["submission_decision"] == "pause_manual_submit"
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Submission Decision" in html
    assert "人工提交审核" in html
    assert "pause_manual_submit" in html


def test_kaggle_env_preflight_reports_safe_setup_without_secrets(tmp_path: Path, monkeypatch):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("AUTOKAGGLE_REMOTE_WORKSPACE", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "demo_user")
    monkeypatch.setenv("KAGGLE_KEY", "super_secret_key")
    memory = CompetitionMemory(tmp_path / "memory")
    result = KaggleSubmitAdapter(competition_dir, memory=memory).preflight_environment()

    assert result.status == "pass"
    assert (competition_dir / "kaggle_env_preflight.json").exists()
    report = (competition_dir / "kaggle_env_preflight.json").read_text(encoding="utf-8")
    assert "kaggle_cli_available" in report
    assert "env_pair_present" in report
    assert "super_secret_key" not in report
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Check Kaggle environment preflight" in html
    assert memory.query(competition_name="titanic_copy")


def test_kaggle_submit_adapter_builds_dry_run_plan(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    result = KaggleSubmitAdapter(competition_dir, memory=memory).plan(dry_run=True)

    assert result.status == "pass"
    assert (competition_dir / "kaggle_submit_plan.json").exists()
    plan = (competition_dir / "kaggle_submit_plan.json").read_text(encoding="utf-8")
    assert "submit_command_preview" in plan
    assert "KAGGLE_KEY" not in plan
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Build Kaggle submit dry-run plan" in html


def test_kaggle_submit_confirmed_blocks_without_approval(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    KaggleSubmitAdapter(competition_dir, memory=memory).plan(dry_run=True)
    result = KaggleSubmitAdapter(competition_dir, memory=memory).confirmed_submit(confirmed=True)

    assert result.status == "blocked"
    payload = (competition_dir / "kaggle_submit_result.json").read_text(encoding="utf-8")
    assert "approve_real_submit" in payload
    assert "KAGGLE_KEY" not in payload
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Attempt confirmed Kaggle submission" in html


def test_leaderboard_feedback_records_public_score(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    result = LeaderboardFeedbackRecorder(competition_dir, memory=memory).record(
        public_score=0.81234,
        leaderboard_rank=1234,
        submission_id="demo-submission",
        source="manual",
        notes="smoke test feedback",
    )

    assert result.status == "pass"
    payload = (competition_dir / "leaderboard_feedback.json").read_text(encoding="utf-8")
    assert "0.81234" in payload
    assert "demo-submission" in payload
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Record leaderboard feedback" in html
    summary = memory.leaderboard_summary(profile_name="tabular_classic")
    assert summary["best_public_score"] == 0.81234
    assert summary["best_leaderboard_rank"] == 1234


def test_leaderboard_feedback_loop_runs_gap_audit_and_brain_plan(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    result = LeaderboardFeedbackLoop(competition_dir, memory=memory).run(
        public_score=0.81234,
        leaderboard_rank=1234,
        submission_id="demo-submission",
        source="manual",
        notes="loop smoke feedback",
        brain_use_llm=False,
    )

    assert result.status in {"pass", "needs_review"}
    assert result.gap_audit_path is not None
    assert result.brain_plan_path is not None
    assert result.experiment_queue_path is not None
    assert result.experiment_roadmap_path is not None
    report = json.loads((competition_dir / "leaderboard_feedback_loop.json").read_text(encoding="utf-8"))
    assert report["leaderboard_feedback_status"] == "pass"
    assert report["gap_audit_status"] == "completed"
    assert report["brain_review_used_llm"] is False
    assert report["next_recommended_experiments"]
    assert report["experiment_queue_status"] == "ready"
    assert report["experiment_roadmap_path"] == str(result.experiment_roadmap_path)
    assert report["next_runnable"]["task_id"]
    assert report["next_command"]
    queue = json.loads(result.experiment_queue_path.read_text(encoding="utf-8"))
    assert queue["next_runnable"]["task_id"] == report["next_runnable"]["task_id"]
    roadmap = json.loads(result.experiment_roadmap_path.read_text(encoding="utf-8"))
    assert roadmap["top_action"]
    ledger_entries = [
        json.loads(line)
        for line in (competition_dir / "runs" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loop_entry = next(entry for entry in reversed(ledger_entries) if entry["task_id"] == "leaderboard_feedback_loop")
    loop_artifact = competition_dir / loop_entry["artifacts_dir"] / "leaderboard_feedback_loop.json"
    artifact_payload = json.loads(loop_artifact.read_text(encoding="utf-8"))
    assert artifact_payload["experiment_roadmap_path"] == str(result.experiment_roadmap_path)
    assert (competition_dir / loop_entry["artifacts_dir"] / "experiment_roadmap.json").exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Close leaderboard feedback loop" in html
    assert "提交后 Brain 决策" in html


def test_leaderboard_gap_auditor_flags_public_cv_gap(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    ExperimentChampionSelector(competition_dir, memory=memory).select()
    SubmissionGate(competition_dir, memory=memory).run(dry_run=True)
    LeaderboardFeedbackRecorder(competition_dir, memory=memory).record(
        public_score=0.40,
        leaderboard_rank=9999,
        submission_id="gap-demo",
    )
    result = LeaderboardGapAuditor(competition_dir, memory=memory).audit()

    assert result.status == "needs_review"
    report = json.loads((competition_dir / "leaderboard_gap_audit.json").read_text(encoding="utf-8"))
    assert report["score_gap"]["materially_worse"] is True
    assert report["risk_level"] == "high"
    assert report["data_drift"]["status"] == "completed"
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Audit leaderboard gap and stability" in html


def test_stability_first_runner_prepares_drop_features(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    gap_audit = {
        "risk_level": "high",
        "data_drift": {
            "top_features": [
                {"feature": "Name", "drift_score": 1.2},
                {"feature": "Ticket", "drift_score": 0.8},
                {"feature": "Sex", "drift_score": 0.0},
            ]
        },
    }
    (competition_dir / "leaderboard_gap_audit.json").write_text(json.dumps(gap_audit), encoding="utf-8")
    report_path = StabilityFirstRunner(competition_dir).prepare_feature_report()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["drop_features"] == ["Title", "TicketPrefix"]
    assert report["risk_level"] == "high"


def test_iteration_orchestrator_runs_review_and_enhancement(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(
        ["sample_submission_baseline", "target_frequency_or_mean_baseline"]
    )
    result = IterationOrchestrator(competition_dir, memory=memory, use_llm=False).run(
        max_iterations=1,
        patience=2,
    )

    assert result.iterations_completed == 1
    assert result.summary_path.exists()
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "iterations_completed" in summary
    assert "best_state" in summary
    assert (competition_dir / "best_score.json").exists()
    html = (competition_dir / "runs" / "index.html").read_text(encoding="utf-8")
    assert "Summarize automatic optimization loop" in html


def test_iteration_orchestrator_stops_when_target_reached(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (competition_dir / "experiments" / "seed").mkdir(parents=True)
    (competition_dir / "experiments" / "seed" / "validation_report.json").write_text(
        '{"status": "completed", "metric_name": "accuracy", "local_score": 0.9}',
        encoding="utf-8",
    )

    result = IterationOrchestrator(competition_dir, use_llm=False).run(
        max_iterations=2,
        target_score=0.8,
    )

    assert result.iterations_completed == 0
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "target_score_reached_before_iteration" in summary


def test_agent_loop_stops_for_local_target_and_writes_goal_state(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    exp_dir = competition_dir / "experiments" / "strong_local"
    exp_dir.mkdir(parents=True)
    (exp_dir / "validation_report.json").write_text(
        json.dumps({"experiment": "strong_local", "status": "completed", "metric_name": "accuracy", "local_score": 0.91}),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_target.json").write_text(
        json.dumps({"estimated_silver_score": 0.90, "metric_name": "accuracy"}),
        encoding="utf-8",
    )

    result = AgentLoopController(competition_dir, use_llm=False, refresh_leaderboard=False).run(
        target="silver",
        max_iterations=2,
    )

    assert result.decision == "prepare_manual_submit"
    assert result.iterations_completed == 0
    assert (competition_dir / "goal_spec.json").exists()
    assert (competition_dir / "agent_loop_state.json").exists()
    assert (competition_dir / "champion_state.json").exists()
    assert (competition_dir / "candidate_pool.json").exists()
    goal = json.loads((competition_dir / "goal_spec.json").read_text(encoding="utf-8"))
    assert goal["target_score"] == 0.90


def test_agent_loop_runs_one_goal_oriented_iteration(tmp_path: Path):
    competition_dir = tmp_path / "titanic_copy"
    competition_dir.mkdir()
    source = COMPETITION_ROOT / "titanic"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "overview.txt"]:
        (competition_dir / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (competition_dir / "leaderboard_target.json").write_text(
        json.dumps({"estimated_silver_score": 0.99, "metric_name": "accuracy"}),
        encoding="utf-8",
    )
    memory = CompetitionMemory(tmp_path / "memory")
    TabularBaselineRunner(competition_dir, memory=memory).run_all(["sample_submission_baseline"])

    result = AgentLoopController(competition_dir, memory=memory, use_llm=False, refresh_leaderboard=False).run(
        target="silver",
        max_iterations=1,
    )

    assert result.iterations_completed == 1
    assert result.mac_brain_handoff_path.exists()
    assert result.decision in {
        "continue_exploit",
        "continue_explore",
        "build_ensemble",
        "prepare_manual_submit",
        "run_validation_audit",
    }
    state = json.loads((competition_dir / "agent_loop_state.json").read_text(encoding="utf-8"))
    pool = json.loads((competition_dir / "candidate_pool.json").read_text(encoding="utf-8"))
    assert state["iterations_completed"] == 1
    assert pool["candidates"] or pool["failed_candidates"]
    assert pool["target_score"] == 0.99
    assert "candidate_count" in pool
    handoff = json.loads(result.mac_brain_handoff_path.read_text(encoding="utf-8"))
    assert handoff["status"] in {"remote_autonomy_continues", "handoff_required"}
    assert handoff["control_plane"]["mode"] == "remote_autonomous_until_handoff"
    assert "Agent Loop Summary" in result.summary_path.read_text(encoding="utf-8")


def test_agent_loop_candidate_pool_names_run_artifacts_by_parent_run(tmp_path: Path):
    competition_dir = tmp_path / "demo_competition"
    competition_dir.mkdir()
    artifacts = competition_dir / "runs" / "0006_baseline_target_frequency_or_mean_baseline" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "validation_report.json").write_text(
        json.dumps({"status": "completed", "metric_name": "accuracy", "local_score": 0.65}),
        encoding="utf-8",
    )
    (artifacts / "validator_result.json").write_text(
        json.dumps({"ok": True, "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    (artifacts / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (competition_dir / "champion_selection.json").write_text(
        json.dumps(
            {
                "champion": {
                    "task_id": "baseline_target_frequency_or_mean_baseline",
                    "metric_name": "accuracy",
                    "local_score": 0.65,
                }
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_target.json").write_text(
        json.dumps({"estimated_silver_score": 0.80, "metric_name": "accuracy"}),
        encoding="utf-8",
    )

    result = AgentLoopController(competition_dir, use_llm=False, refresh_leaderboard=False).run(
        target="silver",
        max_iterations=0,
    )
    pool = json.loads(result.candidate_pool_path.read_text(encoding="utf-8"))

    assert pool["target_score"] == 0.80
    assert pool["champion_score"] == 0.65
    assert pool["gap_to_target"] == pytest.approx(0.15)
    assert pool["candidate_count"] == 1
    assert pool["candidates"][0]["task_id"] == "baseline_target_frequency_or_mean_baseline"
    assert pool["candidates"][0]["task_id"] != "artifacts"


def test_agent_loop_escalates_to_mac_brain_after_no_improvement_threshold(tmp_path: Path):
    competition_dir = tmp_path / "plateau_competition"
    competition_dir.mkdir()
    exp_dir = competition_dir / "experiments" / "plateau_champion"
    exp_dir.mkdir(parents=True)
    (exp_dir / "validation_report.json").write_text(
        json.dumps({"experiment": "plateau_champion", "status": "completed", "metric_name": "accuracy", "local_score": 0.70}),
        encoding="utf-8",
    )
    (competition_dir / "leaderboard_target.json").write_text(
        json.dumps({"estimated_silver_score": 0.90, "metric_name": "accuracy"}),
        encoding="utf-8",
    )
    (competition_dir / "goal_spec.json").write_text(
        json.dumps(
            {
                "target": "silver",
                "metric_name": "accuracy",
                "target_score": 0.90,
                "stop_conditions": {
                    "public_score_reaches_target": True,
                    "no_improvement_rounds": 2,
                    "validator_failure_rounds": 2,
                    "mac_handoff_no_improvement_rounds": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    (competition_dir / "agent_loop_state.json").write_text(
        json.dumps({"no_improvement_rounds": 3, "validator_failure_rounds": 0}),
        encoding="utf-8",
    )

    result = AgentLoopController(competition_dir, use_llm=False, refresh_leaderboard=False).run(
        target="silver",
        max_iterations=1,
    )
    handoff = json.loads(result.mac_brain_handoff_path.read_text(encoding="utf-8"))
    markdown = (competition_dir / "mac_brain_handoff.md").read_text(encoding="utf-8")

    assert result.decision == "escalate_to_mac_brain"
    assert result.iterations_completed == 0
    assert handoff["handoff_required"] is True
    assert handoff["status"] == "handoff_required"
    assert "strategic replanning" in " ".join(handoff["reasons"])
    assert "Mac Brain Handoff" in markdown
