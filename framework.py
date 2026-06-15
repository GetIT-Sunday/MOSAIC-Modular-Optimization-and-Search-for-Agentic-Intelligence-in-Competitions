import os
import sys
import shlex
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
import subprocess

import sys


PROJECT_DIR = Path(__file__).resolve().parent
PREFIX_MULTI_AGENTS_PATH = PROJECT_DIR / "multi_agents"


def run_remote_command(competition: str, mode_flag: str) -> None:
    if os.getenv("AUTOKAGGLE_REMOTE_WORKSPACE") == "1":
        raise RuntimeError("Refusing to recursively launch remote execution from remote workspace.")

    subprocess.run(["bash", str(PROJECT_DIR / "scripts" / "sync_to_dev.sh")], check=True)
    remote_command = (
        "cd workspaces/AutoKaggle && "
        f"python framework.py --competition {competition} {mode_flag} --execution-backend local"
    )
    subprocess.run(["bash", str(PROJECT_DIR / "scripts" / "remote_dev.sh"), remote_command], check=True)
    subprocess.run(
        ["bash", str(PROJECT_DIR / "scripts" / "sync_from_dev.sh"), competition],
        check=True,
    )


def refresh_project_control_panel() -> Path:
    from multi_agents.orchestration import ProjectControlPanel

    return ProjectControlPanel(PREFIX_MULTI_AGENTS_PATH / "competition").write_html()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SOP for a competition.')
    parser.add_argument('--competition', type=str, default='titanic', help='Competition name')
    parser.add_argument('--model', type=str, default='config', help='Planner model name, or "config" to use multi_agents/config.json and environment variables')
    parser.add_argument('--task-card-mode', action='store_true', help='Run the Brain task-card dry loop instead of the legacy SOP.')
    parser.add_argument('--competition-intake', action='store_true', help='Parse selected Kaggle competition intake and generate Brain-ready artifacts.')
    parser.add_argument('--agent-baseline-start', action='store_true', help='Run intake, baseline, and roadmap preparation for a selected competition.')
    parser.add_argument('--project-control-panel', action='store_true', help='Generate the global AutoKaggle project control panel.')
    parser.add_argument('--config-check', action='store_true', help='Check shareable AutoKaggle SSH, remote, LLM, and Kaggle configuration without printing secrets.')
    parser.add_argument('--remote-health-check', action='store_true', help='Run read-only SSH, workspace, conda, disk, GPU, and Kaggle remote health diagnostics.')
    parser.add_argument('--run-baselines', action='store_true', help='Run deterministic baseline experiments and write Run Ledger entries.')
    parser.add_argument('--run-enhancement', action='store_true', help='Run the next uncompleted Remote Brain recommended enhancement experiment.')
    parser.add_argument('--experiment-queue', action='store_true', help='Build a visible queue from Remote Brain recommended experiments.')
    parser.add_argument('--experiment-roadmap', action='store_true', help='Build a prioritized next-action roadmap from queue, gates, package, and leaderboard feedback.')
    parser.add_argument('--tabular-search', action='store_true', help='Run a compact multi-model tabular search and blend.')
    parser.add_argument('--tabular-task-id', type=str, default=None, help='Optional task id for tabular search or tabular risk audit artifacts.')
    parser.add_argument('--tabular-search-seeds', type=str, default='42', help='Comma-separated CV seeds for --tabular-search.')
    parser.add_argument('--tabular-feature-set', choices=['all', 'pruned', 'stable', 'leakage_safe'], default='all', help='Feature set for --tabular-search.')
    parser.add_argument('--tabular-risk-audit', action='store_true', help='Audit CV stability and leaderboard risk for the latest tabular search.')
    parser.add_argument('--tabular-leakage-audit', action='store_true', help='Audit tabular feature leakage, transform-scope risk, and train/test drift.')
    parser.add_argument('--tabular-feature-prune', action='store_true', help='Run tabular feature importance and pruning comparison.')
    parser.add_argument('--select-champion', action='store_true', help='Select the current champion submission from valid experiment artifacts.')
    parser.add_argument('--submission-gate', action='store_true', help='Dry-run final gate before Kaggle submission.')
    parser.add_argument('--submission-target', choices=['champion', 'recommended'], default='champion', help='Submission target for submission gate and Kaggle dry-run.')
    parser.add_argument('--kaggle-env-preflight', action='store_true', help='Check Kaggle CLI, pytest, credentials, and remote workspace safety without submitting.')
    parser.add_argument('--kaggle-discover', action='store_true', help='Refresh Kaggle competition discovery cache through the official Kaggle CLI.')
    parser.add_argument('--kaggle-select', type=str, default=None, help='Select a Kaggle competition slug and create competition_intake.json.')
    parser.add_argument('--kaggle-download', action='store_true', help='Download files when used with --kaggle-select after rules are accepted.')
    parser.add_argument('--kaggle-group', type=str, default='general', help='Kaggle competition group for --kaggle-discover.')
    parser.add_argument('--kaggle-category', type=str, default='all', help='Kaggle competition category for --kaggle-discover.')
    parser.add_argument('--kaggle-sort-by', type=str, default='recentlyCreated', help='Kaggle competition sort order for --kaggle-discover.')
    parser.add_argument('--kaggle-search', type=str, default='', help='Kaggle competition search term for --kaggle-discover.')
    parser.add_argument('--kaggle-page', type=int, default=1, help='Kaggle competition result page for --kaggle-discover.')
    parser.add_argument('--kaggle-submit-dry-run', action='store_true', help='Build a dry-run Kaggle submit command plan without submitting.')
    parser.add_argument('--kaggle-submit-confirmed', action='store_true', help='Attempt real Kaggle submission only if all explicit approval gates pass.')
    parser.add_argument('--leaderboard-feedback', action='store_true', help='Record manual or API leaderboard feedback for the current champion.')
    parser.add_argument('--fill-leaderboard-feedback-template', action='store_true', help='Fill a leaderboard feedback template from Kaggle-returned score/rank/submission id.')
    parser.add_argument('--leaderboard-feedback-from-template', action='store_true', help='Validate leaderboard feedback JSON and run the feedback loop.')
    parser.add_argument('--leaderboard-feedback-loop', action='store_true', help='Record leaderboard feedback, run gap audit, and refresh Brain next-step plan.')
    parser.add_argument('--leaderboard-target', action='store_true', help='Fetch Kaggle leaderboard target snapshot and estimate top/silver gaps for Brain planning.')
    parser.add_argument('--leaderboard-page-size', type=int, default=200, help='Leaderboard entries to request for --leaderboard-target.')
    parser.add_argument('--leaderboard-gap-audit', action='store_true', help='Audit local CV versus public leaderboard gap and train/test drift.')
    parser.add_argument('--stability-first', action='store_true', help='Run a stability-first repeated-CV search after leaderboard gap audit.')
    parser.add_argument('--post-reselection-gate', action='store_true', help='Refresh submission gate and Kaggle dry-run plan after champion reselection.')
    parser.add_argument('--submission-policy', action='store_true', help='Recommend a submission candidate separately from the highest-CV champion.')
    parser.add_argument('--promotion-gate-review', action='store_true', help='Evaluate Remote Brain promotion gates against experiment evidence.')
    parser.add_argument('--post-experiment-pipeline', action='store_true', help='Run promotion, submission policy, submission gate, readiness, and handoff after an experiment.')
    parser.add_argument('--submission-decision-review', action='store_true', help='Review whether a manual submission queue item should proceed.')
    parser.add_argument('--manual-submit-readiness', action='store_true', help='Summarize manual/API Kaggle submit readiness without submitting.')
    parser.add_argument('--submit-decision-handoff', action='store_true', help='Create an auditable human handoff before manual leaderboard submission.')
    parser.add_argument('--manual-submission-package', action='store_true', help='Package the exact submission file, feedback template, and checklist for manual upload.')
    parser.add_argument('--verify-manual-submission-package', action='store_true', help='Verify an existing manual submission package before upload without rebuilding it.')
    parser.add_argument('--post-submit-workflow', action='store_true', help='Create the manual upload to leaderboard-feedback workflow checklist.')
    parser.add_argument('--public-score', type=float, default=None, help='Public leaderboard score to record with --leaderboard-feedback.')
    parser.add_argument('--private-score', type=float, default=None, help='Private leaderboard score to record with --leaderboard-feedback.')
    parser.add_argument('--leaderboard-rank', type=int, default=None, help='Leaderboard rank to record with --leaderboard-feedback.')
    parser.add_argument('--submission-id', type=str, default=None, help='Kaggle submission id to record with --leaderboard-feedback.')
    parser.add_argument('--feedback-source', type=str, default='manual', help='Feedback source, such as manual or kaggle_cli.')
    parser.add_argument('--feedback-notes', type=str, default='', help='Short notes to attach to --leaderboard-feedback.')
    parser.add_argument('--feedback-template', type=str, default=None, help='Path to filled leaderboard feedback JSON for --leaderboard-feedback-from-template.')
    parser.add_argument('--run-filled-feedback-loop', action='store_true', help='After --fill-leaderboard-feedback-template, immediately validate the filled template and run the feedback loop.')
    parser.add_argument('--agent-loop', action='store_true', help='Run the goal-oriented AutoKaggle Agent Loop controller.')
    parser.add_argument('--target', choices=['silver', 'top10', 'top'], default='silver', help='Leaderboard-oriented target for --agent-loop.')
    parser.add_argument('--iterate', action='store_true', help='Run remote Brain review and enhancement iterations.')
    parser.add_argument('--remote-brain-review', action='store_true', help='Run the remote project Brain review over latest experiment artifacts.')
    parser.add_argument('--no-brain-llm', action='store_true', help='Use deterministic Remote Brain fallback planning instead of calling an LLM.')
    parser.add_argument('--max-iterations', type=int, default=1, help='Maximum iterations for --iterate.')
    parser.add_argument('--patience', type=int, default=2, help='Stop --iterate after this many non-improving iterations.')
    parser.add_argument('--target-score', type=float, default=None, help='Stop --iterate once the best score reaches this target.')
    parser.add_argument(
        '--execution-backend',
        choices=['remote_linux', 'local'],
        default='remote_linux',
        help='Execution backend for task-card and baseline modes. Defaults to remote_linux from the local control machine.',
    )
    args = parser.parse_args()
    competition = args.competition
    model = args.model

    if args.project_control_panel:
        path = refresh_project_control_panel()
        print(f"Project control panel generated: {path}")
        sys.exit(0)

    if args.config_check:
        from multi_agents.orchestration import ProjectConfigAgent

        agent = ProjectConfigAgent(PROJECT_DIR)
        status_path = PROJECT_DIR / "multi_agents" / "competition" / "console" / "config_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        agent.write_status(status_path)
        snapshot = agent.snapshot()
        print(f"Config status: {status_path}")
        print(f"Private config: {'present' if snapshot.using_private_config else 'missing'}")
        for check in snapshot.checks:
            print(f"{check.status}: {check.label} - {check.detail}")
        sys.exit(0)

    if args.remote_health_check:
        from multi_agents.orchestration import RemoteHealthCheckAgent

        result = RemoteHealthCheckAgent(PROJECT_DIR).run()
        print(f"Remote health status: {result.status}")
        print(f"Report: {result.report_path}")
        for check in result.checks:
            print(f"{check.status}: {check.label} - {check.detail}")
        refresh_project_control_panel()
        sys.exit(0)

    if args.kaggle_discover:
        from multi_agents.orchestration import KaggleDiscoveryAgent

        result = KaggleDiscoveryAgent(PREFIX_MULTI_AGENTS_PATH / "competition").discover(
            group=args.kaggle_group,
            category=args.kaggle_category,
            sort_by=args.kaggle_sort_by,
            search=args.kaggle_search,
            page=args.kaggle_page,
        )
        refresh_project_control_panel()
        print("Kaggle competition discovery completed.")
        print(f"Status: {result.status}")
        print(f"Cache: {result.cache_path}")
        print(f"Competitions: {len(result.competitions)}")
        for issue in result.issues:
            print(f"Issue: {issue}")
        sys.exit(0)

    if args.kaggle_select:
        from multi_agents.orchestration import KaggleDiscoveryAgent

        result = KaggleDiscoveryAgent(PREFIX_MULTI_AGENTS_PATH / "competition").select(
            args.kaggle_select,
            download=args.kaggle_download,
        )
        refresh_project_control_panel()
        print(f"Kaggle competition selected: {result.competition_slug}")
        print(f"Status: {result.status}")
        print(f"Competition dir: {result.competition_dir}")
        print(f"Intake: {result.intake_path}")
        print(f"Files: {len(result.files)}")
        for issue in result.issues:
            print(f"Issue: {issue}")
        sys.exit(0)

    if args.task_card_mode:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--task-card-mode")
            print(f"Remote Brain task-card dry loop completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import BrainOrchestrator

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        decision = BrainOrchestrator().run_dry_loop(competition_dir)
        print(f"Brain task-card dry loop completed for {competition}.")
        print(f"Profile: {decision.profile.name}")
        print(f"Tasks: {len(decision.coding_tasks)}")
        sys.exit(0)

    if args.competition_intake:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--competition-intake")
            print(f"Remote competition intake completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import CompetitionIntakeAgent

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = CompetitionIntakeAgent(competition_dir).run()
        refresh_project_control_panel()
        print(f"Competition intake completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Intake: {result.intake_path}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Unknown fields: {', '.join(result.unknown_fields) if result.unknown_fields else 'none'}")
        print(f"Blocking items: {', '.join(result.blocking_items) if result.blocking_items else 'none'}")
        print(f"Next command: {result.next_command}")
        sys.exit(0)

    if args.agent_baseline_start:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--agent-baseline-start")
            print(f"Remote agent baseline start completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import CompetitionIntakeAgent, ExperimentRoadmapBuilder, TabularBaselineRunner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        intake_result = CompetitionIntakeAgent(competition_dir).run()
        print(f"Competition intake completed for {competition}.")
        print(f"Status: {intake_result.status}")
        if intake_result.status != "ready_for_baseline":
            print(f"Blocking items: {', '.join(intake_result.blocking_items)}")
            print(f"Next command: {intake_result.next_command}")
            sys.exit(2)
        baseline_results = TabularBaselineRunner(competition_dir).run_all()
        roadmap_result = ExperimentRoadmapBuilder(competition_dir).build()
        refresh_project_control_panel()
        print(f"Agent baseline start completed for {competition}.")
        for result in baseline_results:
            print(f"{result.task_id}: {result.status}")
        print(f"Roadmap: {roadmap_result.roadmap_path}")
        sys.exit(0)

    if args.run_baselines:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--run-baselines")
            print(f"Remote baseline runner completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import TabularBaselineRunner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        results = TabularBaselineRunner(competition_dir).run_all()
        refresh_project_control_panel()
        print(f"Baseline runner completed for {competition}.")
        for result in results:
            print(f"{result.task_id}: {result.status}")
        sys.exit(0)

    if args.remote_brain_review:
        if args.execution_backend == 'remote_linux':
            no_llm = " --no-brain-llm" if args.no_brain_llm else ""
            run_remote_command(competition, f"--remote-brain-review{no_llm}")
            print(f"Remote Brain review completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import RemoteBrainReviewer

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = RemoteBrainReviewer(competition_dir, use_llm=not args.no_brain_llm).review()
        refresh_project_control_panel()
        print(f"Remote Brain review completed for {competition}.")
        print(f"Plan: {result.json_path}")
        print(f"Used LLM: {result.used_llm}")
        sys.exit(0)

    if args.experiment_queue:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--experiment-queue")
            print(f"Remote experiment queue completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import ExperimentQueueBuilder

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = ExperimentQueueBuilder(competition_dir).build()
        refresh_project_control_panel()
        print(f"Experiment queue completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Queue: {result.queue_path}")
        print(f"Markdown: {result.markdown_path}")
        sys.exit(0)

    if args.experiment_roadmap:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--experiment-roadmap")
            print(f"Remote experiment roadmap completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import ExperimentRoadmapBuilder

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = ExperimentRoadmapBuilder(competition_dir).build()
        print(f"Experiment roadmap completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Roadmap: {result.roadmap_path}")
        print(f"Markdown: {result.markdown_path}")
        sys.exit(0)

    if args.run_enhancement:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--run-enhancement")
            print(f"Remote enhancement runner completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import EnhancementRunner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = EnhancementRunner(competition_dir).run_next_recommendation()
        print(f"Enhancement runner completed for {competition}.")
        print(f"{result.task_id}: {result.status}")
        sys.exit(0)

    if args.tabular_search:
        if args.execution_backend == 'remote_linux':
            task_flag = f" --tabular-task-id {args.tabular_task_id}" if args.tabular_task_id else ""
            run_remote_command(
                competition,
                f"--tabular-search --tabular-search-seeds {args.tabular_search_seeds} --tabular-feature-set {args.tabular_feature_set}{task_flag}",
            )
            print(f"Remote tabular search completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import TabularSearchRunner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        seeds = [int(seed.strip()) for seed in args.tabular_search_seeds.split(",") if seed.strip()]
        task_id = args.tabular_task_id
        if task_id is None and args.tabular_feature_set == "leakage_safe":
            task_id = "leakage_safe_search_v1"
        result = TabularSearchRunner(competition_dir).run(
            task_id=task_id or "tabular_model_search_v1",
            cv_seeds=seeds,
            feature_set=args.tabular_feature_set,
        )
        print(f"Tabular search completed for {competition}.")
        print(f"{result.task_id}: {result.status}")
        sys.exit(0)

    if args.tabular_risk_audit:
        if args.execution_backend == 'remote_linux':
            task_flag = f" --tabular-task-id {args.tabular_task_id}" if args.tabular_task_id else ""
            run_remote_command(competition, f"--tabular-risk-audit{task_flag}")
            print(f"Remote tabular risk audit completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import TabularRiskAuditor

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = TabularRiskAuditor(competition_dir).audit(args.tabular_task_id or "tabular_model_search_v1")
        print(f"Tabular risk audit completed for {competition}.")
        print(f"{result.task_id}: {result.status}")
        print(f"Audit: {result.audit_path}")
        sys.exit(0)

    if args.tabular_leakage_audit:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--tabular-leakage-audit")
            print(f"Remote tabular feature leakage audit completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import TabularFeatureLeakageAuditor

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = TabularFeatureLeakageAuditor(competition_dir).audit()
        print(f"Tabular feature leakage audit completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Audit: {result.audit_path}")
        sys.exit(0)

    if args.tabular_feature_prune:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--tabular-feature-prune")
            print(f"Remote tabular feature pruning completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import TabularFeaturePruner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = TabularFeaturePruner(competition_dir).run()
        print(f"Tabular feature pruning completed for {competition}.")
        print(f"{result.task_id}: {result.status}")
        sys.exit(0)

    if args.select_champion:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--select-champion")
            print(f"Remote champion selection completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import ExperimentChampionSelector

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = ExperimentChampionSelector(competition_dir).select()
        print(f"Champion selection completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Selection: {result.selection_path}")
        print(f"Champion: {result.champion_path}")
        sys.exit(0)

    if args.submission_gate:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--submission-gate --submission-target {args.submission_target}")
            print(f"Remote submission gate completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import SubmissionGate

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = SubmissionGate(competition_dir).run(dry_run=True, submission_target=args.submission_target)
        print(f"Submission gate completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Gate: {result.gate_path}")
        sys.exit(0)

    if args.kaggle_env_preflight:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--kaggle-env-preflight")
            print(f"Remote Kaggle environment preflight completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import KaggleSubmitAdapter

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = KaggleSubmitAdapter(competition_dir).preflight_environment()
        print(f"Kaggle environment preflight completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.plan_path}")
        sys.exit(0)

    if args.kaggle_submit_dry_run:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--kaggle-submit-dry-run --submission-target {args.submission_target}")
            print(f"Remote Kaggle submit dry-run plan completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import KaggleSubmitAdapter

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = KaggleSubmitAdapter(competition_dir).plan(dry_run=True, submission_target=args.submission_target)
        print(f"Kaggle submit dry-run plan completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Plan: {result.plan_path}")
        sys.exit(0)

    if args.kaggle_submit_confirmed:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--kaggle-submit-confirmed --submission-target {args.submission_target}")
            print(f"Remote confirmed Kaggle submit path completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import KaggleSubmitAdapter

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = KaggleSubmitAdapter(competition_dir).confirmed_submit(confirmed=True, submission_target=args.submission_target)
        print(f"Confirmed Kaggle submit path completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Result: {result.plan_path}")
        sys.exit(0)

    if args.leaderboard_feedback:
        if args.execution_backend == 'remote_linux':
            mode_parts = ["--leaderboard-feedback"]
            if args.public_score is not None:
                mode_parts.extend(["--public-score", str(args.public_score)])
            if args.private_score is not None:
                mode_parts.extend(["--private-score", str(args.private_score)])
            if args.leaderboard_rank is not None:
                mode_parts.extend(["--leaderboard-rank", str(args.leaderboard_rank)])
            if args.submission_id:
                mode_parts.extend(["--submission-id", args.submission_id])
            mode_parts.extend(["--submission-target", args.submission_target])
            mode_parts.extend(["--feedback-source", args.feedback_source])
            if args.feedback_notes:
                mode_parts.extend(["--feedback-notes", args.feedback_notes.replace(" ", "_")])
            run_remote_command(competition, " ".join(mode_parts))
            print(f"Remote leaderboard feedback recorded for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import LeaderboardFeedbackRecorder

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = LeaderboardFeedbackRecorder(competition_dir).record(
            public_score=args.public_score,
            private_score=args.private_score,
            leaderboard_rank=args.leaderboard_rank,
            submission_id=args.submission_id,
            source=args.feedback_source,
            notes=args.feedback_notes,
            submission_target=args.submission_target,
        )
        print(f"Leaderboard feedback recorded for {competition}.")
        print(f"Status: {result.status}")
        print(f"Feedback: {result.feedback_path}")
        sys.exit(0)

    if args.fill_leaderboard_feedback_template:
        if args.execution_backend == 'remote_linux':
            mode_parts = ["--fill-leaderboard-feedback-template"]
            if args.public_score is not None:
                mode_parts.extend(["--public-score", str(args.public_score)])
            if args.private_score is not None:
                mode_parts.extend(["--private-score", str(args.private_score)])
            if args.leaderboard_rank is not None:
                mode_parts.extend(["--leaderboard-rank", str(args.leaderboard_rank)])
            if args.submission_id:
                mode_parts.extend(["--submission-id", shlex.quote(args.submission_id)])
            mode_parts.extend(["--feedback-source", shlex.quote(args.feedback_source)])
            if args.feedback_notes:
                mode_parts.extend(["--feedback-notes", shlex.quote(args.feedback_notes)])
            if args.feedback_template:
                mode_parts.extend(["--feedback-template", shlex.quote(args.feedback_template)])
            if args.run_filled_feedback_loop:
                mode_parts.append("--run-filled-feedback-loop")
            if args.no_brain_llm:
                mode_parts.append("--no-brain-llm")
            run_remote_command(competition, " ".join(mode_parts))
            print(f"Remote leaderboard feedback template filled for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import LeaderboardFeedbackTemplateFiller

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        input_path = Path(args.feedback_template) if args.feedback_template else None
        result = LeaderboardFeedbackTemplateFiller(competition_dir).fill(
            template_path=input_path,
            public_score=args.public_score,
            private_score=args.private_score,
            leaderboard_rank=args.leaderboard_rank,
            submission_id=args.submission_id,
            source=args.feedback_source,
            notes=args.feedback_notes or "post_submit_feedback",
            run_feedback_loop=args.run_filled_feedback_loop,
            brain_use_llm=not args.no_brain_llm,
        )
        print(f"Leaderboard feedback template filled for {competition}.")
        print(f"Status: {result.status}")
        print(f"Template: {result.filled_template_path}")
        print(f"Report: {result.report_path}")
        if result.feedback_loop_report_path:
            print(f"Feedback loop: {result.feedback_loop_report_path}")
        sys.exit(0)

    if args.leaderboard_feedback_from_template:
        if args.execution_backend == 'remote_linux':
            mode_parts = ["--leaderboard-feedback-from-template"]
            if args.feedback_template:
                mode_parts.extend(["--feedback-template", shlex.quote(args.feedback_template)])
            if args.no_brain_llm:
                mode_parts.append("--no-brain-llm")
            run_remote_command(competition, " ".join(mode_parts))
            print(f"Remote leaderboard feedback input completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import LeaderboardFeedbackInputRunner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        input_path = Path(args.feedback_template) if args.feedback_template else None
        result = LeaderboardFeedbackInputRunner(competition_dir).run(
            input_path=input_path,
            brain_use_llm=not args.no_brain_llm,
        )
        print(f"Leaderboard feedback input completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        if result.feedback_loop_report_path:
            print(f"Feedback loop: {result.feedback_loop_report_path}")
        sys.exit(0)

    if args.leaderboard_feedback_loop:
        if args.execution_backend == 'remote_linux':
            mode_parts = ["--leaderboard-feedback-loop"]
            if args.public_score is not None:
                mode_parts.extend(["--public-score", str(args.public_score)])
            if args.private_score is not None:
                mode_parts.extend(["--private-score", str(args.private_score)])
            if args.leaderboard_rank is not None:
                mode_parts.extend(["--leaderboard-rank", str(args.leaderboard_rank)])
            if args.submission_id:
                mode_parts.extend(["--submission-id", args.submission_id])
            mode_parts.extend(["--submission-target", args.submission_target])
            mode_parts.extend(["--feedback-source", args.feedback_source])
            if args.no_brain_llm:
                mode_parts.append("--no-brain-llm")
            if args.feedback_notes:
                mode_parts.extend(["--feedback-notes", args.feedback_notes.replace(" ", "_")])
            run_remote_command(competition, " ".join(mode_parts))
            print(f"Remote leaderboard feedback loop completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import LeaderboardFeedbackLoop

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = LeaderboardFeedbackLoop(competition_dir).run(
            public_score=args.public_score,
            private_score=args.private_score,
            leaderboard_rank=args.leaderboard_rank,
            submission_id=args.submission_id,
            source=args.feedback_source,
            notes=args.feedback_notes,
            submission_target=args.submission_target,
            brain_use_llm=not args.no_brain_llm,
        )
        print(f"Leaderboard feedback loop completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        sys.exit(0)

    if args.leaderboard_target:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--leaderboard-target --leaderboard-page-size {args.leaderboard_page_size}")
            print(f"Remote leaderboard target completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import LeaderboardTargetAgent

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = LeaderboardTargetAgent(competition_dir).run(page_size=args.leaderboard_page_size)
        print(f"Leaderboard target completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Target: {result.target_path}")
        print(f"Raw: {result.raw_path}")
        sys.exit(0)

    if args.leaderboard_gap_audit:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--leaderboard-gap-audit")
            print(f"Remote leaderboard gap audit completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import LeaderboardGapAuditor

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = LeaderboardGapAuditor(competition_dir).audit()
        print(f"Leaderboard gap audit completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Audit: {result.audit_path}")
        sys.exit(0)

    if args.stability_first:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--stability-first")
            print(f"Remote stability-first runner completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import StabilityFirstRunner

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = StabilityFirstRunner(competition_dir).run()
        print(f"Stability-first runner completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Review: {result.review_path}")
        sys.exit(0)

    if args.post_reselection_gate:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--post-reselection-gate --submission-target {args.submission_target}")
            print(f"Remote post-reselection gate completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import PostReselectionGate

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = PostReselectionGate(competition_dir).run(submission_target=args.submission_target)
        print(f"Post-reselection gate completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        sys.exit(0)

    if args.submission_policy:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--submission-policy")
            print(f"Remote submission policy completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import SubmissionPolicy

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = SubmissionPolicy(competition_dir).run()
        print(f"Submission policy completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Policy: {result.policy_path}")
        sys.exit(0)

    if args.promotion_gate_review:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--promotion-gate-review")
            print(f"Remote promotion gate review completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import PromotionGateEvaluator

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = PromotionGateEvaluator(competition_dir).evaluate()
        print(f"Promotion gate review completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Review: {result.review_path}")
        print(f"Markdown: {result.markdown_path}")
        if result.promoted_submission_path:
            print(f"Promoted submission: {result.promoted_submission_path}")
        sys.exit(0)

    if args.post_experiment_pipeline:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--post-experiment-pipeline --submission-target {args.submission_target}")
            print(f"Remote post-experiment pipeline completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import PostExperimentPipeline

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = PostExperimentPipeline(competition_dir).run(submission_target=args.submission_target)
        print(f"Post-experiment pipeline completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        sys.exit(0)

    if args.submission_decision_review:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--submission-decision-review --submission-target {args.submission_target}")
            print(f"Remote submission decision review completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import SubmissionDecisionReviewer

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = SubmissionDecisionReviewer(competition_dir).review(submission_target=args.submission_target)
        print(f"Submission decision review completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Review: {result.review_path}")
        print(f"Markdown: {result.markdown_path}")
        sys.exit(0)

    if args.manual_submit_readiness:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--manual-submit-readiness --submission-target {args.submission_target}")
            print(f"Remote manual submit readiness completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import ManualSubmitReadinessChecker

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = ManualSubmitReadinessChecker(competition_dir).run(submission_target=args.submission_target)
        print(f"Manual submit readiness completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        sys.exit(0)

    if args.submit_decision_handoff:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--submit-decision-handoff --submission-target {args.submission_target}")
            print(f"Remote submit decision handoff completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import SubmitDecisionHandoff

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = SubmitDecisionHandoff(competition_dir).run(submission_target=args.submission_target)
        print(f"Submit decision handoff completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        print(f"Markdown: {result.markdown_path}")
        sys.exit(0)

    if args.manual_submission_package:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--manual-submission-package --submission-target {args.submission_target}")
            print(f"Remote manual submission package completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import ManualSubmissionPackage

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = ManualSubmissionPackage(competition_dir).build(submission_target=args.submission_target)
        print(f"Manual submission package completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Checklist: {result.checklist_path}")
        sys.exit(0)

    if args.verify_manual_submission_package:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, "--verify-manual-submission-package")
            print(f"Remote manual submission package verification completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import ManualSubmissionPackageVerifier

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = ManualSubmissionPackageVerifier(competition_dir).verify()
        print(f"Manual submission package verification completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        sys.exit(0)

    if args.post_submit_workflow:
        if args.execution_backend == 'remote_linux':
            run_remote_command(competition, f"--post-submit-workflow --submission-target {args.submission_target}")
            print(f"Remote post-submit workflow completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import PostSubmitWorkflow

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = PostSubmitWorkflow(competition_dir).run(submission_target=args.submission_target)
        print(f"Post-submit workflow completed for {competition}.")
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        print(f"Checklist: {result.checklist_path}")
        sys.exit(0)

    if args.agent_loop:
        if args.execution_backend == 'remote_linux':
            target_score = f" --target-score {args.target_score}" if args.target_score is not None else ""
            no_llm = " --no-brain-llm" if args.no_brain_llm else ""
            run_remote_command(
                competition,
                f"--agent-loop --target {args.target} --max-iterations {args.max_iterations}{target_score}{no_llm}",
            )
            print(f"Remote Agent Loop completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import AgentLoopController

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = AgentLoopController(competition_dir, use_llm=not args.no_brain_llm).run(
            target=args.target,
            max_iterations=args.max_iterations,
            target_score=args.target_score,
        )
        refresh_project_control_panel()
        print(f"Agent Loop completed for {competition}.")
        print(f"Decision: {result.decision}")
        print(f"Iterations: {result.iterations_completed}")
        print(f"Goal: {result.goal_path}")
        print(f"State: {result.state_path}")
        print(f"Summary: {result.summary_path}")
        print(f"Mac Brain Handoff: {result.mac_brain_handoff_path}")
        sys.exit(0)

    if args.iterate:
        if args.execution_backend == 'remote_linux':
            target = f" --target-score {args.target_score}" if args.target_score is not None else ""
            no_llm = " --no-brain-llm" if args.no_brain_llm else ""
            run_remote_command(
                competition,
                f"--iterate --max-iterations {args.max_iterations} --patience {args.patience}{target}{no_llm}",
            )
            print(f"Remote optimization loop completed for {competition}.")
            sys.exit(0)

        from multi_agents.orchestration import IterationOrchestrator

        competition_dir = PREFIX_MULTI_AGENTS_PATH / "competition" / competition
        result = IterationOrchestrator(competition_dir, use_llm=not args.no_brain_llm).run(
            max_iterations=args.max_iterations,
            patience=args.patience,
            target_score=args.target_score,
        )
        print(f"Optimization loop completed for {competition}.")
        print(f"Iterations: {result.iterations_completed}")
        print(f"Summary: {result.summary_path}")
        sys.exit(0)

    from multi_agents.state import State
    from multi_agents.sop import SOP
    from utils import PREFIX_MULTI_AGENTS

    sop = SOP(competition, model)
    start_state = State(phase="Understand Background", competition=competition)
    start_message = ""
    new_state = start_state

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f"{PREFIX_MULTI_AGENTS}/competition/{competition}/{competition}.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Start SOP for competition: {competition}")
    while True:
        current_state = new_state
        exec_state_info, new_state = sop.step(state=current_state)
        if exec_state_info == 'Fail':
            logging.error("Failed to update state.")
            exit()
        if exec_state_info == 'Complete':
            logging.info(f"Competition {competition} SOP is completed.")
            break  
