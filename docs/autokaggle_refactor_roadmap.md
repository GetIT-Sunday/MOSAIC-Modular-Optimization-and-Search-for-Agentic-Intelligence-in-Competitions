# AutoKaggle Refactor Roadmap

## Principle

Do not build a single competition solution. Build a framework that can identify
different competition types and assemble the right workflow, tools, and
experiments for each one.

The local Mac is the control plane and project-level Brain. The remote Linux
workspace is the experiment plane and must run training, prediction, validation,
and project-internal Brain decisions that depend on generated artifacts.

## Milestones

### M1: Profile-Driven Orchestration

- Add neutral competition profiles. `[done]`
- Add profile loader. `[done]`
- Add Brain-level task identification. `[done]`
- Keep the original tabular SOP as a compatibility backend.

### M2: Competition Ingestion

- Add Kaggle metadata reader.
- Add data manifest builder. `[done: local competition directory]`
- Add rules and metric parser.
- Add sample submission analyzer.

### M3: Brain Agent

- Brain selects profile. `[done: deterministic skeleton]`
- Brain writes `task_card.md`.
- Brain writes `metric_spec.json`.
- Brain writes the first experiment ladder. `[done: in-memory CodingTask list]`
- Brain reviews Coding Agent outputs before continuing.
- Brain writes run ledger entries for human review. `[started]`

### M4: Coding Agent Contract

- Convert broad phase prompts into narrow executable tasks. `[started]`
- Require scripts, logs, metrics, artifacts, and failure reports.
- Make each task resumable.
- Add `human_review.md` gates before continuing or rerunning tasks. `[started]`
- Record baseline scripts, logs, reports, submissions, and validator outputs in Run Ledger. `[started]`
- Default experiment execution to remote Linux, then sync review artifacts back locally. `[started]`
- Add transparent HumanGate and Remote Brain review to produce the next LLM-guided experiment plan. `[started]`
- Add EnhancementRunner v1 to execute the first recommended experiment from `llm_experiment_plan.json`. `[started]`
- Add IterationOrchestrator v1 for review -> enhancement -> summary loops. `[started]`
- Add best-score memory, patience, and target-score stopping controls. `[started]`
- Route Brain recommendations to multiple tabular enhancement templates: RandomForest, tuned RandomForest, LightGBM, CatBoost, and XGBoost. `[started]`
- Add TabularSearchRunner v1 for multi-model OOF comparison, best/blend/stacking selection, and selected submissions. `[started]`
- Add TabularRiskAuditor v1 for fold stability, ensemble gain, and OOF correlation checks before trusting CV improvements. `[started]`
- Add repeated-CV seed support to TabularSearchRunner for more reliable model and ensemble selection. `[started]`
- Add TabularFeaturePruner v1 for permutation importance and all-vs-pruned feature CV comparison. `[started]`
- Allow TabularSearchRunner to consume pruned feature sets from `feature_report.json`. `[started]`
- Add ExperimentChampionSelector v1 to choose the current risk-adjusted valid submission across experiment and ledger artifacts. `[started]`
- Add SubmissionGate v1 dry-run checks before enabling any Kaggle API submission. `[started]`
- Add Kaggle environment preflight for hard workspace, conda env, CLI, pytest, and credential checks. `[started]`
- Add workspace-scoped remote toolchain setup for `pytest` and Kaggle CLI. `[started]`
- Add workspace-scoped remote Kaggle credential directory preparation and safe validation. `[started]`
- Add lightweight report-only remote artifact sync. `[started]`
- Add KaggleSubmitAdapter v1 dry-run plan with CLI/credential checks and no real submission. `[started]`
- Add confirmed Kaggle submit hard gate requiring explicit flag, submission gate pass, CLI, credentials, and `approve_real_submit` human approval. `[started]`
- Add LeaderboardFeedbackRecorder v1 for manual/API public score, rank, and submission id feedback. `[started]`
- Add leaderboard-aware Remote Brain context and fallback planning for public-vs-CV gaps. `[started]`
- Add LeaderboardGapAuditor v1 for public-vs-CV gap, CV stability, ensemble correlation, and train/test drift checks. `[started]`
- Add StabilityFirstRunner v1 for high-drift feature drops, repeated-CV search, risk audit, and stability review. `[started]`
- Add risk-aware champion reselection using leaderboard-gap penalties and stability-first bonuses. `[started]`
- Add PostReselectionGate v1 to refresh submission gate and Kaggle dry-run plan after champion changes. `[started]`
- Add fast smoke test profile for control-loop verification without heavy model training. `[started]`

### M5: Experiment Memory

- Store experiment configs, scores, artifacts, and leaderboard outcomes. `[started]`
- Query past competitions before planning.
- Track which strategies transfer across task types.

### M6: Submission Gate

- Deterministic validation for every submission. `[started]`
- Shape, schema, score range, duplicate, metric-specific, and leakage checks. `[started]`
- Brain cannot submit until gate passes.

### M7: Domain Tool Expansion

- Tabular tools.
- Vision tools.
- NLP tools.
- Time-series tools.
- Recommender/ranking tools.
- Bio sequence and ontology tools.

## Validation Strategy

Use multiple competitions as framework benchmarks:

- a tabular CSV task for backward compatibility
- a bio sequence multi-label task for non-tabular stress testing
- a vision or NLP task for data-modal diversity

Each benchmark should answer: did the framework correctly identify the task,
choose the right profile, run a valid baseline, and improve through experiment
memory?
