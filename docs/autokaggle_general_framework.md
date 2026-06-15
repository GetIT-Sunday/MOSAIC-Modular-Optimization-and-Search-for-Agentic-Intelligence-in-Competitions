# AutoKaggle General Framework

## Goal

Refactor AutoKaggle into a general multi-agent competition framework.

The framework should handle the full competition lifecycle:

1. identify the competition task type
2. download or ingest competition data
3. inspect data and rules
4. choose an execution profile
5. plan experiments
6. delegate implementation to a coding agent
7. run experiments on the configured worker
8. validate submissions
9. record leaderboard and experiment history
10. use history to improve the next run

The original tabular AutoKaggle pipeline becomes one profile, not the whole
system.

## Core Abstractions

### Competition Profile

A profile describes a competition family:

- expected input files
- task type
- lifecycle phases
- metric type
- submission format
- allowed tool families
- validation checks
- recommended baseline ladder

Examples:

- `tabular_classic`
- `image_classification`
- `nlp_text_classification`
- `bio_sequence_multilabel`
- `time_series_forecasting`
- `recommender_ranking`

### Brain Agent

The Brain is the top-level orchestrator. It does not directly implement
experiments. Its responsibilities are:

- read competition overview, rules, files, and sample submission
- select or propose a competition profile
- audit whether data and metric assumptions are correct
- design the experiment ladder
- decide when results are good enough to submit
- collect historical run outcomes
- hand precise implementation tasks to the Coding Agent

### Coding Agent

The Coding Agent owns implementation:

- write scripts
- run scripts
- debug failures
- produce artifacts
- report metrics and outputs back to the Brain

The Coding Agent should receive narrow, executable tasks from the Brain rather
than vague competition goals.

### Competition Memory

The memory layer stores:

- competition metadata
- profile decisions
- experiment configurations
- validation scores
- leaderboard submissions
- failure modes
- successful patterns from historical competitions

This memory should be queryable before new planning rounds.

## Target Control Loop

```text
Competition Ingest
  -> Brain: identify task and profile
  -> Brain: build task card and metric spec
  -> Coding Agent: data audit script
  -> Brain: review audit and choose baseline ladder
  -> Coding Agent: implement baseline experiment
  -> Validator: check metrics and submission format
  -> Brain: decide next experiment or submission
  -> Memory: persist run, score, artifacts, lessons
```

## Near-Term Refactor

1. Neutral profile files live in `multi_agents/domain_profiles/`.
2. Profile loading and task identification live in `multi_agents/orchestration/`.
3. The deterministic Brain skeleton can now produce Coding Agent task cards.
4. Competition memory stores JSONL experiment and leaderboard records.
5. Existing tabular SOP remains the first execution backend.
6. Bio/ontology utilities live under neutral domain tools without naming a
   specific competition.

## Current Code Skeleton

- `profile.py`: loads competition profiles and scores file matches.
- `task_identifier.py`: maps competition signals to a profile decision.
- `brain.py`: builds signals, selects profiles, and plans initial Coding Agent tasks.
- `coding_task.py`: defines the narrow executable task contract.
- `memory.py`: records experiments, scores, ranks, artifacts, and summaries.
- `ingestion.py`: builds `data_manifest.json` from local competition files.
- `validator.py`: checks submission shape, IDs, missing predictions, and basic label/range issues.
- `run_ledger.py`: records per-step audit bundles and writes a static HTML dashboard.
- `baseline_runner.py`: generates and runs deterministic tabular baselines, then records logs, reports, submissions, and validator outputs in the ledger.
- `enhancement_runner.py`: executes Remote Brain recommendations through tabular experiment templates. It now routes RandomForest, tuned RandomForest, LightGBM, CatBoost, and XGBoost requests into separate runnable scripts.
- `tabular_search_runner.py`: runs a compact multi-model tabular search with OOF tracking, compares best single/blend/stacking candidates, and writes the selected submission.
- `tabular_risk_auditor.py`: audits tabular search stability using fold variance, OOF model correlations, ensemble gain, and risk-level recommendations.
- `tabular_feature_pruner.py`: estimates original-column permutation importance and compares all-feature vs pruned-feature CV before writing a selected submission.
- `tabular_feature_leakage_auditor.py`: audits target leakage indicators, train/test drift, engineered-feature drift, and transform-scope risks such as rarity bucketing fit separately on train/test.
- `champion_selector.py`: scans experiment and Run Ledger artifacts, applies validator/risk gates, and writes the current `champion_submission.csv`.
- `submission_gate.py`: performs the final dry-run gate before submission by checking champion selection, submission schema, risk, and human gate state.
- `kaggle_submit_adapter.py`: checks the remote Kaggle environment, builds a Kaggle CLI dry-run submit plan, checks CLI and credential presence without printing secrets, and keeps real submission behind explicit confirmation plus human approval gates.
- `leaderboard_feedback.py`: records manual or API leaderboard feedback so future Brain reviews can compare local CV, public score, rank, and submission history.
- `leaderboard_gap_auditor.py`: audits public-vs-CV gaps, CV stability signals, and train/test drift before additional leaderboard submissions.
- `stability_first_runner.py`: drops high-drift raw/derived features, reruns repeated-CV tabular search, audits risk, and writes a stability-first review.
- `post_reselection_gate.py`: refreshes the submission gate and Kaggle dry-run plan after champion reselection.

## Current MVP Entry

The local machine is the project-control Brain. Experiment execution defaults to
the remote Linux workspace:

```text
/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac/workspaces/AutoKaggle
```

Run a deterministic remote Brain dry loop without invoking LLM agents:

```bash
python3 framework.py --competition titanic --task-card-mode
```

This writes `task_card.md`, `metric_spec.json`, `data_manifest.json`,
`experiment_plan.json`, `brain_review.json`, and per-task CodingAgent prompts
under `experiments/<task_id>/coding_prompt.md`.

It also writes a human-reviewable run ledger:

```text
runs/
  ledger.jsonl
  index.html
  0001_brain_plan/
    input.json
    prompt.md
    scorecard.json
    human_review.md
    artifacts/
  0002_sample_submission_validation/
    input.json
    prompt.md
    scorecard.json
    human_review.md
    artifacts/
```

Run deterministic tabular baselines on remote Linux and sync the produced
artifacts back for local review:

```bash
python3 framework.py --competition titanic --run-baselines
```

This writes baseline artifacts under `experiments/<baseline_name>/` and appends
one ledger card per baseline plus a best-baseline review card.

Use `--execution-backend local` only for tests or emergency debugging. Real
experiments should stay on the remote Linux backend.

Run the remote project Brain review over the latest experiment artifacts:

```bash
python3 framework.py --competition titanic --remote-brain-review
```

Human gates are transparent by default. A `human_review.md` with
`decision: continue` does not alter automation. During debugging, set
`decision: patch_prompt`, `rerun`, or `stop` and add notes; the remote Brain
will include those notes in `llm_experiment_plan.md/json`.

When `leaderboard_feedback.json` exists, the remote Brain also includes public
score, rank, submission id, champion selection, risk audits, and leaderboard
memory summary in its planning context. If public score is materially worse
than local CV, the deterministic fallback Brain recommends a
`leaderboard_gap_audit_v1` task before further leaderboard chasing.

Run the next uncompleted Remote Brain recommended enhancement experiment on
remote Linux:

```bash
python3 framework.py --competition titanic --run-enhancement
```

This reads `llm_experiment_plan.json`, skips already completed recommendations,
executes the next actionable experiment, writes `enhancement_review.json`, and
appends enhancement run cards to the Experiment Control Panel.

For tabular competitions, the first enhancement executor is still deliberately
template-based, but it is no longer a single-model runner. The Brain can ask for
RandomForest, tuned RandomForest, LightGBM, CatBoost, or XGBoost in the
experiment title/task text, and the runner will select the matching script.
This gives us a controlled bridge between LLM planning and real model search
before replacing templates with a fuller CodingAgent implementation.

Run one or more automatic optimization iterations:

```bash
python3 framework.py --competition titanic --iterate --max-iterations 1
```

Each iteration performs Remote Brain review, runs the next unexecuted
recommendation, writes `optimization_loop_summary.json`, and syncs the updated
Experiment Control Panel back to the local machine.

The loop records `best_score.json` and supports conservative stopping controls:

```bash
python3 framework.py --competition titanic --iterate --max-iterations 3 --patience 2 --target-score 0.84
```

`patience` stops after consecutive non-improving iterations. `target-score`
stops once the current best score reaches the requested threshold.

Run a stronger tabular search layer:

```bash
python3 framework.py --competition titanic --tabular-search
```

This executes a compact model sweep on remote Linux: linear/ridge,
RandomForest, ExtraTrees, HistGradientBoosting, and optional LightGBM, XGBoost,
and CatBoost when available. It records fold scores, `model_report.json`,
`oof_predictions.csv`, `ensemble_report.json`, the selected `submission.csv`,
validator output, and a Run Ledger card. The selected submission is chosen by
OOF comparison across best single model, top-model mean blend, and a simple
logistic/ridge stacking candidate when compatible. This is the first L3
building block for moving from "valid experiments" toward "competitive tabular
optimization."

Use repeated CV seeds when a result needs a stronger stability check:

```bash
python3 framework.py --competition titanic --tabular-search --tabular-search-seeds 42,123,777
```

The runner averages OOF/test predictions across the requested seed folds and
records per-seed scores in `model_report.json`.

After running feature pruning, the search runner can consume the selected
pruned feature set:

```bash
python3 framework.py --competition titanic --tabular-search --tabular-feature-set pruned
```

This reads `experiments/tabular_feature_prune_v1/feature_report.json` and uses
its `kept_features` list when those features are available after shared feature
engineering. If the feature report is missing or unusable, the runner records a
warning and falls back to all features.

After running the leakage auditor, the search runner can consume risky feature
drops:

```bash
python3 framework.py --competition titanic --tabular-search \
  --tabular-feature-set leakage_safe \
  --tabular-task-id leakage_safe_search_v1
```

This reads `tabular_feature_leakage_audit.json` and drops its
`recommended_drop_features` before model search. It keeps the result in a
separate experiment directory so it can be compared against the original,
pruned, and stability-first candidates.

Audit whether the latest tabular search improvement is trustworthy:

```bash
python3 framework.py --competition titanic --tabular-risk-audit
```

This reads `validation_report.json`, `model_report.json`,
`ensemble_report.json`, and `oof_predictions.csv`, then writes
`risk_audit.json` plus a Run Ledger card. The audit highlights high fold
variance, small ensemble gains, highly correlated ensemble members, and missing
OOF evidence. This is a guardrail before allowing the Brain to treat a CV gain
as a reliable next-step signal.

Run feature importance and pruning comparison:

```bash
python3 framework.py --competition titanic --tabular-feature-prune
```

This computes permutation importance on original tabular columns, writes
`feature_report.json`, compares all-feature vs pruned-feature 5-fold CV, and
writes a selected `submission.csv`. It is an early feature-control layer before
heavier feature generation or target/frequency encoding.

Audit feature leakage and train/test drift before another leaderboard-driven
round:

```bash
python3 framework.py --competition titanic --tabular-leakage-audit
```

This writes `tabular_feature_leakage_audit.json` and
`experiments/tabular_feature_leakage_audit_v1/leakage_report.json`. It checks
raw train/test drift, common engineered-feature drift, target-like columns,
train/test ID overlap, and transform-scope risks. For Titanic, it can flag
`Title` rarity bucketing when rarity is computed separately on train and test,
then recommend a stable/pruned search that drops or patches the risky feature.

Select the current champion submission:

```bash
python3 framework.py --competition titanic --select-champion
```

This scans current experiment outputs and historical Run Ledger artifacts,
requires a valid `validator_result.json`, applies a simple risk penalty from
available `risk_audit.json` files, writes `champion_selection.json`, and copies
the selected submission to `champion_submission.csv`.

Champion selection is also leaderboard-aware. When `leaderboard_gap_audit.json`
shows a high-risk public-vs-CV gap for the previous champion, the selector adds
a contextual penalty to that original champion. If a validated
`stability_first_search_v1` candidate has low/medium risk, it receives a small
stability bonus. This lets the framework prefer a slightly lower-CV but more
stable candidate when public feedback shows the old champion may be overfit.

Run the final dry-run submission gate:

```bash
python3 framework.py --competition titanic --submission-gate
```

This does not submit to Kaggle. It validates `champion_submission.csv`, checks
`champion_selection.json`, risk level, competition metadata, and the latest
champion-selection human gate, then writes `submission_gate.json` and a Run
Ledger card.

After risk-aware champion reselection, refresh both the final gate and dry-run
submit plan:

```bash
python3 framework.py --competition titanic --post-reselection-gate
```

This does not submit to Kaggle. It reruns `submission_gate`, rebuilds
`kaggle_submit_plan.json`, checks that the refreshed gate references the current
champion, and writes `post_reselection_gate.json`.

Check the remote Kaggle environment before wiring credentials or submit tools:

```bash
python3 framework.py --competition titanic --kaggle-env-preflight
```

This does not submit to Kaggle. It writes `kaggle_env_preflight.json` and a Run
Ledger card with:

- hard remote workspace check
- active Python and conda environment
- Kaggle CLI availability
- `pytest` availability for remote verification
- credential presence in allowed locations, without printing secret values
- credential file permission warnings

If the preflight reports missing `pytest` or Kaggle CLI, install the minimal
remote toolchain inside the hard workspace-scoped `mac` conda environment:

```bash
scripts/setup_remote_toolchain.sh
```

The script sets `CONDA_ENVS_PATH`, `CONDA_PKGS_DIRS`, and `PIP_CACHE_DIR` under
the hard remote workspace before installing `pytest` and `kaggle`.

Build a Kaggle submit dry-run plan:

```bash
python3 framework.py --competition titanic --kaggle-submit-dry-run
```

This still does not submit to Kaggle. It requires `submission_gate.json` to
pass, checks whether the Kaggle CLI and credentials are present in allowed
locations, writes `kaggle_submit_plan.json`, and records the exact command that
would be used later after explicit human approval.

The real-submit code path is intentionally locked behind multiple gates:

```bash
python3 framework.py --competition titanic --kaggle-submit-confirmed
```

It will not run unless `submission_gate.json` passes, Kaggle CLI and credentials
are available, the flag above is present, and the latest
`kaggle_submit_plan` human review contains `approve_real_submit` in `notes`.
When any condition is missing, it writes `kaggle_submit_result.json` with a
blocked status instead of submitting.

For real credentials, keep them inside the competition workspace boundary. The
adapter checks environment variables or:

```text
multi_agents/competition/<competition>/.kaggle/kaggle.json
```

Prepare the remote directory and validate file permissions without printing
secret values:

```bash
scripts/prepare_remote_kaggle_credentials.sh titanic
```

For Titanic, the remote credential file is:

```text
/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac/workspaces/AutoKaggle/multi_agents/competition/titanic/.kaggle/kaggle.json
```

Paste Kaggle's API JSON into that file on the remote host. The helper will set
the directory to `700`, the file to `600`, and verify that `username` and `key`
exist without printing either value. The sync scripts intentionally exclude
`.kaggle/`, so credentials stay on the remote Linux workspace.

When only report files and the Run Ledger index are needed, use the lightweight
sync mode:

```bash
scripts/sync_from_dev.sh titanic --lite
```

Use the default full sync when experiment folders or run artifacts are needed.

Use the fast smoke suite for ordinary framework edits:

```bash
scripts/run_smoke_tests.sh
```

The smoke suite avoids heavy model training. It checks the control loop:
ingestion, submission validation, fake-artifact champion selection, submission
gate, Kaggle dry-run plan, and post-reselection gate refresh. Run the full
orchestration tests before larger runner changes or before trusting remote
experiment behavior:

```bash
python3 -m pytest -q tests/test_orchestration.py tests/test_ontology_multilabel_metrics.py
```

After a manual or API Kaggle submission, record the public feedback and close
the post-submit loop in one command:

```bash
python3 framework.py --competition titanic --leaderboard-feedback-loop \
  --public-score 0.81234 \
  --leaderboard-rank 1234 \
  --submission-id <kaggle-submission-id> \
  --feedback-source manual
```

This writes `leaderboard_feedback.json`, immediately refreshes
`leaderboard_gap_audit.json`, asks the Remote Brain for the next experiment
plan, writes `leaderboard_feedback_loop.json`, and appends Run Ledger cards for
each stage. Use this command after real public feedback so the next optimization
round is driven by leaderboard evidence, not only local CV.

To record feedback without running the full loop, use:

```bash
python3 framework.py --competition titanic --leaderboard-feedback \
  --public-score 0.81234 \
  --leaderboard-rank 1234 \
  --submission-id <kaggle-submission-id> \
  --feedback-source manual
```

This writes `leaderboard_feedback.json`, appends a Run Ledger card, and stores
the public score/rank in competition memory. The next Brain review can then
compare local CV, public leaderboard behavior, and risk-audit signals instead
of optimizing only against local validation.

Audit leaderboard gap and stability:

```bash
python3 framework.py --competition titanic --leaderboard-gap-audit
```

This writes `leaderboard_gap_audit.json`, appends a Run Ledger card, and
combines public-vs-local score gap, champion risk audit, fold variance,
ensemble correlation, and train/test feature drift signals. If public score is
materially worse than local CV, the audit recommends pausing leaderboard
submissions and running stability-first experiments.

Run a stability-first search after a high-risk leaderboard gap:

```bash
python3 framework.py --competition titanic --stability-first
```

This writes `experiments/stability_first_features_v1/feature_report.json`,
runs `experiments/stability_first_search_v1/` with repeated CV and stable
feature drops, creates a risk audit for that search, and writes
`stability_first_review.json`. For Titanic, high-drift raw features such as
`Name`, `Ticket`, and `Cabin` map to derived features like `Title`,
`TicketPrefix`, `HasCabin`, and `CabinDeck`.
