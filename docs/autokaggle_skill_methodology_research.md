# AutoKaggle Skill Methodology Research

## Why This Matters

AutoKaggle should not rely on ad-hoc prompts such as "try LightGBM" or "add an MLP".
For silver-or-better Kaggle performance, the Brain must reason through reusable
competition methodology:

- what objective gap matters: leaderboard target, medal line, public/private risk
- why a method could improve score
- which validation harness can falsify the idea
- which implementation harness can execute it reproducibly
- when a result is strong enough to promote

This document defines the first writing standard for AutoKaggle skills.

## External Findings

Public descriptions of agent skills converge on the same shape: a skill is a
folder of targeted instructions, scripts, and resources that an agent loads when
the task context requires it. This makes skills closer to domain onboarding and
workflow memory than a single tool call.

Recent research also highlights two important design pressures:

- skills are operational text, so metadata and instructions influence discovery,
  selection, and safety;
- successful skill ecosystems are broad, repetitive, and workflow-heavy, but
  skill length has to stay within practical prompt budgets.

Sources reviewed:

- Anthropic Skills coverage: skills are folders containing instructions, scripts,
  and resources, usable across assistant, code, API, and SDK workflows.
- `Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large
  Language Model Functionality` reports a large public ecosystem with heavy
  adoption in software engineering, information retrieval, and content creation.
- `Under the Hood of SKILL.md` warns that SKILL.md metadata and wording can
  affect discovery, selection, and governance.
- `EvoSkills` frames skills as multi-file packages that need generation and
  verification rather than isolated prompts.

## Local Skill Corpus Statistics

Corpus:

- project-local ARIS/Codex skills under `.agents/skills` and `.aris/upstream`
- user/system Codex skills under `~/.codex/skills`
- total sampled `SKILL.md` files: 84

Measured word counts:

| Statistic | Words |
| --- | ---: |
| min | 219 |
| p25 | 891 |
| median | 1436 |
| mean | 1803 |
| p75 | 2077 |
| p90 | 3453 |
| max | 6357 |

Distribution:

| Range | Count |
| --- | ---: |
| 0-500 | 3 |
| 500-1000 | 23 |
| 1000-2000 | 36 |
| 2000-3500 | 13 |
| 3500+ | 9 |

Structural observations:

- 84/84 use YAML frontmatter.
- 68/84 include an explicit workflow section.
- Mature skills usually start with trigger metadata, then a compact procedure,
  constants, validation rules, and output format.
- Larger skills move details into scripts or references when the workflow is too
  broad.

## Recommended AutoKaggle Skill Size

For AutoKaggle, use this budget:

- normal methodology skill: 900-1800 words
- complex orchestration skill: 1800-2600 words
- avoid exceeding 3000 words unless the skill has no alternative reference split
- move detailed model recipes, paper notes, and examples into `references/`
- move deterministic execution into `scripts/` or project harness modules

This keeps the Brain focused. The skill should teach the agent how to think and
decide; the harness should do the repetitive execution.

## Skill vs Harness Boundary

Skill:

- chooses a strategy
- explains why it may improve leaderboard performance
- identifies risks
- defines validation gates
- names the harness to use

Harness:

- loads data
- constructs folds
- trains models
- writes OOF predictions
- writes submissions
- computes metrics
- emits reproducible reports

Example:

```json
{
  "skill": "tabular_optimization",
  "hypothesis": "GBDT models may capture redshift, color-band, and spectral-type interactions missed by the linear baseline.",
  "harness": "stratified_gbdt_oof_harness",
  "expected_gain": "+0.002 to +0.008 accuracy",
  "risk": "public/private drift or overfitting to strong categorical proxies",
  "promotion_gate": "5-fold mean accuracy improves and fold std does not increase materially"
}
```

## Proposed AutoKaggle Skill Layout

```text
multi_agents/skills/
  leaderboard_target/
    SKILL.md
    references/medal_estimation.md
  tabular_optimization/
    SKILL.md
    references/gbdt.md
    references/tabular_nn.md
    references/feature_engineering.md
  validation_risk/
    SKILL.md
    references/leakage.md
    references/public_private_gap.md
  ensemble_strategy/
    SKILL.md
    references/oof_blending.md
```

Each `SKILL.md` should contain:

1. frontmatter with precise trigger description
2. when to use / when not to use
3. required context files
4. decision workflow
5. hypothesis template
6. harness selection rules
7. promotion gate rules
8. required output JSON schema

## First Four Skills

### `leaderboard_target`

Purpose:

- fetch or ingest Kaggle leaderboard signals
- estimate top score, top-N score, silver-line proxy, and gap to target
- convert "high baseline" into "distance to leaderboard objective"

Output:

```json
{
  "leaderboard_available": true,
  "metric_name": "accuracy",
  "top_score": 0.98,
  "top_10_score": 0.97,
  "estimated_silver_score": 0.95,
  "current_best_local_score": 0.927,
  "gap_to_silver": 0.023,
  "gap_to_top": 0.053,
  "target_policy": "silver_or_better"
}
```

### `tabular_optimization`

Purpose:

- choose model families and feature ideas for tabular competitions
- prioritize GBDT, categorical handling, feature interactions, and only then
  neural/tabular foundation routes when justified

Key model families:

- Logistic / linear baseline
- RandomForest / ExtraTrees
- LightGBM / XGBoost / CatBoost
- residual MLP with categorical embeddings
- FT-Transformer / TabPFN when data shape supports it
- OOF blending / stacking

### `validation_risk`

Purpose:

- prevent local score chasing
- design folds, drift checks, leakage checks, and public/private gap response

Required checks:

- 5-fold or repeated stratified CV
- OOF confusion matrix
- fold variance
- train/test drift summary
- suspicious feature audit
- metric mismatch check

### `ensemble_strategy`

Purpose:

- compare candidate OOF predictions
- blend only when errors are complementary
- avoid leaderboard-only hill climbing unless Human Gate explicitly allows it

Required checks:

- OOF score per candidate
- pairwise prediction correlation
- blend weights
- stability under folds
- submission validator pass

## Brain Planning Contract

After these skills exist, Brain tasks should no longer be plain instructions.
Each task must include:

```json
{
  "task_id": "gbdt_interaction_search_v1",
  "skill_used": "tabular_optimization",
  "harness": "stratified_gbdt_oof_harness",
  "hypothesis": "What relationship or failure mode this tests.",
  "model_family": "catboost",
  "expected_gain": "+0.002 to +0.006 accuracy",
  "risk": "What could make the gain fake or non-generalizing.",
  "validation_plan": "5-fold stratified CV with OOF confusion matrix.",
  "promotion_gate": {
    "metric": "accuracy",
    "min_cv_gain": 0.002,
    "max_fold_std_increase": 0.001,
    "validator_required": true
  }
}
```

## Next Implementation Plan

1. Add `multi_agents/skills/` with the four v1 skills above.
2. Add `SkillRegistry` to load skill metadata and bodies.
3. Add `HarnessRegistry` with named harness capabilities.
4. Add `LeaderboardTargetAgent` and `leaderboard_target.json`.
5. Modify Remote Brain context so it reads:
   - `leaderboard_target.json`
   - skill registry
   - available harnesses
   - baseline review
6. Modify experiment queue so every task has:
   - `skill_used`
   - `harness`
   - `hypothesis`
   - `expected_gain`
   - `risk`
   - `promotion_gate`
