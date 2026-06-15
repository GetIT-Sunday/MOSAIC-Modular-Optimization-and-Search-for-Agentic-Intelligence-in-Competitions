# AutoKaggle 控制台设计文档

## 设计目标

把 AutoKaggle 的页面从“单个 Run Ledger 状态报告”升级为由 `ProjectConsoleAgent` 管理的两级 Kaggle 风格控制台：

```text
AutoKaggle 总控制台
  -> ProjectConsoleAgent 状态管理
  -> 真实 Kaggle 题目池
  -> 全局配置与认证状态
  -> 所有 competition workspace
  -> 全局下一步建议

单竞赛控制台
  -> 当前竞赛 intake
  -> baseline / Brain plan / experiments
  -> submission / feedback / human gate
```

总控制台回答“现在整个 AutoKaggle 项目该做什么”；单竞赛控制台回答“这个竞赛当前进展如何”。

参考 Kaggle 当前公开页面的风格：白底、左侧导航、扁平表格、搜索栏、筛选 chips、蓝色链接、轻量状态标记、少装饰、少厚重卡片。

## 视觉原则

- **Kaggle-like, not enterprise dashboard**：避免重阴影、大面积彩色卡片、奖杯/装饰图标。
- **白底与留白**：页面背景白色，分区用细边框和分隔线，不用深色侧栏。
- **蓝色作为行动色**：链接、主操作按钮、选中 tab 使用 Kaggle 风格蓝色。
- **表格优先**：题目池、workspace、baseline、run ledger 都用表格或紧凑列表。
- **状态 chip 轻量化**：小号圆角标签，例如 `baseline_ready`、`feedback_pending`、`auth_missing`。
- **中文内容，英文协议**：页面文案中文；CLI flag、JSON 字段、状态枚举保持英文。
- **两级导航清晰**：总控制台显示全局信息，单竞赛控制台不重复显示全局题目池。

## 页面结构

### 1. AutoKaggle 总控制台

路径：

```text
multi_agents/competition/index.html
multi_agents/competition/console/pool.html
multi_agents/competition/console/workspaces.html
multi_agents/competition/console/config.html
multi_agents/competition/console/roadmap.html
multi_agents/competition/console/snapshot.json
```

职责：

- 由 `ProjectConsoleAgent` 统一采集状态、推断 workspace stage、决策下一步动作并渲染页面。
- 显示银牌级 Agent 总目标。
- 显示 Kaggle 认证、SSH、远端 workspace、conda env、Brain/Coding LLM 状态。
- 显示真实 Kaggle 题目池。
- 显示已有 competition workspace。
- 给出唯一的全局推荐下一步命令。

当前布局：

```text
左侧导航
  AutoKaggle
  总览 -> index.html
  题目池 -> console/pool.html
  Workspaces -> console/workspaces.html
  配置 -> console/config.html
  路线图 -> console/roadmap.html

主内容
  每个导航项是独立 HTML 页面，不再使用同页锚点跳转。
  总览页只显示摘要、Agent、下一步建议和 workspace preview。
  题目池/Workspaces/配置/路线图分别显示完整信息。
```

核心模块：

| 模块 | 内容 | 数据来源 |
|---|---|---|
| Global Objective | 银牌级 Agent 目标与当前阶段 | `docs/kaggle_agent_silver_roadmap.md` / 静态摘要 |
| Environment Status | Kaggle auth、SSH、workspace、conda、LLM 配置 | CLI preflight / config / remote scripts |
| Kaggle Competition Pool | slug、category、deadline、teams、action | `kaggle_competitions_cache.json` |
| Competition Workspaces | slug、profile、stage、last update、link | `multi_agents/competition/*` |
| Next Best Action | 当前唯一推荐命令 | 聚合 discovery/intake/baseline/roadmap 状态 |
| Agent Snapshot | 控制台生成者、cache status、workspace stage、next action | `console/snapshot.json` |

### 2. 单竞赛控制台

路径：

```text
multi_agents/competition/<competition>/runs/index.html
multi_agents/competition/<competition>/runs/console/data.html
multi_agents/competition/<competition>/runs/console/brain.html
multi_agents/competition/<competition>/runs/console/experiments.html
multi_agents/competition/<competition>/runs/console/submission.html
```

职责：

- 只显示当前竞赛自己的状态。
- 不再重复展示全局 Kaggle 题目池。
- 顶部提供“返回总控制台”链接。
- 保留 Run Ledger 作为页面中的“实验记录”模块。
- 由 `RunLedger` 生成多页面竞赛控制台，导航是真实 HTML 跳转，不再把所有内容挤在单页。

建议布局：

```text
Breadcrumb: 返回 AutoKaggle 总控制台
Header: <competition> 控制台
Nav:
  Overview -> runs/index.html
  Data -> runs/console/data.html
  Brain Plan -> runs/console/brain.html
  Experiments -> runs/console/experiments.html
  Submission -> runs/console/submission.html

Overview
  竞赛 Intake
  当前阶段
  下一步命令

Data
  data_manifest 摘要
  unknown 字段

Baseline
  baseline table
  best baseline

Brain Plan
  recommended experiments
  runner_kind
  promotion gate

Experiments
  Experiment Board 排行榜
  折叠审计日志

Submission
  manual package
  leaderboard feedback
  CV-LB gap
```

核心模块：

| 模块 | 内容 | 数据来源 |
|---|---|---|
| Competition Header | slug、profile、metric、stage | `data_manifest.json`, `competition_intake.json` |
| Intake | files、ID、target、metric、unknown | `competition_intake.json`, `data_manifest.json` |
| Baseline | baseline 状态、CV、best baseline | `baseline_review.json`, `experiments/*` |
| Brain Plan | 推荐实验、runner kind、promotion gate | `llm_experiment_plan.json`, `experiment_queue.json` |
| Experiment Board | rank、task、kind、runner、status、metric、CV、delta、artifact links | `runs/ledger.jsonl`, `scorecard.json` |
| Audit Log | 按 run kind 折叠展示原始 run cards | `runs/ledger.jsonl` |
| Submission Feedback | package、public score、gap、next command | submission/feedback artifacts |

## 状态模型

Workspace stage 建议统一成以下枚举：

```text
not_started
selected
data_pending
intake_ready
baseline_ready
brain_planned
experiment_running
submission_ready
feedback_pending
completed
needs_review
```

全局下一步优先级：

1. Kaggle 未认证：提示认证。
2. 题目池未刷新：提示 `--kaggle-discover`。
3. 题目已推荐但未选择：提示 `--kaggle-select <slug> --kaggle-download`。
4. workspace 有数据但未 intake：提示 `--competition-intake`。
5. intake ready 但未 baseline：提示 `--agent-baseline-start`。
6. baseline ready 但未 Brain plan：提示 `--remote-brain-review`。
7. Brain plan ready 但无队列：提示 `--experiment-queue`。
8. 队列有 runnable：提示 `--run-enhancement`。
9. submission ready：提示 manual package / feedback loop。

## 实施 Plan

### Phase 1: 总控制台 Agent

- 新增 `ProjectConsoleAgent`，保留 `ProjectControlPanel` 作为兼容入口。
- 读取 `kaggle_competitions_cache.json` 和所有 `multi_agents/competition/*` workspace。
- 生成 `multi_agents/competition/index.html` 与 `multi_agents/competition/console/*.html`。
- 总控制台显示摘要；独立页面显示 Kaggle 题目池、workspace 表格、配置、路线图。
- 不读取或显示任何密钥内容。

状态：已实现。

### Phase 2: 单竞赛控制台清理

- 从单竞赛 `runs/index.html` 移除全局 Kaggle 题目池。
- 顶部增加“返回 AutoKaggle 总控制台”链接。
- 将单竞赛页面重构为 Kaggle 风格多页面导航：
  `Overview / Data / Brain Plan / Experiments / Submission`。
- 保留现有 Run Ledger artifact links，不破坏已有测试。

状态：已实现第一版。

### Phase 3: 状态聚合

- 新增 workspace stage 推断函数。
- 对每个 workspace 读取关键文件并推断：
  `selected`、`intake_ready`、`baseline_ready`、`brain_planned`、`feedback_pending` 等。
- 在总控制台中按 stage 排序或突出当前推荐目标。
- 将全局下一步建议写入页面。

### Phase 4: Kaggle 风格视觉细化

- CSS 使用白底、细分隔线、轻量 chip、蓝色链接。
- 控制卡片半径不超过 8px。
- 避免重阴影、渐变、大图标和装饰背景。
- 表格列宽固定，防止长 slug 挤压布局。
- 移动端保持单列可读。

### Phase 5: 下一轮优化

- Experiments 页面已经升级为 `Experiment Board v1`：主视图展示 Top 20 去重实验排行榜，原始 Run Ledger cards 进入折叠审计日志。
- 将 `ProjectConsoleAgent` 的 snapshot 扩展为结构化 JSON API，供未来前端应用或本地服务读取。
- 给总控制台增加“动作队列”：不仅显示唯一下一步，也显示后续 3-5 个候选动作。
- 在 Workspaces 页面加入筛选：stage、profile、是否真实 Kaggle、是否远端同步。
- 在题目池页面加入选择理由：deadline、teams、任务类型、适合 baseline 的程度。
- 在单竞赛控制台继续拆分为真实页面或 tabs：Overview / Data / Baseline / Brain / Experiments / Submission。
- 增加远端健康检查 Agent：Kaggle auth、conda、磁盘、GPU、workspace 写权限、LLM 配置。
- 后续可升级为轻量 Web App，但当前静态 HTML 更容易在本地和远端同步使用。

## 测试计划

- `ProjectConsoleAgent` / `ProjectControlPanel` 能在没有 Kaggle cache 时生成总控制台，并显示刷新题目池命令。
- 总控制台导航生成真实页面链接，不再依赖 `#pool` 这类同页锚点。
- `console/snapshot.json` 标记生成者为 `ProjectConsoleAgent`。
- 有 Kaggle cache 时，总控制台显示真实 competition slug、category、deadline、teams。
- 总控制台列出 `titanic`、`bank_churn` 等已有 workspace，并链接到对应 `runs/index.html`。
- 单竞赛控制台不再重复显示全局 Kaggle 题目池。
- 单竞赛控制台包含返回总控制台链接。
- 单竞赛控制台生成真实页面：
  `runs/console/data.html`、`brain.html`、`experiments.html`、`submission.html`。
- 单竞赛控制台导航不使用 `#data` / `#experiments` 这类同页锚点。
- Experiments 页面主视图是实验排行榜，不再直接展示完整 card 墙。
- 原始 Run Ledger cards 保留在折叠审计日志，保证可追溯。
- workspace stage 推断覆盖：
  `not_started`、`selected`、`intake_ready`、`baseline_ready`、`brain_planned`、`feedback_pending`。
- HTML 中不包含 API key、Kaggle token 或其他密钥值。
- 本地测试通过后同步远端固定 workspace，并做远端 import/compile 检查。

## 验收标准

- 用户打开 `multi_agents/competition/index.html` 即可理解整个项目当前状态。
- 用户打开单竞赛控制台时，不会误以为 Kaggle 题目池属于该竞赛。
- 页面视觉接近 Kaggle：轻量、白底、表格、蓝色链接、chip 状态。
- 页面明确给出下一步命令，减少用户来回问“下一步做什么”。
- 现有 Run Ledger、baseline、Brain review、feedback loop 功能不退化。
