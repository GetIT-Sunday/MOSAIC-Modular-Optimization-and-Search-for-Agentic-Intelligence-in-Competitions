# AutoKaggle 银牌级 Kaggle Agent 计划表

## 目标模式 Objective

将 AutoKaggle 优化为一个通用 Kaggle 多智能体竞赛系统：由用户人工选择真实 Kaggle 题目，系统自动解析竞赛目标、数据、指标和提交格式，在远端 Linux 隔离工作区内跑通 baseline，并由 Brain LLM 持续规划模型选择、特征工程、调参、集成和提交策略，分发给 Coding LLM/Runner 执行，最终在至少一个真实 Kaggle 竞赛中达到银牌及以上水平。

约束：

- 本地 Mac 是控制平面和架构 Brain。
- 远端 Linux 是实际实验执行平面。
- 远端只允许使用硬编码 workspace：
  `/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac`
- 远端项目路径：
  `/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac/workspaces/AutoKaggle`
- 远端 conda env 固定为 `mac`。
- Kaggle、LLM、SSH 等密钥只记录路径或存在性，不写入 HTML、JSON、日志或 git。
- 第一阶段真实提交必须经过 Human Gate，不允许自动误提交。

## 总体流程

```text
人工选择 Kaggle 题目
-> KaggleDiscoveryAgent 获取真实题目池
-> CompetitionIntakeAgent 解析题目、数据、规则、指标、submission 格式
-> Brain Agent 固化 task_card / metric_spec / experiment_plan
-> Coding Agent / Runner 跑通 baseline
-> Validator 校验 submission
-> Run Ledger 中文控制台展示结果
-> Brain LLM 规划下一轮实验
-> Coding LLM/Runner 执行窄任务
-> Promotion / Submission / Human Gate
-> 人工或 API 回填 leaderboard feedback
-> Brain 基于 CV-LB gap 继续优化
```

## 阶段计划表

| 阶段 | 目标 | 当前状态 | 核心产物 | 下一步动作 |
|---|---|---:|---|---|
| 0. 远端隔离 | Mac 控制，远端 Linux 执行，限制 workspace | 已完成 | `scripts/remote_dev.sh`, `scripts/sync_to_dev.sh` | 持续保持硬编码边界 |
| 1. 中文 Run Ledger | 让实验状态、Gate、反馈闭环可人工查看 | 已完成 | `runs/index.html` | 继续扩展为 Agent 控制台 |
| 1.5. Kaggle 风格控制台 | 拆分总控制台与单竞赛控制台，降低全局/单题混淆 | 设计完成 | `docs/autokaggle_console_design.md` | 实现 `multi_agents/competition/index.html` |
| 2. Kaggle 题目池 | 接入 Kaggle 官方 CLI，拉取真实竞赛列表 | 代码完成，认证待配置 | `KaggleDiscoveryAgent`, `kaggle_competitions_cache.json` | 配置远端 Kaggle API 认证 |
| 3. 人工选题 Intake | 用户从真实题目池选择竞赛，生成 intake | 代码入口完成 | `competition_intake.json` | 认证后选择第一个真实竞赛 |
| 4. 竞赛解析 | 解析目标列、ID、metric、submission 格式、unknown 字段 | 部分完成 | `data_manifest.json`, `task_card.md`, `metric_spec.json` | 把 Kaggle metadata/rules 接入解析 |
| 5. Baseline 闭环 | 先跑通 sample/sklearn/GBDT baseline | 已有 runner，需接入新入口 | `baseline_review.json`, `experiments/*` | 对选中竞赛一键 baseline |
| 6. Brain 决策 | Brain LLM 决定模型、调参、特征和任务队列 | 已有 Remote Brain，需产品化 | `llm_experiment_plan.json`, `experiment_queue.json` | 结构化 Brain 输出协议 |
| 7. Coding 执行 | Coding LLM/Runner 执行 Brain 窄任务 | 部分模板化完成 | `run.py`, `validation_report.json`, `submission.csv` | 增加真实 Coding LLM 执行器 |
| 8. 风险与校验 | 校验 submission、泄漏、稳定性、CV-LB gap | 部分完成 | `validator_result.json`, `risk_audit.json` | 按任务类型扩展 validator |
| 9. Leaderboard Feedback | 回填 public score 驱动下一轮 | 手动闭环已完成 | `leaderboard_feedback_loop.json` | 接入 Kaggle submissions/leaderboard |
| 10. 银牌优化循环 | 多轮搜索、集成、gap audit、提交预算管理 | 未系统化 | roadmap + memory + queue | 选择目标竞赛后集中迭代 |

## 当前准确进度

当前处于阶段 2 到阶段 3 之间：

```text
KaggleDiscoveryAgent 已实现
-> 本地测试通过
-> 远端已同步
-> 远端 Kaggle CLI 存在，版本 2.2.0
-> 远端 Kaggle API 未认证
-> 认证后即可刷新真实题目池并进入人工选题
```

已验证：

```text
78 passed
```

远端 Kaggle 当前阻塞：

```text
Authentication required to call the Kaggle API.
```

## 近期执行清单

### A. 配置 Kaggle 认证

推荐在远端配置 Kaggle token 或 OAuth：

```bash
ssh dev
cd /home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac
conda run -n mac kaggle auth login
```

或使用 Kaggle API token，但 token 文件必须放在远端硬 workspace 范围内，不能散落到共享服务器其他目录。

### B. 刷新真实题目池

```bash
python framework.py --kaggle-discover --kaggle-category featured --kaggle-sort-by recentlyCreated
```

可选筛选：

```bash
python framework.py --kaggle-discover --kaggle-category research --kaggle-sort-by latestDeadline
python framework.py --kaggle-discover --kaggle-category playground --kaggle-sort-by recentlyCreated
python framework.py --kaggle-discover --kaggle-search protein
```

### C. 人工选择目标竞赛

```bash
python framework.py --kaggle-select <competition_slug>
```

如已接受规则并允许下载：

```bash
python framework.py --kaggle-select <competition_slug> --kaggle-download
```

### D. 进入 AutoKaggle 闭环

```bash
python framework.py --competition <competition_slug> --task-card-mode
python framework.py --competition <competition_slug> --run-baselines
python framework.py --competition <competition_slug> --remote-brain-review
python framework.py --competition <competition_slug> --experiment-queue
python framework.py --competition <competition_slug> --run-enhancement
```

## 成功标准

短期成功标准：

- 中文控制台能显示真实 Kaggle 题目池。
- 用户能选择一个真实竞赛并生成 `competition_intake.json`。
- 系统能下载或消费该竞赛数据。
- 系统能自动生成 task card、metric spec、data manifest。
- 系统能跑通至少一个合法 baseline submission。
- Run Ledger 能展示代码、日志、分数、validator 结果和下一步命令。

中期成功标准：

- Brain LLM 能基于 baseline 结果规划下一批实验。
- Coding LLM/Runner 能执行 Brain 的窄任务。
- 每轮实验都有可复现脚本、日志、分数、submission、校验报告。
- 系统能基于 leaderboard feedback 分析 CV-LB gap。
- 系统能自动选择 champion，并通过 Human Gate 准备提交包。

最终成功标准：

- 在至少一个真实 Kaggle 竞赛上，通过 AutoKaggle 多轮自动化迭代，取得银牌及以上排名。
- 全过程可追踪、可复现、可人工介入。
- 框架不绑定单一竞赛，能迁移到 tabular、bio sequence、NLP、vision、time series 等不同 profile。

## 关键设计原则

- 先跑通 baseline，再追求高分。
- Brain 负责决策，Coding 负责执行。
- LLM 决策必须结构化落盘，不能只写自然语言建议。
- 每个实验必须产出脚本、日志、指标、submission、validator。
- 每次 leaderboard feedback 都要绑定 submission hash，避免误用旧分数。
- Human Gate 是调试和安全阀，成熟后可以逐步减少人工介入。
- 真实 Kaggle submit 默认禁止自动执行，必须显式批准。
