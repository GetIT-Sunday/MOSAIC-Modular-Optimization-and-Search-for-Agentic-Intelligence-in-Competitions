<a name="mosaic"></a>
<p align="center">
  <img src="assets/banner.png" alt="MOSAIC banner" width="100%">
</p>

<p align="center">
  <h1 align="center">🧩 MOSAIC</h1>
  <p align="center">
    <strong>模块化多智能体竞赛智能优化框架</strong><br>
    <em>Modular Optimization and Search for Agentic Intelligence in Competitions</em>
  </p>
  <p align="center">
    <a href="#-项目概述">项目概述</a> •
    <a href="#-架构设计">架构设计</a> •
    <a href="#-快速开始">快速开始</a> •
    <a href="#%EF%B8%8F-cli-参考">CLI 参考</a> •
    <a href="#-路线图">路线图</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/状态-研究进行中-blueviolet?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-yellow?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/智能体-Brain%20%2B%20Coding-00d4ff?style=flat-square" alt="Agents">
  <img src="https://img.shields.io/github/stars/GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions?style=social" alt="Stars">
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

---

## 📖 项目概述

MOSAIC 是一个**模块化多智能体框架**，用于自主参与数据科学竞赛。核心设计目标：构建一个通用竞赛智能层，覆盖完整生命周期——从竞赛解析、实验规划、远端隔离执行、提交验证，到基于排行榜反馈的持续优化。

> **研究进行中。** MOSAIC 正在积极开发，目标是在真实 Kaggle 竞赛中实现银牌级别的自主竞赛能力。

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Brain–Coding 智能体环路</h3>
      <ul>
        <li>Brain Agent：读取竞赛概述、选择 Profile、规划实验梯队</li>
        <li>Coding Agent：编写脚本、执行实验、调试、将指标回报 Brain</li>
        <li>严格职责分离——Brain 不写代码，Coding Agent 不做策略决策</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🗂️ 竞赛 Profile 系统</h3>
      <ul>
        <li>Profile 驱动任务识别：<code>tabular_classic</code>、<code>image_classification</code>、<code>nlp_text_classification</code>、<code>time_series_forecasting</code> 等</li>
        <li>每个 Profile 定义生命周期阶段、评估指标类型、提交格式、允许工具族和验证规则</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔒 远端执行隔离</h3>
      <ul>
        <li>本地 Mac 作为控制平面 + Brain</li>
        <li>远端 Linux 作为隔离实验执行环境</li>
        <li>同步脚本：<code>sync_to_dev.sh</code> / <code>sync_from_dev.sh</code></li>
        <li>Conda 环境按 workspace 固定</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 结构化实验记忆</h3>
      <ul>
        <li>Run Ledger：每个竞赛独立 HTML 控制台</li>
        <li>每轮存储 CV 分数、排行榜反馈和失败模式</li>
        <li>Brain 基于历史规划下一轮优化周期</li>
        <li>真实竞赛提交前强制 Human Gate</li>
      </ul>
    </td>
  </tr>
</table>

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 🏗️ 架构设计

```
framework.py（入口）
├── CompetitionIntakeAgent    — 解析竞赛概述、数据清单、评估指标、任务卡
├── KaggleDiscoveryAgent      — 通过 Kaggle CLI 获取真实竞赛池
├── Brain Agent               — 编排者：Profile 选择、实验梯队、任务分发
│   └── Coding Agent          — 实现层：编写 → 执行 → 调试 → 回报指标
├── Validator                 — 提交格式校验、CV 门控、排行榜反馈摄入
├── Run Ledger                — 每竞赛 HTML 控制台（runs/index.html）
└── 远端执行层
    ├── scripts/sync_to_dev.sh
    ├── scripts/sync_from_dev.sh
    └── scripts/remote_dev.sh
```

**多智能体 SOP**（`multi_agents/sop.py`）协调 Domain Profile、工具库、提示词、记忆层和编排层。

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 🚀 快速开始

**① 环境配置**

```bash
git clone https://github.com/GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions.git
cd MOSAIC-...
conda create -n mosaic python=3.11
conda activate mosaic
pip install -r requirements.txt
```

**② 配置 API Key**

创建 `api_key.txt`：
```
sk-xxx                           # 你的 LLM API Key
https://api.openai.com/v1        # Base URL（支持任何 OpenAI 兼容接口）
```

**③ 检查配置**

```bash
python framework.py --config-check
python framework.py --remote-health-check   # 使用远端执行时
```

**④ 准备竞赛数据**

```
multi_agents/competition/<竞赛名>/
├── train.csv
├── test.csv
├── sample_submission.csv
└── overview.txt    # 竞赛概述 + 数据说明
```

**⑤ 运行 MOSAIC**

```bash
# 单竞赛单次运行
python framework.py --competition titanic

# 完整基准测试（所有竞赛，各 5 轮）
bash run_multi_agents.sh
```

**⑥ 查看结果**

```bash
python framework.py --project-control-panel
# 打开 runs/index.html — Run Ledger 控制台
```

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 🛠️ CLI 参考

<details>
<summary><strong>所有 framework.py 参数 — 点击展开</strong></summary>
<br>

| 参数 | 说明 |
|------|------|
| `--competition NAME` | 目标竞赛名称 |
| `--model MODEL` | Planner 模型，或 `"config"` 使用 `multi_agents/config.json` |
| `--task-card-mode` | 运行 Brain 任务卡 dry loop |
| `--competition-intake` | 解析竞赛并生成 Brain 所需工件 |
| `--agent-baseline-start` | Intake + baseline + 路线图准备 |
| `--run-baselines` | 运行确定性 baseline 实验 |
| `--run-enhancement` | 执行下一个未完成的 Brain 推荐增强实验 |
| `--experiment-queue` | 构建可视实验队列 |
| `--experiment-roadmap` | 构建优先级次动作路线图 |
| `--tabular-search` | 多模型表格搜索与融合 |
| `--tabular-risk-audit` | 审计 CV 稳定性和排行榜风险 |
| `--tabular-leakage-audit` | 审计特征泄漏、变换范围风险、训练/测试分布漂移 |
| `--remote-execution` | 通过 SSH 在远端 Linux 执行 |
| `--config-check` | 检查 SSH/远端/LLM/Kaggle 配置（不打印密钥） |
| `--remote-health-check` | 远端诊断：SSH、workspace、conda、磁盘、GPU、Kaggle |
| `--project-control-panel` | 生成全局 Run Ledger HTML |

</details>

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 🗺️ 路线图

| 阶段 | 目标 | 状态 |
|------|------|------|
| 0. 远端隔离 | Mac 控制平面 + 远端 Linux 执行 | ✅ 已完成 |
| 1. Run Ledger | 人类可读的实验状态 + Gate + 反馈闭环 | ✅ 已完成 |
| 2. Kaggle 竞赛池 | 通过 Kaggle CLI 获取真实竞赛列表 | ✅ 代码完成 |
| 3. 竞赛解析 | 解析目标、数据、评估指标、提交格式 | 🔄 进行中 |
| 4. Baseline 闭环 | 可复现的 sklearn / GBDT baseline 流水线 | 🔄 进行中 |
| 5. Brain 决策环路 | Brain LLM 规划模型、特征、任务队列 | 🔄 进行中 |
| 6. 表格搜索 | 多模型搜索、融合、CV-LB gap 分析 | 🔄 进行中 |
| 7. 银牌目标 | 真实 Kaggle 提交、Human Gate、排行榜反馈 | 🎯 目标 |

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 📁 项目结构

```
MOSAIC/
├── framework.py              # 主入口
├── multi_agents/
│   ├── sop.py                # 多智能体 SOP 协调器
│   ├── agents/               # 智能体实现
│   ├── domain_profiles/      # 竞赛 Profile 定义
│   ├── domain_tools/         # 各领域工具库
│   ├── orchestration/        # ProjectControlPanel、Run Ledger
│   ├── prompts/              # LLM 提示词模板
│   ├── memory.py             # 实验记忆层
│   └── skills/               # 可复用 ML 技能函数
├── docs/                     # 架构文档和路线图
├── scripts/
│   ├── sync_to_dev.sh        # 推送 workspace 到远端
│   ├── sync_from_dev.sh      # 从远端拉取结果
│   └── remote_dev.sh         # SSH 远端执行封装
├── tests/                    # 测试套件
└── requirements.txt
```

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 🤝 贡献

欢迎研究合作与贡献——提 Issue 或 PR 即可。

<div align="right"><a href="#mosaic">↑ 返回顶部</a></div>

---

## 📄 许可证

Apache 2.0 — 详见 [LICENSE.md](LICENSE.md)

---

<p align="center">
  <strong>⭐ 如果 MOSAIC 对你有价值，请给一个 Star — 让更多人关注这个研究！</strong>
</p>

<p align="center">
  <a href="https://star-history.com/#GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions&Date">
    <img src="https://api.star-history.com/svg?repos=GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions&type=Date" alt="Star History Chart" width="600">
  </a>
</p>

<p align="center">
  <sub>Made with ✨ by <a href="https://github.com/GetIT-Sunday">GetIT-Sunday</a> using <a href="https://github.com/GetIT-Sunday/ReadmeMagic-github-readme-design-skill">ReadmeMagic</a></sub>
</p>
