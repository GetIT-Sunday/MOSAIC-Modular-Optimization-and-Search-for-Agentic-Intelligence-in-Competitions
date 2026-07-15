<a name="mosaic"></a>
<p align="center">
  <img src="assets/banner.png" alt="MOSAIC banner" width="100%">
</p>

<p align="center">
  <h1 align="center">🧩 MOSAIC</h1>
  <p align="center">
    <strong>Modular Optimization and Search for Agentic Intelligence in Competitions</strong><br>
    <em>模块化多智能体竞赛智能优化框架</em>
  </p>
  <p align="center">
    <a href="#-overview">Overview</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#%EF%B8%8F-cli-reference">CLI Reference</a> •
    <a href="#-roadmap">Roadmap</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active%20research-blueviolet?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-yellow?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/agents-Brain%20%2B%20Coding-00d4ff?style=flat-square" alt="Agents">
  <img src="https://img.shields.io/github/stars/GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions?style=social" alt="Stars">
</p>

<p align="center">
  <strong>English</strong> | <a href="README_ZH.md">中文</a>
</p>

---

## 📖 Overview

MOSAIC is a **modular multi-agent framework** for autonomous participation in data science competitions. The core design goal: a general-purpose competition intelligence layer that handles the full lifecycle — from competition intake to experiment planning, remote execution, validation, and leaderboard-feedback-driven optimization.

> **Research in progress.** MOSAIC is actively being developed toward silver-medal-level autonomous performance on real Kaggle competitions.

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Brain–Coding Agent Loop</h3>
      <ul>
        <li>Brain Agent: reads competition overview, selects profile, plans experiment ladder</li>
        <li>Coding Agent: writes scripts, runs experiments, debugs, reports metrics back to Brain</li>
        <li>Narrow task handoff — Brain never writes code; Coding Agent never makes strategy calls</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🗂️ Competition Profile System</h3>
      <ul>
        <li>Profile-driven task identification: <code>tabular_classic</code>, <code>image_classification</code>, <code>nlp_text_classification</code>, <code>time_series_forecasting</code>, and more</li>
        <li>Each profile defines lifecycle phases, metric type, submission format, allowed tool families, and validation checks</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔒 Remote Execution Isolation</h3>
      <ul>
        <li>Local Mac as control plane + Brain</li>
        <li>Remote Linux as isolated experiment executor</li>
        <li>Sync scripts: <code>sync_to_dev.sh</code> / <code>sync_from_dev.sh</code></li>
        <li>Conda environment pinned per workspace</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 Structured Experiment Memory</h3>
      <ul>
        <li>Run Ledger: per-competition HTML control panel</li>
        <li>CV score, leaderboard feedback, and failure modes stored per run</li>
        <li>Brain uses history to plan the next improvement cycle</li>
        <li>Human Gate before any real competition submission</li>
      </ul>
    </td>
  </tr>
</table>

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 🏗️ Architecture

```
framework.py (entry point)
├── CompetitionIntakeAgent    — parse competition overview, data manifest, metric spec, task card
├── KaggleDiscoveryAgent      — fetch real competition pool via Kaggle CLI
├── Brain Agent               — orchestrator: profile selection, experiment ladder, task dispatch
│   └── Coding Agent          — implementation: write → run → debug → report
├── Validator                 — submission format check, CV gate, leaderboard feedback ingestion
├── Run Ledger                — HTML control panel per competition (runs/index.html)
└── Remote Execution Layer
    ├── scripts/sync_to_dev.sh
    ├── scripts/sync_from_dev.sh
    └── scripts/remote_dev.sh
```

**Multi-agent SOP** (`multi_agents/sop.py`) coordinates domain profiles, tools, prompts, memory, and orchestration layers.

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 🚀 Quick Start

**① Environment setup**

```bash
git clone https://github.com/GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions.git
cd MOSAIC-...
conda create -n mosaic python=3.11
conda activate mosaic
pip install -r requirements.txt
```

**② Configure API key**

Create `api_key.txt`:
```
sk-xxx                           # Your LLM API key
https://api.openai.com/v1        # Base URL (or any OpenAI-compatible endpoint)
```

**③ Check configuration**

```bash
python framework.py --config-check
python framework.py --remote-health-check   # if using remote execution
```

**④ Prepare competition data**

```
multi_agents/competition/<competition_name>/
├── train.csv
├── test.csv
├── sample_submission.csv
└── overview.txt    # Competition overview + data description
```

**⑤ Run MOSAIC**

```bash
# Single competition, single run
python framework.py --competition titanic

# Full benchmark (all competitions, 5 runs each)
bash run_multi_agents.sh
```

**⑥ View results**

```bash
python framework.py --project-control-panel
# Opens runs/index.html — the Run Ledger control panel
```

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 🛠️ CLI Reference

<details>
<summary><strong>All framework.py flags — click to expand</strong></summary>
<br>

| Flag | Description |
|------|-------------|
| `--competition NAME` | Target competition name |
| `--model MODEL` | Planner model, or `"config"` to use `multi_agents/config.json` |
| `--task-card-mode` | Run Brain task-card dry loop |
| `--competition-intake` | Parse competition and generate Brain-ready artifacts |
| `--agent-baseline-start` | Intake + baseline + roadmap preparation |
| `--run-baselines` | Run deterministic baseline experiments |
| `--run-enhancement` | Run next uncompleted Brain-recommended experiment |
| `--experiment-queue` | Build visible experiment queue |
| `--experiment-roadmap` | Build prioritized next-action roadmap |
| `--tabular-search` | Multi-model tabular search and blend |
| `--tabular-risk-audit` | Audit CV stability and leaderboard risk |
| `--tabular-leakage-audit` | Audit feature leakage, transform-scope risk, train/test drift |
| `--remote-execution` | Execute on remote Linux via SSH |
| `--config-check` | Check SSH, remote, LLM, and Kaggle config (no secrets printed) |
| `--remote-health-check` | Remote diagnostics: SSH, workspace, conda, disk, GPU, Kaggle |
| `--project-control-panel` | Generate global Run Ledger HTML |

</details>

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 🗺️ Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 0. Remote isolation | Mac control plane + remote Linux executor | ✅ Done |
| 1. Run Ledger | Human-readable experiment state + Gate + feedback loop | ✅ Done |
| 2. Kaggle competition pool | Real competition list via Kaggle CLI | ✅ Code complete |
| 3. Competition intake | Parse objectives, data, metric, submission format | 🔄 In progress |
| 4. Baseline loop | Reproducible sklearn / GBDT baseline pipeline | 🔄 In progress |
| 5. Brain decision loop | Brain LLM plans models, features, task queue | 🔄 In progress |
| 6. Tabular search | Multi-model search, blend, CV-LB gap analysis | 🔄 In progress |
| 7. Silver-medal target | Real Kaggle submission, Human Gate, leaderboard feedback | 🎯 Target |

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 📁 Project Structure

```
MOSAIC/
├── framework.py              # Main entry point
├── multi_agents/
│   ├── sop.py                # Multi-agent SOP coordinator
│   ├── agents/               # Agent implementations
│   ├── domain_profiles/      # Competition profile definitions
│   ├── domain_tools/         # Tool libraries per domain
│   ├── orchestration/        # ProjectControlPanel, Run Ledger
│   ├── prompts/              # LLM prompt templates
│   ├── memory.py             # Experiment memory layer
│   └── skills/               # Reusable ML skill functions
├── docs/                     # Architecture and roadmap docs
├── scripts/
│   ├── sync_to_dev.sh        # Push workspace to remote
│   ├── sync_from_dev.sh      # Pull results from remote
│   └── remote_dev.sh         # SSH remote execution wrapper
├── tests/                    # Test suite
└── requirements.txt
```

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 🤝 Contributing

Research collaborations and contributions welcome — open an issue or PR.

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 📄 License

Apache 2.0 — see [LICENSE.md](LICENSE.md) for details.

---

<p align="center">
  <strong>⭐ If MOSAIC interests you, give it a Star — it helps the research reach more people.</strong>
</p>

<p align="center">
  <a href="https://star-history.com/#GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions&Date">
    <img src="https://api.star-history.com/svg?repos=GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions&type=Date" alt="Star History Chart" width="600">
  </a>
</p>

<p align="center">
  <sub>Made with ✨ by <a href="https://github.com/GetIT-Sunday">GetIT-Sunday</a> using <a href="https://github.com/GetIT-Sunday/ReadmeMagic-github-readme-design-skill">ReadmeMagic</a></sub>
</p>
