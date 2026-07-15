<a name="mosaic"></a>
<p align="center">
  <img src="assets/banner.png" alt="MOSAIC banner" width="100%">
</p>

<p align="center">
  <h1 align="center">🧩 MOSAIC</h1>
  <p align="center">
    <strong>Modular Optimization and Search for Agentic Intelligence in Competitions</strong><br>
    <em>模块化多智能体数据科学竞赛框架</em>
  </p>
  <p align="center">
    <a href="#-introduction">Introduction</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-results">Results</a> •
    <a href="#-configuration">Configuration</a> •
    <a href="#-citation">Citation</a>
  </p>
</p>

<p align="center">
  <a href="https://m-a-p.ai/AutoKaggle.github.io/"><img src="https://img.shields.io/badge/🏠-Home Page-8A2BE2?style=flat-square" alt="Home Page"></a>
  <a href="https://arxiv.org/abs/2410.20424"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square" alt="arXiv Paper"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-yellow?style=flat-square" alt="Python">
  <img src="https://img.shields.io/github/stars/GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions?style=social" alt="Stars">
</p>

<p align="center">
  <strong>English</strong> | <a href="README_ZH.md">中文</a>
</p>

---

## 📖 Introduction

MOSAIC is a modular multi-agent framework for autonomous data science competitions. Built on the AutoKaggle foundation, MOSAIC extends the competition workflow beyond tabular pipelines with:

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Brain-Coding Loop</h3>
      <ul>
        <li>Profile-driven task identification</li>
        <li>Brain-Coding agent control loop</li>
        <li>Structured experiment memory</li>
        <li>Leaderboard-feedback-driven optimization</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🔒 Robust Execution</h3>
      <ul>
        <li>Remote execution isolation</li>
        <li>Validation gates</li>
        <li>Risk auditing</li>
        <li>Comprehensive reporting</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>👥 Multi-Agent Collaboration</h3>
      <ul>
        <li>5 specialized agents: Reader, Planner, Developer, Reviewer, Summarizer</li>
        <li>6 key competition phases</li>
        <li>Iterative development & unit testing</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🛠️ ML Tools Library</h3>
      <ul>
        <li>Validated data cleaning functions</li>
        <li>Feature engineering utilities</li>
        <li>Modeling helpers</li>
      </ul>
    </td>
  </tr>
</table>

<p align="center">
  <img src="./mdPICs/kaggle_main.png" alt="MOSAIC main workflow" width="85%">
</p>

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
sk-xxx                           # Your API key
https://api.openai.com/v1       # Base URL
```

**③ Prepare competition data**

Place Kaggle competition data in `./multi_agents/competition/`:
```
competition/
├── train.csv
├── test.csv
├── sample_submission.csv
└── overview.txt    # Copy Overview + Data sections from Kaggle competition page
```

**④ Run MOSAIC**

```bash
bash run_multi_agent.sh
```

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## ⚙️ Configuration

<details>
<summary><strong>Configuration parameters — click to expand</strong></summary>
<br>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `competitions` | — | Target competition names |
| `start_run` | 1 | Start run index |
| `end_run` | 5 | End run index |
| `dest_dir_param` | `"all_tools"` | Output directory label |
| `model` | `gpt-4o` | Base model for Planner & Developer |

Other agents default to `gpt-4o-mini`. To change them, modify `_create_agent` in `multi_agents/sop.py`.

**Output structure:**
```
multi_agents/experiments_history/
└── <competition>/
    └── <model>/
        └── <dest_dir_param>/
            └── <run_number>/
```

</details>

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 📊 Results

Evaluated across **8 diverse Kaggle competitions**:

| Metric | Score |
|--------|-------|
| Validation Submission Rate | **85%** |
| Comprehensive Score | **0.82** |

<p align="center">
  <img src="./mdPICs/main_results.png" alt="Main results" width="80%">
  <img src="./mdPICs/average_nps.png" alt="Average NPS" width="80%">
</p>

<p align="center">
  <img src="./mdPICs/unit_test.png" alt="Unit test workflow" width="80%">
</p>

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 📝 Citation

```bibtex
@misc{li2024autokagglemultiagentframeworkautonomous,
  title={AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions},
  author={Ziming Li and Qianbo Zang and David Ma and Jiawei Guo and Tianyu Zheng and
          Minghao liu and Xinyao Niu and Yue Wang and Jian Yang and Jiaheng Liu and
          Wanjun Zhong and Wangchunshu Zhou and Wenhao Huang and Ge Zhang},
  year={2024},
  eprint={2410.20424},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2410.20424},
}
```

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

<div align="right"><a href="#mosaic">↑ back to top</a></div>

---

## 📄 License

Apache 2.0 License — see [LICENSE.md](LICENSE.md) for details.

> **Disclaimer**: This project is not affiliated with, endorsed by, or officially associated with Kaggle or Google. The name "Kaggle" is used solely to indicate competition compatibility.

---

<p align="center">
  <strong>⭐ If MOSAIC helped your research, please give it a Star!</strong>
</p>

<p align="center">
  <a href="https://star-history.com/#GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions&Date">
    <img src="https://api.star-history.com/svg?repos=GetIT-Sunday/MOSAIC-Modular-Optimization-and-Search-for-Agentic-Intelligence-in-Competitions&type=Date" alt="Star History Chart" width="600">
  </a>
</p>

<p align="center">
  <sub>Made with ✨ by <a href="https://github.com/GetIT-Sunday">GetIT-Sunday</a> using <a href="https://github.com/GetIT-Sunday/ReadmeMagic-github-readme-design-skill">ReadmeMagic</a></sub>
</p>
