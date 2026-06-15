from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .project_config import ProjectConfigAgent, ProjectConfigSnapshot


@dataclass(frozen=True)
class WorkspaceSummary:
    slug: str
    profile: str
    stage: str
    last_updated: str
    next_command: str
    console_href: Optional[str]


@dataclass(frozen=True)
class ProjectConsoleSnapshot:
    cache: Dict[str, Any]
    workspaces: List[WorkspaceSummary]
    next_action: Dict[str, str]


class ProjectConsoleAgent:
    """Project-level agent that owns the AutoKaggle global console."""

    def __init__(self, competition_root: Path):
        self.competition_root = competition_root.resolve()
        self.competition_root.mkdir(parents=True, exist_ok=True)
        self.project_root = self.competition_root.parents[1]
        self.index_path = self.competition_root / "index.html"
        self.console_dir = self.competition_root / "console"

    def build_snapshot(self) -> ProjectConsoleSnapshot:
        cache = self._read_json(self.competition_root / "kaggle_competitions_cache.json")
        workspaces = self._collect_workspaces()
        return ProjectConsoleSnapshot(
            cache=cache,
            workspaces=workspaces,
            next_action=self._next_best_action(cache, workspaces),
        )

    def write_html(self) -> Path:
        snapshot = self.build_snapshot()
        self.console_dir.mkdir(parents=True, exist_ok=True)
        pages = {
            self.index_path: self._render_page(snapshot, active="overview"),
            self.console_dir / "pool.html": self._render_page(snapshot, active="pool"),
            self.console_dir / "workspaces.html": self._render_page(snapshot, active="workspaces"),
            self.console_dir / "config.html": self._render_page(snapshot, active="config"),
            self.console_dir / "roadmap.html": self._render_page(snapshot, active="roadmap"),
        }
        for path, content in pages.items():
            path.write_text(content, encoding="utf-8")
        (self.console_dir / "snapshot.json").write_text(
            json.dumps(self._snapshot_to_dict(snapshot), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ProjectConfigAgent(self.project_root).write_status(self.console_dir / "config_status.json")
        return self.index_path

    def _collect_workspaces(self) -> List[WorkspaceSummary]:
        summaries: List[WorkspaceSummary] = []
        for path in sorted(self.competition_root.iterdir()):
            if not path.is_dir() or path.name.startswith(".") or path.name in {"__pycache__", "console"}:
                continue
            if not self._looks_like_workspace(path):
                continue
            stage = self._workspace_stage(path)
            summaries.append(
                WorkspaceSummary(
                    slug=path.name,
                    profile=self._workspace_profile(path),
                    stage=stage,
                    last_updated=self._last_updated(path),
                    next_command=self._workspace_next_command(path, stage),
                    console_href=f"../{path.name}/runs/index.html"
                    if (path / "runs" / "index.html").exists()
                    else None,
                )
            )
        stage_order = {
            "needs_review": 0,
            "data_pending": 1,
            "selected": 2,
            "intake_ready": 3,
            "baseline_ready": 4,
            "brain_planned": 5,
            "experiment_running": 6,
            "submission_ready": 7,
            "feedback_pending": 8,
            "completed": 9,
            "not_started": 10,
        }
        return sorted(summaries, key=lambda item: (stage_order.get(item.stage, 99), item.slug))

    def _looks_like_workspace(self, path: Path) -> bool:
        markers = [
            "competition_intake.json",
            "data_manifest.json",
            "task_card.md",
            "baseline_review.json",
            "llm_experiment_plan.json",
            "experiment_queue.json",
            "train.csv",
            "test.csv",
            "sample_submission.csv",
        ]
        return any((path / marker).exists() for marker in markers) or (path / "runs").exists()

    def _workspace_profile(self, path: Path) -> str:
        manifest = self._read_json(path / "data_manifest.json")
        if manifest.get("profile"):
            return str(manifest["profile"])
        if manifest.get("task_type"):
            return str(manifest["task_type"])
        intake = self._read_json(path / "competition_intake.json")
        cached = intake.get("cached_competition") or {}
        return str(cached.get("category") or intake.get("task_type") or "unknown")

    def _workspace_stage(self, path: Path) -> str:
        if (path / "leaderboard_feedback_loop.json").exists():
            return "feedback_pending"
        if (path / "manual_submission_package" / "manifest.json").exists():
            return "submission_ready"
        if (path / "experiment_queue.json").exists():
            queue = self._read_json(path / "experiment_queue.json")
            return "brain_planned" if (queue.get("next_runnable") or {}).get("task_id") else "needs_review"
        if (path / "llm_experiment_plan.json").exists():
            return "brain_planned"
        if (path / "baseline_review.json").exists():
            return "baseline_ready"
        if (path / "data_manifest.json").exists() and (path / "task_card.md").exists():
            return "intake_ready"
        intake = self._read_json(path / "competition_intake.json")
        status = str(intake.get("status", ""))
        if (intake.get("intake_agent") or {}).get("blocking_items"):
            return "data_pending"
        if status == "downloading":
            return "data_pending"
        if status in {"rules_not_accepted", "download_failed", "needs_review"}:
            return "needs_review"
        if status in {"selected", "files_listed"}:
            return "selected"
        if any((path / name).exists() for name in ["train.csv", "test.csv", "sample_submission.csv"]):
            return "data_pending"
        return "not_started"

    def _workspace_next_command(self, path: Path, stage: str) -> str:
        slug = path.name
        intake = self._read_json(path / "competition_intake.json")
        intake_status = str(intake.get("status") or "")
        if intake_status == "rules_not_accepted":
            return f"在 Kaggle 网页接受 {slug} 的规则后运行 python framework.py --kaggle-select {slug} --kaggle-download"
        if intake_status == "auth_missing":
            return "python framework.py --remote-health-check"
        if intake_status == "download_failed":
            return f"python framework.py --kaggle-select {slug} --kaggle-download"
        commands = {
            "selected": f"python framework.py --competition {slug} --competition-intake",
            "data_pending": f"python framework.py --kaggle-select {slug} --kaggle-download",
            "intake_ready": f"python framework.py --competition {slug} --agent-baseline-start",
            "baseline_ready": f"python framework.py --competition {slug} --remote-brain-review",
            "brain_planned": f"python framework.py --competition {slug} --experiment-queue",
            "experiment_running": f"python framework.py --competition {slug} --run-enhancement",
            "submission_ready": f"python framework.py --competition {slug} --manual-submission-package",
            "feedback_pending": f"python framework.py --competition {slug} --leaderboard-feedback-loop",
            "needs_review": f"python framework.py --competition {slug} --competition-intake",
        }
        return commands.get(stage, f"python framework.py --competition {slug} --competition-intake")

    def _next_best_action(self, cache: Dict[str, Any], workspaces: List[WorkspaceSummary]) -> Dict[str, str]:
        status = str(cache.get("status") or "not_refreshed")
        if status in {"auth_missing", "needs_kaggle_cli", "rules_not_accepted"}:
            return {
                "title": self._kaggle_blocking_title(status),
                "command": self._kaggle_next_command(status, "<competition_slug>"),
                "reason": "Kaggle 题目池还不能稳定刷新，先修复外部认证或 CLI。",
                "owner_agent": "ProjectConsoleAgent",
            }
        competitions = cache.get("competitions") or []
        if status != "pass":
            return {
                "title": "刷新 Kaggle 真实题目池",
                "command": "python framework.py --kaggle-discover --kaggle-category playground --kaggle-sort-by recentlyCreated",
                "reason": "总控制台需要真实题目池作为选题入口。",
                "owner_agent": "ProjectConsoleAgent",
            }
        selected_slugs = {workspace.slug for workspace in workspaces}
        for item in competitions:
            slug = str(item.get("ref") or "")
            if slug and slug not in selected_slugs:
                return {
                    "title": f"选择并下载推荐题目 {slug}",
                    "command": f"python framework.py --kaggle-select {slug} --kaggle-download",
                    "reason": "题目池已有候选，但还没有进入 AutoKaggle workspace。",
                    "owner_agent": "ProjectConsoleAgent",
                }
        actionable = [workspace for workspace in workspaces if workspace.stage != "completed"]
        if actionable:
            workspace = actionable[0]
            return {
                "title": f"推进 {workspace.slug} 到下一阶段",
                "command": workspace.next_command,
                "reason": f"当前阶段为 {workspace.stage}，需要继续闭环。",
                "owner_agent": "ProjectConsoleAgent",
            }
        return {
            "title": "刷新题目池并选择新竞赛",
            "command": "python framework.py --kaggle-discover --kaggle-category playground --kaggle-sort-by recentlyCreated",
            "reason": "当前 workspace 暂无可推进动作。",
            "owner_agent": "ProjectConsoleAgent",
        }

    def _last_updated(self, path: Path) -> str:
        mtimes = [child.stat().st_mtime for child in path.rglob("*") if child.is_file()]
        return datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M") if mtimes else "-"

    def _render_page(self, snapshot: ProjectConsoleSnapshot, *, active: str) -> str:
        content = {
            "overview": self._render_overview(snapshot),
            "pool": self._render_pool(snapshot),
            "workspaces": self._render_workspaces_page(snapshot),
            "config": self._render_config(snapshot),
            "roadmap": self._render_roadmap(snapshot),
        }[active]
        return self._layout(
            title=self._page_title(active),
            active=active,
            body=content,
            root_prefix="" if active == "overview" else "../",
        )

    def _render_overview(self, snapshot: ProjectConsoleSnapshot) -> str:
        cache_status = str(snapshot.cache.get("status") or "not_refreshed")
        preview = "".join(
            f"<li><strong>{html.escape(workspace.slug)}</strong><span class=\"chip {html.escape(workspace.stage)}\">{html.escape(workspace.stage)}</span></li>"
            for workspace in snapshot.workspaces[:5]
        ) or "<li><strong>暂无 workspace</strong><span class=\"chip\">not_started</span></li>"
        return f"""
        <header class="page">
          <h2>AutoKaggle 总控制台</h2>
          <p>由 ProjectConsoleAgent 管理项目级状态：题目池、远端配置、workspace 阶段、下一步动作与银牌路线图。</p>
        </header>
        {self._render_next_action(snapshot)}
        <section class="split">
          <article class="panel">
            <div class="panel-head">
              <div>
                <h3>项目状态</h3>
                <p class="muted">总览只显示摘要；详细列表请进入左侧独立页面。</p>
              </div>
              <span class="chip {html.escape(cache_status)}">Kaggle cache: {html.escape(cache_status)}</span>
            </div>
            <ul class="compact-list">
              <li><strong>管理 Agent</strong><span>ProjectConsoleAgent</span></li>
              <li><strong>Workspace 数量</strong><span>{len(snapshot.workspaces)}</span></li>
              <li><strong>题目池数量</strong><span>{len(snapshot.cache.get("competitions") or [])}</span></li>
              <li><strong>远端执行</strong><span>dev / mac / hard workspace</span></li>
            </ul>
          </article>
          <article class="panel">
            <div class="panel-head">
              <div>
                <h3>当前 Workspace</h3>
                <p class="muted">按需要推进的优先级排序。</p>
              </div>
              <a href="console/workspaces.html">查看全部</a>
            </div>
            <ul class="compact-list">{preview}</ul>
          </article>
        </section>
        """

    def _render_pool(self, snapshot: ProjectConsoleSnapshot) -> str:
        cache_status = str(snapshot.cache.get("status") or "not_refreshed")
        return f"""
        <header class="page">
          <h2>Kaggle 题目池</h2>
          <p>真实 Kaggle competition discovery 缓存。这里是人工选题入口，不属于任何单个竞赛。</p>
        </header>
        {self._render_next_action(snapshot)}
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>可选竞赛</h3>
              <p class="muted">缓存时间：{html.escape(str(snapshot.cache.get("created_at") or "-"))}</p>
            </div>
            <span class="chip {html.escape(cache_status)}">{html.escape(cache_status)}</span>
          </div>
          <table>
            <thead>
              <tr>
                <th style="width: 30%">Slug</th>
                <th>Category</th>
                <th>Deadline</th>
                <th>Teams</th>
                <th>Status</th>
                <th style="width: 28%">Action</th>
              </tr>
            </thead>
            <tbody>{self._render_competition_pool(snapshot.cache)}</tbody>
          </table>
        </section>
        """

    def _render_workspaces_page(self, snapshot: ProjectConsoleSnapshot) -> str:
        return f"""
        <header class="page">
          <h2>Competition Workspaces</h2>
          <p>每个 workspace 对应一个竞赛题目。进入后只看该题目的 intake、baseline、Brain plan 与实验记录。</p>
        </header>
        {self._render_next_action(snapshot)}
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>Workspace 阶段</h3>
              <p class="muted">由 ProjectConsoleAgent 根据关键产物自动推断。</p>
            </div>
            <span class="chip">{len(snapshot.workspaces)} 个 workspace</span>
          </div>
          <table>
            <thead>
              <tr>
                <th style="width: 20%">Slug</th>
                <th>Profile</th>
                <th>Stage</th>
                <th>Last Update</th>
                <th style="width: 32%">Next Command</th>
                <th>Console</th>
              </tr>
            </thead>
            <tbody>{self._render_workspaces(snapshot.workspaces)}</tbody>
          </table>
        </section>
        """

    def _render_config(self, snapshot: ProjectConsoleSnapshot) -> str:
        config_snapshot = ProjectConfigAgent(self.project_root).snapshot()
        return f"""
        <header class="page">
          <h2>配置与执行边界</h2>
          <p>这里由 ProjectConfigAgent 检查用户运行所需配置：SSH、远端 workspace、conda、LLM 模型、API key 状态和 Kaggle 认证入口。不读取、不展示任何密钥内容。</p>
        </header>
        {self._render_config_setup(config_snapshot)}
        {self._render_config_checks(config_snapshot)}
        {self._render_remote_health()}
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>运行环境</h3>
              <p class="muted">Mac 作为控制端；Linux 远端作为真实实验执行端。</p>
            </div>
            <span class="chip">read_only_status</span>
          </div>
          <ul class="status-strip">{self._render_environment_status(snapshot.cache)}</ul>
        </section>
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>安全边界</h3>
              <p class="muted">远端操作必须限制在硬编码 workspace 下。</p>
            </div>
          </div>
          <ul class="compact-list">
            <li><strong>远端 Host</strong><span>dev</span></li>
            <li><strong>硬 workspace</strong><span>/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac</span></li>
            <li><strong>项目路径</strong><span>/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac/workspaces/AutoKaggle</span></li>
            <li><strong>Conda 环境</strong><span>.conda/envs/mac</span></li>
            <li><strong>密钥策略</strong><span>只显示路径/状态，不写入 HTML、JSON、日志或 git。</span></li>
          </ul>
        </section>
        """

    def _render_remote_health(self) -> str:
        report = self._read_json(self.console_dir / "remote_health_check.json")
        if not report:
            return """
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>远端健康检查</h3>
              <p class="muted">尚未运行。该检查只读，不会修改远端文件。</p>
            </div>
            <span class="chip needs_review">not_run</span>
          </div>
          <ul class="compact-list">
            <li><strong>运行命令</strong><span><code>python framework.py --remote-health-check</code></span></li>
            <li><strong>检查范围</strong><span>SSH、workspace、路径边界、conda、Python、磁盘、GPU、Kaggle 配置目录</span></li>
          </ul>
        </section>
            """
        rows = "".join(
            f"""
            <tr>
              <td><strong>{html.escape(str(item.get("label", item.get("key", "unknown"))))}</strong><small>{html.escape(str(item.get("key", "")))}</small></td>
              <td><span class="chip {html.escape(str(item.get("status", "unknown")))}">{html.escape(str(item.get("status", "unknown")))}</span></td>
              <td>{html.escape(str(item.get("detail", "")))}</td>
            </tr>
            """
            for item in report.get("checks", [])
        )
        return f"""
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>远端健康检查</h3>
              <p class="muted">最后检查：{html.escape(str(report.get("generated_at", "-")))}</p>
            </div>
            <a href="remote_health_check.json">remote_health_check.json</a>
            <span class="chip {html.escape(str(report.get("status", "unknown")))}">{html.escape(str(report.get("status", "unknown")))}</span>
          </div>
          <table>
            <thead>
              <tr>
                <th style="width: 26%">项目</th>
                <th style="width: 16%">状态</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        """

    def _render_config_setup(self, config_snapshot: ProjectConfigSnapshot) -> str:
        private_status = "已存在" if config_snapshot.using_private_config else "未创建"
        return f"""
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>用户配置方式</h3>
              <p class="muted">每个人维护自己的私有配置文件，不提交密钥。</p>
            </div>
            <a href="../../../docs/user_setup.md">用户配置文档</a>
            <span class="chip {'pass' if config_snapshot.using_private_config else 'needs_review'}">{private_status}</span>
          </div>
          <ul class="compact-list">
            <li><strong>1. 复制示例配置</strong><span><code>cp autokaggle_config.example.json autokaggle_config.json</code></span></li>
            <li><strong>2. 填写 SSH/远端</strong><span>ssh_alias、workspace、project_subdir、conda_env</span></li>
            <li><strong>3. 填写 LLM 模型</strong><span>planner_model、coding_model、cheap_model、base_url</span></li>
            <li><strong>4. 设置 API Key</strong><span>环境变量或本地 api_key.txt，页面只显示状态</span></li>
            <li><strong>5. 运行检查</strong><span><code>python framework.py --config-check</code></span></li>
          </ul>
        </section>
        """

    def _render_config_checks(self, config_snapshot: ProjectConfigSnapshot) -> str:
        rows = "".join(
            f"""
            <tr>
              <td><strong>{html.escape(check.label)}</strong><small>{html.escape(check.key)}</small></td>
              <td><span class="chip {html.escape(check.status)}">{html.escape(check.status)}</span></td>
              <td>{html.escape(check.detail)}</td>
            </tr>
            """
            for check in config_snapshot.checks
        )
        return f"""
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>配置检查</h3>
              <p class="muted">来源：{html.escape(config_snapshot.config_path)}；示例：{html.escape(config_snapshot.example_path)}</p>
            </div>
            <a href="config_status.json">config_status.json</a>
          </div>
          <table>
            <thead>
              <tr>
                <th style="width: 26%">项目</th>
                <th style="width: 16%">状态</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        """

    def _render_roadmap(self, snapshot: ProjectConsoleSnapshot) -> str:
        steps = [
            ("1", "题目池刷新与人工选题", "已接入 Kaggle CLI cache，下一步围绕真实题目推进。"),
            ("2", "Intake 与目标解析", "生成 task card、metric spec、data manifest、unknown 字段。"),
            ("3", "Baseline 闭环", "先覆盖 tabular baseline，再扩展 CV、模型搜索和 leaderboard 风险控制。"),
            ("4", "Brain 决策与 Coding Runner", "Brain 输出窄任务，Runner 只执行，不擅自改变目标。"),
            ("5", "银牌冲刺能力", "加入历史方案检索、特征工程库、调参、集成、提交反馈循环。"),
        ]
        rows = "".join(
            f"<li><strong>{number}. {html.escape(title)}</strong><span>{html.escape(desc)}</span></li>"
            for number, title, desc in steps
        )
        return f"""
        <header class="page">
          <h2>银牌路线图</h2>
          <p>这个页面不是实验结果，而是 ProjectConsoleAgent 对项目级推进路线的可视化入口。</p>
        </header>
        {self._render_next_action(snapshot)}
        <section class="panel">
          <div class="panel-head">
            <div>
              <h3>当前路线</h3>
              <p class="muted">详细文档见 docs/kaggle_agent_silver_roadmap.md。</p>
            </div>
            <a href="../../../docs/kaggle_agent_silver_roadmap.md">打开路线图文档</a>
          </div>
          <ul class="compact-list roadmap-list">{rows}</ul>
        </section>
        """

    def _render_next_action(self, snapshot: ProjectConsoleSnapshot) -> str:
        action = snapshot.next_action
        return f"""
        <section class="next-action">
          <strong>{html.escape(action["title"])}</strong>
          <p class="muted">{html.escape(action["reason"])}</p>
          <code>{html.escape(action["command"])}</code>
          <span class="chip">Owner: {html.escape(action.get("owner_agent", "ProjectConsoleAgent"))}</span>
        </section>
        """

    def _render_environment_status(self, cache: Dict[str, Any]) -> str:
        kaggle_status = self._kaggle_status_label(str(cache.get("status") or "not_refreshed"))
        items = [
            ("Kaggle 认证", kaggle_status),
            ("SSH 远端服务器", "dev"),
            ("远端 Workspace", "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac"),
            ("Conda 环境", "mac"),
            ("Brain LLM", "mimo-v2.5-pro"),
            ("Coding LLM", "mimo-v2.5"),
        ]
        return "".join(
            f"""
            <li>
              <span>{html.escape(label)}</span>
              <strong>{html.escape(value)}</strong>
            </li>
            """
            for label, value in items
        )

    def _render_competition_pool(self, cache: Dict[str, Any]) -> str:
        competitions = cache.get("competitions") or []
        if not competitions:
            return """
            <tr>
              <td colspan="6">暂无真实题目缓存。先刷新 Kaggle 题目池。</td>
            </tr>
            """
        rows = []
        cache_status = str(cache.get("status") or "unknown")
        for item in competitions[:20]:
            slug = str(item.get("ref") or "unknown")
            rows.append(
                f"""
                <tr>
                  <td><strong>{html.escape(slug)}</strong><small>{html.escape(str(item.get("title") or ""))}</small></td>
                  <td>{html.escape(str(item.get("category") or "unknown"))}</td>
                  <td>{html.escape(str(item.get("deadline") or "-"))}</td>
                  <td>{html.escape(str(item.get("team_count") or item.get("teamCount") or "-"))}</td>
                  <td><span class="chip {html.escape(cache_status)}">{html.escape(cache_status)}</span></td>
                  <td><code>python framework.py --kaggle-select {html.escape(slug)} --kaggle-download</code></td>
                </tr>
                """
            )
        return "".join(rows)

    def _render_workspaces(self, workspaces: List[WorkspaceSummary]) -> str:
        if not workspaces:
            return """
            <tr>
              <td colspan="6">还没有 competition workspace。请先从题目池选择一个竞赛。</td>
            </tr>
            """
        rows = []
        for workspace in workspaces:
            link = (
                f'<a href="{html.escape(workspace.console_href)}">进入控制台</a>'
                if workspace.console_href
                else "<span>等待生成</span>"
            )
            rows.append(
                f"""
                <tr>
                  <td><strong>{html.escape(workspace.slug)}</strong></td>
                  <td>{html.escape(workspace.profile)}</td>
                  <td><span class="chip {html.escape(workspace.stage)}">{html.escape(workspace.stage)}</span></td>
                  <td>{html.escape(workspace.last_updated)}</td>
                  <td><code>{html.escape(workspace.next_command)}</code></td>
                  <td>{link}</td>
                </tr>
                """
            )
        return "".join(rows)

    def _layout(self, *, title: str, active: str, body: str, root_prefix: str) -> str:
        nav = self._render_nav(active, root_prefix)
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{self._css()}</style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>AutoKaggle</h1>
      <p class="agent-badge">ProjectConsoleAgent</p>
      {nav}
    </aside>
    <main>{body}</main>
  </div>
</body>
</html>
"""

    def _render_nav(self, active: str, root_prefix: str) -> str:
        links = [
            ("overview", "总览", f"{root_prefix}index.html"),
            ("pool", "题目池", f"{root_prefix}console/pool.html"),
            ("workspaces", "Workspaces", f"{root_prefix}console/workspaces.html"),
            ("config", "配置", f"{root_prefix}console/config.html"),
            ("roadmap", "路线图", f"{root_prefix}console/roadmap.html"),
        ]
        return "\n".join(
            f'<a class="{"active" if key == active else ""}" href="{html.escape(href)}">{html.escape(label)}</a>'
            for key, label, href in links
        )

    def _page_title(self, active: str) -> str:
        titles = {
            "overview": "AutoKaggle 总控制台",
            "pool": "Kaggle 题目池",
            "workspaces": "Competition Workspaces",
            "config": "配置与执行边界",
            "roadmap": "银牌路线图",
        }
        return titles[active]

    def _snapshot_to_dict(self, snapshot: ProjectConsoleSnapshot) -> Dict[str, Any]:
        return {
            "generated_by": "ProjectConsoleAgent",
            "cache_status": snapshot.cache.get("status"),
            "next_action": snapshot.next_action,
            "workspaces": [workspace.__dict__ for workspace in snapshot.workspaces],
        }

    def _kaggle_status_label(self, status: str) -> str:
        mapping = {
            "pass": "已认证 / 可刷新",
            "auth_missing": "未认证",
            "needs_kaggle_cli": "CLI 未安装",
            "rules_not_accepted": "需接受规则",
            "not_refreshed": "未刷新",
        }
        return mapping.get(status, status)

    def _kaggle_blocking_title(self, status: str) -> str:
        mapping = {
            "auth_missing": "完成 Kaggle 认证",
            "needs_kaggle_cli": "安装 Kaggle CLI",
            "rules_not_accepted": "接受 Kaggle 竞赛规则",
        }
        return mapping.get(status, "检查 Kaggle 状态")

    def _kaggle_next_command(self, status: str, slug: str) -> str:
        if status == "auth_missing":
            return 'cd /home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac && "$PWD/.conda/envs/mac/bin/kaggle" auth login'
        if status == "needs_kaggle_cli":
            return "pip install kaggle"
        if status == "rules_not_accepted":
            return f"在 Kaggle 网页接受 {slug} 的规则后重试下载"
        return "python framework.py --kaggle-discover --kaggle-category playground --kaggle-sort-by recentlyCreated"

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _css(self) -> str:
        return """
    :root {
      --blue: #20beff;
      --blue-dark: #008abc;
      --text: #202124;
      --muted: #5f6368;
      --line: #dadce0;
      --soft: #f8f9fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #ffffff;
    }
    .layout {
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 24px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      background: #ffffff;
    }
    aside h1 {
      font-size: 21px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }
    .agent-badge {
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 20px;
    }
    aside a {
      display: block;
      padding: 9px 10px;
      border-radius: 6px;
      color: var(--muted);
      text-decoration: none;
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 2px;
    }
    aside a.active, aside a:hover {
      color: var(--blue-dark);
      background: #e8f7fd;
    }
    main {
      max-width: 1180px;
      width: 100%;
      padding: 28px 32px 48px;
    }
    header.page {
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 20px;
    }
    h2, h3, p { margin: 0; }
    header.page h2 {
      font-size: 28px;
      line-height: 1.2;
      margin-bottom: 8px;
    }
    header.page p, .muted {
      color: var(--muted);
      line-height: 1.5;
    }
    .split {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 22px;
      background: #ffffff;
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
    }
    .panel-head h3 {
      font-size: 18px;
      margin-bottom: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid #eef0f2;
      padding: 11px 14px;
      vertical-align: top;
      font-size: 14px;
      word-break: break-word;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      background: #ffffff;
    }
    code {
      background: #f1f3f4;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 3px 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }
    a {
      color: var(--blue-dark);
      text-decoration: none;
      font-weight: 600;
    }
    small {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid #d2d6da;
      background: #f8f9fa;
      color: #3c4043;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
    }
    .chip.pass, .chip.baseline_ready, .chip.intake_ready, .chip.submission_ready {
      border-color: #b7e1cd;
      background: #e6f4ea;
      color: #137333;
    }
    .chip.selected, .chip.brain_planned {
      border-color: #c6dafc;
      background: #e8f0fe;
      color: #174ea6;
    }
    .chip.data_pending, .chip.feedback_pending, .chip.needs_review, .chip.auth_missing {
      border-color: #fdd663;
      background: #fef7e0;
      color: #b06000;
    }
    .next-action {
      border: 1px solid #c6dafc;
      border-radius: 8px;
      padding: 16px 18px;
      background: #f8fbff;
      margin-bottom: 22px;
    }
    .next-action strong {
      display: block;
      font-size: 17px;
      margin-bottom: 8px;
    }
    .next-action code {
      display: block;
      margin: 12px 0 8px;
      padding: 10px;
      overflow-x: auto;
    }
    .compact-list, .status-strip {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .compact-list li, .status-strip li {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 12px 16px;
      border-bottom: 1px solid #eef0f2;
      min-height: 46px;
    }
    .compact-list span, .status-strip span {
      color: var(--muted);
      text-align: right;
      word-break: break-word;
    }
    .roadmap-list li {
      align-items: flex-start;
      flex-direction: column;
    }
    .roadmap-list span {
      text-align: left;
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      main { padding: 22px 16px 40px; }
      .compact-list li, .status-strip li { flex-direction: column; }
      .compact-list span, .status-strip span { text-align: left; }
    }
        """


class ProjectControlPanel(ProjectConsoleAgent):
    """Backward-compatible name for the project-level console agent."""
