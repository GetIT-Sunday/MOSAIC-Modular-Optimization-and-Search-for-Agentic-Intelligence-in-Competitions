from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunLedgerEntry:
    run_id: str
    task_id: str
    agent: str
    status: str
    title: str
    input_path: str
    prompt_path: str
    artifacts_dir: str
    scorecard_path: str
    human_review_path: str
    html_report_path: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


class RunLedger:
    def __init__(self, competition_dir: Path):
        self.competition_dir = competition_dir.resolve()
        self.runs_dir = self.competition_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.console_dir = self.runs_dir / "console"
        self.index_path = self.runs_dir / "ledger.jsonl"
        self.html_path = self.runs_dir / "index.html"

    def create_entry(
        self,
        *,
        task_id: str,
        agent: str,
        title: str,
        status: str,
        input_payload: Dict[str, Any],
        prompt: str,
        scorecard: Dict[str, Any],
        artifacts: Optional[Dict[str, Path]] = None,
    ) -> RunLedgerEntry:
        run_id = self._next_run_id(task_id)
        run_dir = self.runs_dir / run_id
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / "input.json"
        prompt_path = run_dir / "prompt.md"
        scorecard_path = run_dir / "scorecard.json"
        human_review_path = run_dir / "human_review.md"

        input_path.write_text(
            json.dumps(input_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        prompt_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")
        scorecard_path.write_text(
            json.dumps(scorecard, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not human_review_path.exists():
            human_review_path.write_text(self._default_human_review(), encoding="utf-8")

        for name, source in (artifacts or {}).items():
            if source.exists() and source.is_file():
                target = artifacts_dir / f"{name}{source.suffix}"
                target.write_bytes(source.read_bytes())

        entry = RunLedgerEntry(
            run_id=run_id,
            task_id=task_id,
            agent=agent,
            status=status,
            title=title,
            input_path=str(input_path.relative_to(self.competition_dir)),
            prompt_path=str(prompt_path.relative_to(self.competition_dir)),
            artifacts_dir=str(artifacts_dir.relative_to(self.competition_dir)),
            scorecard_path=str(scorecard_path.relative_to(self.competition_dir)),
            human_review_path=str(human_review_path.relative_to(self.competition_dir)),
            html_report_path=str(self.html_path.relative_to(self.competition_dir)),
        )
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self.write_html()
        return entry

    def list_entries(self) -> List[RunLedgerEntry]:
        if not self.index_path.exists():
            return []
        entries = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(RunLedgerEntry(**json.loads(line)))
        return entries

    def write_html(self) -> Path:
        entries = self.list_entries()
        cards = "\n".join(self._render_entry(entry) for entry in entries)
        experiment_board = self._render_experiment_board(entries)
        audit_log = self._render_audit_log(entries)
        summary = self._render_summary(entries)
        submit_decision_handoff = self._render_submit_decision_handoff()
        feedback_template_fill = self._render_feedback_template_fill()
        leaderboard_feedback_loop = self._render_leaderboard_feedback_loop()
        submission_handoff = self._render_submission_handoff()
        experiment_queue = self._render_experiment_queue()
        experiment_roadmap = self._render_experiment_roadmap()
        competition_intake = self._render_competition_intake()
        submission_decision = self._render_submission_decision()
        promotion_gate = self._render_promotion_gate()
        self.console_dir.mkdir(parents=True, exist_ok=True)
        legacy_search_index = (
            competition_intake
            + submit_decision_handoff
            + feedback_template_fill
            + leaderboard_feedback_loop
            + submission_handoff
            + experiment_roadmap
            + experiment_queue
            + submission_decision
            + promotion_gate
            + f'<section class="grid">{cards}</section>'
        )
        overview_body = (
            summary
            + (competition_intake or self._empty_panel("竞赛 Intake", "尚未生成 intake。"))
            + f'<template id="ledger-search-index">{legacy_search_index}</template>'
        )
        data_body = competition_intake or self._empty_panel("数据与 Intake", "尚未生成 data manifest 或 competition intake。")
        brain_body = (
            experiment_roadmap
            + experiment_queue
            + promotion_gate
            or self._empty_panel("Brain 计划", "尚未生成 Brain plan、实验队列或 promotion gate。")
        )
        experiments_body = experiment_board + audit_log if entries else self._empty_panel("实验记录", "暂无 Run Ledger 记录。")
        submission_body = (
            submit_decision_handoff
            + feedback_template_fill
            + leaderboard_feedback_loop
            + submission_handoff
            + submission_decision
            or self._empty_panel("提交与反馈", "尚未生成提交交接、榜单反馈或人工提交审核。")
        )
        pages = {
            self.html_path: ("overview", overview_body, ""),
            self.console_dir / "data.html": ("data", data_body, "../"),
            self.console_dir / "brain.html": ("brain", brain_body, "../"),
            self.console_dir / "experiments.html": ("experiments", experiments_body, "../"),
            self.console_dir / "submission.html": ("submission", submission_body, "../"),
        }
        for path, (active, body, root_prefix) in pages.items():
            path.write_text(
                self._html_template(
                    competition_name=self.competition_dir.name,
                    active=active,
                    body=body,
                    root_prefix=root_prefix,
                    total=len(entries),
                ),
                encoding="utf-8",
            )
        return self.html_path

    def _next_run_id(self, task_id: str) -> str:
        count = len(self.list_entries()) + 1
        safe_task_id = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in task_id
        )
        return f"{count:04d}_{safe_task_id}"

    def _render_entry(self, entry: RunLedgerEntry) -> str:
        scorecard = self._read_json(self.competition_dir / entry.scorecard_path)
        scores = scorecard.get("scores", {})
        issues = scorecard.get("issues", [])
        action = scorecard.get("recommended_human_action", "review")
        metric_name = scorecard.get("metric_name", "")
        local_score = scorecard.get("local_score", None)
        metric_block = ""
        if metric_name or local_score is not None:
            metric_block = f"""
          <section class="metric-strip">
            <div><span>评价指标</span><strong>{html.escape(str(metric_name or "unknown"))}</strong></div>
            <div><span>本地分数</span><strong>{html.escape(str(local_score if local_score is not None else "n/a"))}</strong></div>
          </section>
          """
        score_items = "".join(
            f"<li><span>{html.escape(str(key))}</span><strong>{html.escape(str(value))}</strong></li>"
            for key, value in scores.items()
        )
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues)
        if not issue_items:
            issue_items = "<li>暂无记录的问题。</li>"
        return f"""
        <article class="run-card">
          <header>
            <div>
              <p class="eyebrow">{html.escape(entry.run_id)} · {html.escape(entry.agent)}</p>
              <h2>{html.escape(entry.title)}</h2>
            </div>
            <span class="status {html.escape(entry.status)}">{html.escape(entry.status)}</span>
          </header>
          <dl>
            <div><dt>任务</dt><dd>{html.escape(entry.task_id)}</dd></div>
            <div><dt>创建时间</dt><dd>{html.escape(entry.created_at)}</dd></div>
            <div><dt>建议动作</dt><dd>{html.escape(action)}</dd></div>
          </dl>
          {metric_block}
          <section>
            <h3>分数</h3>
            <ul class="scores">{score_items}</ul>
          </section>
          <section>
            <h3>问题</h3>
            <ul>{issue_items}</ul>
          </section>
          <nav>
            <a href="../{html.escape(entry.input_path)}">输入</a>
            <a href="../{html.escape(entry.prompt_path)}">提示词</a>
            <a href="../{html.escape(entry.scorecard_path)}">评分卡</a>
            <a href="../{html.escape(entry.human_review_path)}">人工审核</a>
            <a href="../{html.escape(entry.artifacts_dir)}/">产物</a>
          </nav>
        </article>
        """

    def _render_experiment_board(self, entries: List[RunLedgerEntry]) -> str:
        rows_by_task: Dict[str, tuple] = {}
        best_score = self._best_experiment_score(entries)
        for entry in entries:
            scorecard = self._read_json(self.competition_dir / entry.scorecard_path)
            run_kind = self._run_kind(entry, scorecard)
            if run_kind not in {"baseline", "model_experiment"}:
                continue
            local_score = scorecard.get("local_score")
            delta = self._format_delta(local_score, best_score)
            candidate = (self._score_sort_value(local_score), entry, scorecard, run_kind, delta)
            existing = rows_by_task.get(entry.task_id)
            if existing is None or candidate[0] >= existing[0]:
                rows_by_task[entry.task_id] = candidate
        experiment_rows = list(rows_by_task.values())
        experiment_rows.sort(key=lambda item: item[0], reverse=True)
        total_experiments = len(experiment_rows)
        displayed_rows = experiment_rows[:20]
        if not experiment_rows:
            rows = """
            <tr>
              <td colspan="10">暂无模型实验。baseline、tabular search 或 enhancement 运行后会出现在这里。</td>
            </tr>
            """
        else:
            rows = "".join(
                self._render_experiment_row(rank, entry, scorecard, run_kind, delta)
                for rank, (_, entry, scorecard, run_kind, delta) in enumerate(displayed_rows, start=1)
            )
            if total_experiments > len(displayed_rows):
                rows += f"""
            <tr>
              <td colspan="10">仅展示 Top {len(displayed_rows)} / {total_experiments} 个去重实验；完整历史请看下方审计日志。</td>
            </tr>
                """
        summary = self._render_experiment_board_summary(experiment_rows)
        return f"""
        <section class="experiment-board">
          <header>
            <div>
              <p class="eyebrow">Experiment Board</p>
              <h2>实验排行榜</h2>
            </div>
            <span class="status ready">Top {len(displayed_rows)} / {total_experiments}</span>
          </header>
          {summary}
          <table class="experiment-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th style="width: 20%">Task</th>
                <th>Kind</th>
                <th>Runner</th>
                <th>Status</th>
                <th>Metric</th>
                <th>CV</th>
                <th>Delta</th>
                <th>Action</th>
                <th>Artifacts</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        """

    def _render_experiment_board_summary(self, rows: List[tuple]) -> str:
        if not rows:
            return """
          <section class="board-summary">
            <div><span>Champion</span><strong>none</strong></div>
            <div><span>Best CV</span><strong>n/a</strong></div>
            <div><span>实验数</span><strong>0</strong></div>
          </section>
            """
        _, entry, scorecard, run_kind, _ = rows[0]
        score = scorecard.get("local_score")
        metric = scorecard.get("metric_name") or "metric"
        return f"""
          <section class="board-summary">
            <div><span>当前 Champion</span><strong>{html.escape(entry.title)}</strong></div>
            <div><span>Best CV</span><strong>{html.escape(str(metric))}={html.escape(self._format_score(score))}</strong></div>
            <div><span>类型</span><strong>{html.escape(run_kind)}</strong></div>
          </section>
        """

    def _render_experiment_row(
        self,
        rank: int,
        entry: RunLedgerEntry,
        scorecard: Dict[str, Any],
        run_kind: str,
        delta: str,
    ) -> str:
        local_score = scorecard.get("local_score")
        metric = scorecard.get("metric_name") or "n/a"
        action = scorecard.get("recommended_human_action", "review")
        return f"""
              <tr>
                <td>{rank}</td>
                <td><strong>{html.escape(entry.title)}</strong><small>{html.escape(entry.task_id)}</small></td>
                <td><span class="status {html.escape(run_kind)}">{html.escape(run_kind)}</span></td>
                <td>{html.escape(entry.agent)}</td>
                <td><span class="status {html.escape(entry.status)}">{html.escape(entry.status)}</span></td>
                <td>{html.escape(str(metric))}</td>
                <td>{html.escape(self._format_score(local_score))}</td>
                <td>{html.escape(delta)}</td>
                <td>{html.escape(str(action))}</td>
                <td>
                  <a href="../{html.escape(entry.scorecard_path)}">评分卡</a>
                  <a href="../{html.escape(entry.artifacts_dir)}/">产物</a>
                </td>
              </tr>
        """

    def _render_audit_log(self, entries: List[RunLedgerEntry]) -> str:
        grouped: Dict[str, List[RunLedgerEntry]] = {}
        for entry in entries:
            scorecard = self._read_json(self.competition_dir / entry.scorecard_path)
            grouped.setdefault(self._run_kind(entry, scorecard), []).append(entry)
        details = []
        preferred_order = [
            "baseline",
            "model_experiment",
            "champion",
            "brain_decision",
            "validation",
            "submission",
            "feedback",
            "audit",
            "system",
        ]
        for kind in preferred_order:
            group_entries = grouped.get(kind, [])
            if not group_entries:
                continue
            cards = "\n".join(self._render_entry(entry) for entry in group_entries)
            open_attr = " open" if kind in {"baseline", "model_experiment", "champion"} else ""
            details.append(
                f"""
        <details class="audit-group"{open_attr}>
          <summary>{html.escape(kind)} · {len(group_entries)} 条</summary>
          <section class="grid">{cards}</section>
        </details>
                """
            )
        return f"""
        <section class="audit-log">
          <header>
            <div>
              <p class="eyebrow">Run Ledger</p>
              <h2>审计日志</h2>
            </div>
            <span class="status">{len(entries)} runs</span>
          </header>
          {''.join(details)}
        </section>
        """

    def _run_kind(self, entry: RunLedgerEntry, scorecard: Dict[str, Any]) -> str:
        agent = entry.agent
        task = entry.task_id
        title = entry.title.lower()
        if agent == "baseline_runner" or task.startswith("baseline_"):
            return "baseline"
        if agent in {
            "enhancement_runner",
            "tabular_search_runner",
            "stability_first_runner",
            "tabular_feature_pruner",
        }:
            return "model_experiment"
        if agent == "champion_selector" or "champion" in task:
            return "champion"
        if agent in {"brain", "remote_brain", "experiment_queue", "experiment_roadmap", "promotion_gate"}:
            return "brain_decision"
        if agent in {"validator", "competition_intake_agent", "submission_gate"}:
            return "validation"
        if "submit" in task or agent in {
            "kaggle_submit_adapter",
            "manual_submission_package",
            "manual_submission_package_verifier",
            "manual_submit_readiness",
            "submission_decision_review",
            "submission_policy",
            "submit_decision_handoff",
            "post_submit_workflow",
            "post_experiment_pipeline",
            "post_reselection_gate",
        }:
            return "submission"
        if "feedback" in task or "leaderboard" in agent:
            return "feedback"
        if "audit" in task or "audit" in title or agent in {
            "leaderboard_gap_auditor",
            "tabular_risk_auditor",
            "tabular_feature_leakage_auditor",
        }:
            return "audit"
        return str(scorecard.get("run_kind") or "system")

    def _best_experiment_score(self, entries: List[RunLedgerEntry]) -> Optional[float]:
        best_score: Optional[float] = None
        for entry in entries:
            scorecard = self._read_json(self.competition_dir / entry.scorecard_path)
            if self._run_kind(entry, scorecard) not in {"baseline", "model_experiment"}:
                continue
            score = scorecard.get("local_score")
            if isinstance(score, (int, float)):
                best_score = score if best_score is None else max(best_score, float(score))
        return best_score

    def _score_sort_value(self, score: Any) -> float:
        if isinstance(score, (int, float)):
            return float(score)
        return float("-inf")

    def _format_score(self, score: Any) -> str:
        if isinstance(score, float):
            return f"{score:.6f}"
        if isinstance(score, int):
            return str(score)
        return "n/a"

    def _format_delta(self, score: Any, best_score: Optional[float]) -> str:
        if not isinstance(score, (int, float)) or best_score is None:
            return "n/a"
        return f"{float(score) - best_score:+.6f}"

    def _render_summary(self, entries: List[RunLedgerEntry]) -> str:
        status_counts: Dict[str, int] = {}
        best_score = None
        best_metric = ""
        best_title = ""
        for entry in entries:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
            scorecard = self._read_json(self.competition_dir / entry.scorecard_path)
            score = scorecard.get("local_score")
            if isinstance(score, (int, float)) and (best_score is None or score > best_score):
                best_score = score
                best_metric = str(scorecard.get("metric_name", "metric"))
                best_title = entry.title
        status_items = "".join(
            f"<div><span>{html.escape(status)}</span><strong>{count}</strong></div>"
            for status, count in sorted(status_counts.items())
        )
        best_text = "还没有带分数的实验。"
        if best_score is not None:
            best_text = f"{html.escape(best_title)} · {html.escape(best_metric)}={html.escape(str(best_score))}"
        return f"""
        <section class="control-summary">
          <div class="summary-card">
            <span>当前最佳实验</span>
            <strong>{best_text}</strong>
          </div>
          <div class="summary-card status-summary">
            <span>运行状态统计</span>
            <div>{status_items}</div>
          </div>
          <div class="summary-card">
            <span>人工审核 Gate</span>
            <strong>编辑每个 run 的 human_review.md，决定 continue、rerun、patch_prompt 或 stop。</strong>
          </div>
        </section>
        """

    def _render_kaggle_discovery(self) -> str:
        cache = self._read_json(self.competition_dir.parent / "kaggle_competitions_cache.json")
        if not cache:
            return f"""
        <section class="kaggle-discovery">
          <header>
            <div>
              <p class="eyebrow">Kaggle Discovery Agent</p>
              <h2>Kaggle 题目池</h2>
            </div>
            <span class="status needs_review">not_refreshed</span>
          </header>
          <section class="handoff-command">
            <span>刷新最新 Kaggle 题目</span>
            <pre><code>python framework.py --kaggle-discover --kaggle-category featured --kaggle-sort-by recentlyCreated</code></pre>
          </section>
        </section>
        """
        competitions = cache.get("competitions") or []
        rows = "".join(
            f"""
            <li>
              <span>{html.escape(str(item.get("ref", "unknown")))}</span>
              <strong>{html.escape(str(item.get("category", "unknown")))} · {html.escape(str(item.get("deadline", "no deadline")))}</strong>
            </li>
            """
            for item in competitions[:8]
        ) or "<li><span>暂无题目。</span><strong>请刷新</strong></li>"
        issues = cache.get("issues") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无问题。</li>"
        first_slug = competitions[0].get("ref") if competitions else "<competition_slug>"
        status = str(cache.get("status", "unknown"))
        blocking_text = self._kaggle_blocking_text(status)
        next_command = self._kaggle_next_command(status, first_slug)
        return f"""
        <section class="kaggle-discovery">
          <header>
            <div>
              <p class="eyebrow">Kaggle Discovery Agent</p>
              <h2>Kaggle 题目池</h2>
            </div>
            <span class="status {html.escape(status)}">{html.escape(status)}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>缓存时间</span>
              <strong>{html.escape(str(cache.get("created_at", "unknown")))}</strong>
            </div>
            <div>
              <span>题目数量</span>
              <strong>{len(competitions)}</strong>
            </div>
            <div>
              <span>数据来源</span>
              <strong>kaggle competitions list</strong>
            </div>
            <div>
              <span>当前阻塞项</span>
              <strong>{html.escape(blocking_text)}</strong>
            </div>
          </div>
          <ul class="queue-list">{rows}</ul>
          <section class="handoff-command">
            <span>下一步命令</span>
            <pre><code>{html.escape(next_command)}</code></pre>
          </section>
          <section class="handoff-command">
            <span>选择题目并生成 Intake</span>
            <pre><code>python framework.py --kaggle-select {html.escape(str(first_slug))}</code></pre>
          </section>
          <section class="handoff-command">
            <span>刷新题目池</span>
            <pre><code>{html.escape(" ".join(str(part) for part in cache.get("command", ["python", "framework.py", "--kaggle-discover"])))}</code></pre>
          </section>
          <div class="handoff-notes">
            <section>
              <h3>发现问题</h3>
              <ul>{issue_items}</ul>
            </section>
          </div>
          <nav>
            <a href="../../kaggle_competitions_cache.json">题目池缓存</a>
          </nav>
        </section>
        """

    def _kaggle_blocking_text(self, status: str) -> str:
        mapping = {
            "needs_kaggle_cli": "需要安装 Kaggle CLI",
            "auth_missing": "需要配置 Kaggle API 认证",
            "rules_not_accepted": "需要在 Kaggle 网页接受竞赛规则",
            "downloading": "Kaggle 数据正在下载",
            "competition_not_found": "竞赛 slug 不存在或不可访问",
            "needs_review": "需要查看 Kaggle CLI 错误",
            "empty": "题目池为空，需要换筛选条件",
            "pass": "无阻塞",
        }
        return mapping.get(status, "待确认")

    def _kaggle_next_command(self, status: str, first_slug: str) -> str:
        if status == "auth_missing":
            return "cd /home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac && conda run -n mac kaggle auth login"
        if status == "needs_kaggle_cli":
            return "pip install kaggle"
        if status == "pass":
            return f"python framework.py --kaggle-select {first_slug}"
        return "python framework.py --kaggle-discover --kaggle-category featured --kaggle-sort-by recentlyCreated"

    def _render_competition_intake(self) -> str:
        intake = self._read_json(self.competition_dir / "competition_intake.json")
        if not intake:
            return ""
        files = intake.get("files") or []
        file_rows = "".join(
            f"""
            <li>
              <span>{html.escape(str(item.get("name") or item.get("Name") or item.get("fileName") or item.get("ref") or "unknown"))}</span>
              <strong>{html.escape(str(item.get("size") or item.get("Size") or item.get("totalBytes") or "n/a"))}</strong>
            </li>
            """
            for item in files[:8]
        ) or "<li><span>尚未读取文件列表。</span><strong>可能需要接受规则</strong></li>"
        issues = intake.get("issues") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无问题。</li>"
        intake_agent = intake.get("intake_agent") or {}
        unknown_fields = intake_agent.get("unknown_fields") or []
        blocking_items = intake_agent.get("blocking_items") or []
        unknown_items = "".join(f"<li>{html.escape(str(field))}</li>" for field in unknown_fields) or "<li>暂无 unknown 字段。</li>"
        blocking_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in blocking_items) or "<li>暂无阻塞项。</li>"
        commands = intake.get("recommended_commands") or []
        command_text = "\n".join(str(command) for command in commands)
        next_command = intake_agent.get("next_command") or command_text
        return f"""
        <section class="competition-intake">
          <header>
            <div>
              <p class="eyebrow">Competition Intake Agent</p>
              <h2>竞赛 Intake</h2>
            </div>
            <span class="status {html.escape(str(intake.get("status", "unknown")))}">{html.escape(str(intake.get("status", "unknown")))}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>Kaggle Slug</span>
              <strong>{html.escape(str(intake.get("competition_slug", self.competition_dir.name)))}</strong>
            </div>
            <div>
              <span>下一步</span>
              <strong>{html.escape(str(intake.get("next_step", "unknown")))}</strong>
            </div>
            <div>
              <span>是否请求下载</span>
              <strong>{html.escape(str(intake.get("download_requested", False)))}</strong>
            </div>
            <div>
              <span>unknown 字段数</span>
              <strong>{len(unknown_fields)}</strong>
            </div>
          </div>
          <ul class="queue-list">{file_rows}</ul>
          <section class="handoff-command">
            <span>下一步命令</span>
            <pre><code>{html.escape(str(next_command))}</code></pre>
          </section>
          <section class="handoff-command">
            <span>推荐命令</span>
            <pre><code>{html.escape(command_text)}</code></pre>
          </section>
          <div class="handoff-notes">
            <section>
              <h3>Intake 问题</h3>
              <ul>{issue_items}</ul>
            </section>
            <section>
              <h3>unknown 字段</h3>
              <ul>{unknown_items}</ul>
            </section>
            <section>
              <h3>当前阻塞项</h3>
              <ul>{blocking_rows}</ul>
            </section>
          </div>
          <nav>
            <a href="../competition_intake.json">Intake JSON</a>
          </nav>
        </section>
        """

    def _render_submission_decision(self) -> str:
        review = self._read_json(self.competition_dir / "submission_decision_review.json")
        if not review:
            return ""
        issues = review.get("issues") or []
        warnings = review.get("warnings") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无阻塞问题。</li>"
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) or "<li>暂无警告。</li>"
        decision = review.get("decision", "unknown")
        status = review.get("status", "unknown")
        stability = review.get("cv_stability_audit") or {}
        return f"""
        <section class="submission-decision">
          <header>
            <div>
              <p class="eyebrow">Submission Decision</p>
              <h2>人工提交审核</h2>
            </div>
            <span class="status {html.escape(str(status))}">{html.escape(str(status))}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>队列任务</span>
              <strong>{html.escape(str(review.get("queue_task_id", "unknown")))}</strong>
            </div>
            <div>
              <span>决策</span>
              <strong>{html.escape(str(decision))}</strong>
            </div>
            <div>
              <span>提交目标</span>
              <strong>{html.escape(str(review.get("submission_target", "unknown")))}</strong>
            </div>
            <div>
              <span>稳定性风险</span>
              <strong>{html.escape(str(stability.get("risk_level", "unknown")))}</strong>
            </div>
          </div>
          <div class="handoff-notes">
            <section>
              <h3>阻塞问题</h3>
              <ul>{issue_items}</ul>
            </section>
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
          </div>
          <nav>
            <a href="../submission_decision_review.json">审核 JSON</a>
            <a href="../submission_decision_review.md">审核 Markdown</a>
            <a href="../experiments/cv_stability_audit_v1/cv_stability_audit.json">稳定性审计</a>
          </nav>
        </section>
        """

    def _render_promotion_gate(self) -> str:
        review = self._read_json(self.competition_dir / "promotion_gate_review.json")
        if not review:
            return ""
        promoted = review.get("promoted_candidate") or {}
        evaluations = review.get("evaluations") or []
        rows = "".join(
            f"""
            <li>
              <span>{html.escape(str(item.get("task_id", "unknown")))}</span>
              <strong>{html.escape(str(item.get("decision", "unknown")))} · {html.escape(str(item.get("runner_kind", "unknown")))}</strong>
            </li>
            """
            for item in evaluations[:6]
        )
        if not rows:
            rows = "<li><span>暂无已评估候选。</span><strong>needs review</strong></li>"
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in (review.get("issues") or [])[:6]) or "<li>暂无阻塞问题。</li>"
        return f"""
        <section class="promotion-gate">
          <header>
            <div>
              <p class="eyebrow">Promotion Gate</p>
              <h2>候选晋级审核</h2>
            </div>
            <span class="status {html.escape(str(review.get("status", "unknown")))}">{html.escape(str(review.get("status", "unknown")))}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>决策</span>
              <strong>{html.escape(str(review.get("decision", "unknown")))}</strong>
            </div>
            <div>
              <span>晋级任务</span>
              <strong>{html.escape(str(promoted.get("task_id", "none")))}</strong>
            </div>
            <div>
              <span>分数</span>
              <strong>{html.escape(str(promoted.get("local_score", "n/a")))}</strong>
            </div>
            <div>
              <span>指标方向</span>
              <strong>{html.escape(str(review.get("metric_direction", "unknown")))}</strong>
            </div>
          </div>
          <ul class="queue-list">{rows}</ul>
          <div class="handoff-notes">
            <section>
                <h3>晋级问题</h3>
              <ul>{issue_items}</ul>
            </section>
          </div>
          <nav>
            <a href="../promotion_gate_review.json">审核 JSON</a>
            <a href="../promotion_gate_review.md">审核 Markdown</a>
            <a href="../promoted_submission.csv">晋级 submission</a>
          </nav>
        </section>
        """

    def _render_submit_decision_handoff(self) -> str:
        handoff = self._read_json(self.competition_dir / "submit_decision_handoff.json")
        if not handoff:
            return ""
        candidate = handoff.get("candidate") or {}
        evidence = handoff.get("evidence_summary") or {}
        issues = handoff.get("issues") or []
        warnings = handoff.get("warnings") or []
        evidence_items = "".join(
            f"<li><span>{html.escape(str(key))}</span><strong>{html.escape(str(value))}</strong></li>"
            for key, value in evidence.items()
        ) or "<li><span>暂无证据字段。</span><strong>needs review</strong></li>"
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无阻塞问题。</li>"
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) or "<li>暂无警告。</li>"
        command = handoff.get("post_submit_workflow_command", "")
        return f"""
        <section class="submit-decision-handoff">
          <header>
            <div>
              <p class="eyebrow">Submit Decision Handoff</p>
              <h2>人工榜单提交决策</h2>
            </div>
            <span class="status {html.escape(str(handoff.get("status", "unknown")))}">{html.escape(str(handoff.get("status", "unknown")))}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>决策</span>
              <strong>{html.escape(str(handoff.get("decision", "unknown")))}</strong>
            </div>
            <div>
              <span>提交目标</span>
              <strong>{html.escape(str(handoff.get("submission_target", "unknown")))}</strong>
            </div>
            <div>
              <span>候选任务</span>
              <strong>{html.escape(str(candidate.get("task_id", "unknown")))}</strong>
            </div>
            <div>
              <span>本地分数</span>
              <strong>{html.escape(str(candidate.get("local_score", "n/a")))}</strong>
            </div>
          </div>
          <section class="handoff-file">
            <span>人工上传文件</span>
            <code>{html.escape(str(handoff.get("submission_path", "unknown")))}</code>
          </section>
          <ul class="queue-list">{evidence_items}</ul>
          <section class="handoff-command">
            <span>上传后执行命令</span>
            <pre><code>{html.escape(str(command))}</code></pre>
          </section>
          <div class="handoff-notes">
            <section>
              <h3>阻塞问题</h3>
              <ul>{issue_items}</ul>
            </section>
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
          </div>
          <nav>
            <a href="../submit_decision_handoff.json">交接 JSON</a>
            <a href="../submit_decision_handoff.md">交接 Markdown</a>
            <a href="../manual_submit_readiness.json">就绪检查</a>
            <a href="../submission_policy.json">提交策略</a>
            <a href="../recommended_submission.csv">推荐 submission</a>
          </nav>
        </section>
        """

    def _render_leaderboard_feedback_loop(self) -> str:
        loop = self._read_json(self.competition_dir / "leaderboard_feedback_loop.json")
        if not loop:
            return ""
        next_runnable = loop.get("next_runnable") or {}
        score_gap = loop.get("score_gap") or {}
        issues = loop.get("issues") or []
        warnings = loop.get("warnings") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无阻塞问题。</li>"
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) or "<li>暂无警告。</li>"
        command = loop.get("next_command") or ""
        return f"""
        <section class="leaderboard-feedback-loop">
          <header>
            <div>
              <p class="eyebrow">Leaderboard Feedback Loop</p>
              <h2>提交后 Brain 决策</h2>
            </div>
            <span class="status {html.escape(str(loop.get("status", "unknown")))}">{html.escape(str(loop.get("status", "unknown")))}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>决策</span>
              <strong>{html.escape(str(loop.get("decision", "unknown")))}</strong>
            </div>
            <div>
              <span>差距风险</span>
              <strong>{html.escape(str(loop.get("gap_risk_level", "unknown")))}</strong>
            </div>
            <div>
              <span>公开榜分数</span>
              <strong>{html.escape(str(score_gap.get("public_score", "n/a")))}</strong>
            </div>
            <div>
              <span>下一可执行任务</span>
              <strong>{html.escape(str(next_runnable.get("task_id", "none")))}</strong>
            </div>
          </div>
          <section class="handoff-command">
            <span>下一步命令</span>
            <pre><code>{html.escape(str(command or loop.get("next_action", "")))}</code></pre>
          </section>
          <div class="handoff-notes">
            <section>
              <h3>阻塞问题</h3>
              <ul>{issue_items}</ul>
            </section>
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
          </div>
          <nav>
            <a href="../leaderboard_feedback_loop.json">反馈闭环</a>
            <a href="../leaderboard_gap_audit.json">差距审计</a>
            <a href="../llm_experiment_plan.json">Brain 计划</a>
            <a href="../experiment_queue.json">实验队列</a>
          </nav>
        </section>
        """

    def _render_feedback_template_fill(self) -> str:
        fill = self._read_json(self.competition_dir / "leaderboard_feedback_template_fill.json")
        if not fill:
            return ""
        issues = fill.get("issues") or []
        warnings = fill.get("warnings") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无填写问题。</li>"
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) or "<li>暂无警告。</li>"
        return f"""
        <section class="feedback-template-fill">
          <header>
            <div>
              <p class="eyebrow">Leaderboard Score Input</p>
              <h2>榜单分数回填</h2>
            </div>
            <span class="status {html.escape(str(fill.get("status", "unknown")))}">{html.escape(str(fill.get("status", "unknown")))}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>公开榜分数</span>
              <strong>{html.escape(str(fill.get("public_score", "n/a")))}</strong>
            </div>
            <div>
              <span>榜单排名</span>
              <strong>{html.escape(str(fill.get("leaderboard_rank", "n/a")))}</strong>
            </div>
            <div>
              <span>Submission ID</span>
              <strong>{html.escape(str(fill.get("submission_id", "n/a")))}</strong>
            </div>
            <div>
              <span>候选任务</span>
              <strong>{html.escape(str(fill.get("candidate_task_id", "n/a")))}</strong>
            </div>
            <div>
              <span>预期 SHA</span>
              <strong>{html.escape(str(fill.get("expected_submission_sha256", "n/a")))}</strong>
            </div>
            <div>
              <span>是否已跑反馈闭环</span>
              <strong>{html.escape(str(bool(fill.get("feedback_loop_report_path"))))}</strong>
            </div>
          </div>
          <div class="handoff-notes">
            <section>
              <h3>填写问题</h3>
              <ul>{issue_items}</ul>
            </section>
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
          </div>
          <nav>
            <a href="../leaderboard_feedback_template_fill.json">回填报告</a>
            <a href="../leaderboard_feedback_loop.json">反馈闭环</a>
            <a href="../experiment_roadmap.json">路线图</a>
          </nav>
        </section>
        """

    def _render_experiment_queue(self) -> str:
        queue = self._read_json(self.competition_dir / "experiment_queue.json")
        if not queue:
            return ""
        items = queue.get("queue") or []
        item_rows = "".join(
            f"""
            <li>
              <span>{html.escape(str(item.get("order")))}. {html.escape(str(item.get("task_id", "unknown")))}</span>
              <strong>{html.escape(str(item.get("status", "unknown")))} · {html.escape(str(item.get("action_type", "unknown")))}</strong>
            </li>
            """
            for item in items[:6]
        )
        if not item_rows:
            item_rows = "<li><span>暂无队列任务。</span><strong>needs review</strong></li>"
        next_runnable = queue.get("next_runnable") or {}
        next_text = next_runnable.get("task_id") or "none"
        next_command = next_runnable.get("next_command") or ""
        return f"""
        <section class="experiment-queue">
          <header>
            <div>
              <p class="eyebrow">Remote Brain Queue</p>
              <h2>下一批实验队列</h2>
            </div>
            <span class="status {html.escape(str(queue.get("status", "unknown")))}">{html.escape(str(queue.get("status", "unknown")))}</span>
          </header>
          <div class="queue-next">
            <span>下一可执行任务</span>
            <strong>{html.escape(str(next_text))}</strong>
          </div>
          <ul class="queue-list">{item_rows}</ul>
          <section class="handoff-command">
            <span>建议命令</span>
            <pre><code>{html.escape(str(next_command))}</code></pre>
          </section>
          <nav>
            <a href="../experiment_queue.json">队列 JSON</a>
            <a href="../experiment_queue.md">队列 Markdown</a>
            <a href="../llm_experiment_plan.json">Brain 计划</a>
          </nav>
        </section>
        """

    def _render_experiment_roadmap(self) -> str:
        roadmap = self._read_json(self.competition_dir / "experiment_roadmap.json")
        if not roadmap:
            return ""
        items = roadmap.get("items") or []
        item_rows = "".join(
            f"""
            <li>
              <span>P{html.escape(str(item.get("priority", "?")))} · {html.escape(str(item.get("action_id", "unknown")))}</span>
              <strong>{html.escape(str(item.get("status", "unknown")))} · {html.escape(str(item.get("owner_agent", "unknown")))}</strong>
            </li>
            """
            for item in items[:6]
        )
        if not item_rows:
            item_rows = "<li><span>暂无路线图任务。</span><strong>needs review</strong></li>"
        top = roadmap.get("top_action") or {}
        if top.get("status") == "waiting_for_human" and top.get("next_command"):
            command = top.get("next_command")
        else:
            command = roadmap.get("next_action") or top.get("next_command") or ""
        freshness_panel = self._render_feedback_freshness_panel(
            roadmap.get("leaderboard_feedback_freshness") or {}
        )
        package_verification_panel = self._render_package_verification_panel(
            roadmap.get("manual_submission_package_verification") or {}
        )
        return f"""
        <section class="experiment-roadmap">
          <header>
            <div>
              <p class="eyebrow">Brain Backlog</p>
              <h2>实验路线图</h2>
            </div>
            <span class="status {html.escape(str(roadmap.get("status", "unknown")))}">{html.escape(str(roadmap.get("status", "unknown")))}</span>
          </header>
          <div class="queue-next">
            <span>最高优先级动作</span>
            <strong>{html.escape(str(top.get("title", "none")))}</strong>
          </div>
          {package_verification_panel}
          {freshness_panel}
          <ul class="queue-list">{item_rows}</ul>
          <section class="handoff-command">
            <span>建议命令</span>
            <pre><code>{html.escape(str(command))}</code></pre>
          </section>
          <nav>
            <a href="../experiment_roadmap.json">路线图 JSON</a>
            <a href="../experiment_roadmap.md">路线图 Markdown</a>
            <a href="../experiment_queue.json">队列 JSON</a>
            <a href="../manual_submission_package/manifest.json">手动提交包</a>
          </nav>
        </section>
        """

    def _render_package_verification_panel(self, verification: Dict[str, Any]) -> str:
        if not verification:
            return ""
        submission = verification.get("actual_submission_file") or {}
        issues = verification.get("issues") or []
        warnings = verification.get("warnings") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无提交包校验问题。</li>"
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
        warning_section = ""
        if warning_items:
            warning_section = f"""
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
            """
        return f"""
          <section class="package-verification">
            <div class="freshness-head">
              <span>手动提交包校验</span>
              <strong>{html.escape(str(verification.get("status", "unknown")))}</strong>
            </div>
            <div class="handoff-grid">
              <div>
                <span>决策</span>
                <strong>{html.escape(str(verification.get("decision", "unknown")))}</strong>
              </div>
              <div>
                <span>候选任务</span>
                <strong>{html.escape(str(verification.get("candidate_task_id", "n/a")))}</strong>
              </div>
              <div>
                <span>Submission SHA</span>
                <strong>{html.escape(str(submission.get("sha256", "n/a")))}</strong>
              </div>
              <div>
                <span>Submission 行数</span>
                <strong>{html.escape(str(submission.get("row_count", "n/a")))}</strong>
              </div>
            </div>
            <div class="handoff-notes">
              <section>
                <h3>提交包问题</h3>
                <ul>{issue_items}</ul>
              </section>
              {warning_section}
            </div>
          </section>
        """

    def _render_feedback_freshness_panel(self, freshness: Dict[str, Any]) -> str:
        if not freshness:
            return ""
        expected = freshness.get("expected") or {}
        actual = freshness.get("actual") or {}
        issues = freshness.get("issues") or []
        warnings = freshness.get("warnings") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>暂无反馈新鲜度问题。</li>"
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
        warning_section = ""
        if warning_items:
            warning_section = f"""
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
            """
        expected_columns = ", ".join(str(column) for column in (expected.get("expected_submission_columns") or [])) or "n/a"
        actual_columns = ", ".join(str(column) for column in (actual.get("expected_submission_columns") or [])) or "n/a"
        return f"""
          <section class="feedback-freshness">
            <div class="freshness-head">
              <span>Leaderboard 反馈新鲜度</span>
              <strong>{html.escape(str(freshness.get("status", "unknown")))}</strong>
            </div>
            <div class="handoff-grid">
              <div>
                <span>预期候选</span>
                <strong>{html.escape(str(expected.get("candidate_task_id", "n/a")))}</strong>
              </div>
              <div>
                <span>实际反馈</span>
                <strong>{html.escape(str(actual.get("candidate_task_id", "n/a")))}</strong>
              </div>
              <div>
                <span>预期 SHA</span>
                <strong>{html.escape(str(expected.get("expected_submission_sha256", "n/a")))}</strong>
              </div>
              <div>
                <span>实际 SHA</span>
                <strong>{html.escape(str(actual.get("expected_submission_sha256", "n/a")))}</strong>
              </div>
              <div>
                <span>预期行数</span>
                <strong>{html.escape(str(expected.get("expected_submission_rows", "n/a")))}</strong>
              </div>
              <div>
                <span>实际行数</span>
                <strong>{html.escape(str(actual.get("expected_submission_rows", "n/a")))}</strong>
              </div>
              <div>
                <span>预期列</span>
                <strong>{html.escape(expected_columns)}</strong>
              </div>
              <div>
                <span>实际列</span>
                <strong>{html.escape(actual_columns)}</strong>
              </div>
            </div>
            <div class="handoff-notes">
              <section>
                <h3>新鲜度问题</h3>
                <ul>{issue_items}</ul>
              </section>
              {warning_section}
            </div>
          </section>
        """

    def _render_submission_handoff(self) -> str:
        handoff = self._read_json(self.competition_dir / "post_submit_workflow.json")
        if not handoff:
            return ""
        candidate = handoff.get("candidate") or {}
        status = str(handoff.get("status", "unknown"))
        target = handoff.get("submission_target", "unknown")
        submission_path = handoff.get("submission_path", "unknown")
        command = handoff.get("feedback_loop_command_template", "")
        issues = handoff.get("issues") or []
        warnings = handoff.get("warnings") or []
        issue_items = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues)
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
        if not issue_items:
            issue_items = "<li>暂无阻塞问题。</li>"
        if not warning_items:
            warning_items = "<li>暂无警告。</li>"
        checklist = self.competition_dir / "post_submit_workflow.md"
        checklist_link = ""
        if checklist.exists():
            checklist_link = '<a href="../post_submit_workflow.md">检查清单</a>'
        feedback_template = self.competition_dir / "leaderboard_feedback_input_template.json"
        feedback_template_link = ""
        if feedback_template.exists():
            feedback_template_link = '<a href="../leaderboard_feedback_input_template.json">反馈输入模板</a>'
        return f"""
        <section class="submission-handoff">
          <header>
            <div>
              <p class="eyebrow">Current Submit Handoff</p>
              <h2>人工上传到反馈闭环</h2>
            </div>
            <span class="status {html.escape(status)}">{html.escape(status)}</span>
          </header>
          <div class="handoff-grid">
            <div>
              <span>提交目标</span>
              <strong>{html.escape(str(target))}</strong>
            </div>
            <div>
              <span>候选任务</span>
              <strong>{html.escape(str(candidate.get("task_id", "unknown")))}</strong>
            </div>
            <div>
              <span>评价指标</span>
              <strong>{html.escape(str(candidate.get("metric_name", "unknown")))}</strong>
            </div>
            <div>
              <span>本地分数</span>
              <strong>{html.escape(str(candidate.get("local_score", "n/a")))}</strong>
            </div>
          </div>
          <section class="handoff-file">
            <span>上传文件</span>
            <code>{html.escape(str(submission_path))}</code>
          </section>
          <section class="handoff-command">
            <span>记录 leaderboard 反馈</span>
            <pre><code>{html.escape(str(command))}</code></pre>
          </section>
          <div class="handoff-notes">
            <section>
              <h3>阻塞问题</h3>
              <ul>{issue_items}</ul>
            </section>
            <section>
              <h3>警告</h3>
              <ul>{warning_items}</ul>
            </section>
          </div>
          <nav>
            {checklist_link}
            {feedback_template_link}
            <a href="../post_submit_workflow.json">工作流 JSON</a>
            <a href="../manual_submit_readiness.json">就绪检查</a>
            <a href="../recommended_submission.csv">推荐 submission</a>
          </nav>
        </section>
        """

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _default_human_review() -> str:
        return """# 人工审核

decision: continue

允许的决策:
- continue
- rerun
- patch_prompt
- stop

备注:
"""

    @staticmethod
    def _empty_panel(title: str, message: str) -> str:
        return f"""
        <section class="empty-panel">
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(message)}</p>
        </section>
        """

    @staticmethod
    def _competition_nav(*, active: str, root_prefix: str) -> str:
        links = [
            ("overview", "Overview", f"{root_prefix}index.html"),
            ("data", "Data", f"{root_prefix}console/data.html"),
            ("brain", "Brain Plan", f"{root_prefix}console/brain.html"),
            ("experiments", "Experiments", f"{root_prefix}console/experiments.html"),
            ("submission", "Submission", f"{root_prefix}console/submission.html"),
        ]
        return "\n".join(
            f'<a class="{"active" if key == active else ""}" href="{html.escape(href)}">{html.escape(label)}</a>'
            for key, label, href in links
        )

    @staticmethod
    def _html_template(*, competition_name: str, active: str, body: str, root_prefix: str, total: int) -> str:
        global_href = f"{root_prefix}../../index.html"
        nav = RunLedger._competition_nav(active=active, root_prefix=root_prefix)
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Competition 控制台：{html.escape(competition_name)}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f8fa;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      margin-bottom: 24px;
      border-bottom: 1px solid #d8dee8;
      padding-bottom: 18px;
    }}
    .competition-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: -10px 0 22px;
      border-bottom: 1px solid #d8dee8;
      padding-bottom: 14px;
    }}
    .competition-nav a {{
      border: 0;
      border-radius: 999px;
      padding: 7px 12px;
      color: #52606d;
      background: #ffffff;
      border: 1px solid #d8dee8;
      font-weight: 600;
    }}
    .competition-nav a.active, .competition-nav a:hover {{
      color: #008abc;
      background: #e8f7fd;
      border-color: #b8e8fb;
    }}
    .breadcrumb {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #008abc;
      font-size: 14px;
      font-weight: 600;
      text-decoration: none;
      border-bottom: 0;
      margin-bottom: 14px;
    }}
    h1, h2, h3, p {{
      margin: 0;
    }}
    h1 {{
      font-size: 28px;
      line-height: 1.2;
    }}
    .summary {{
      color: #52606d;
      margin-top: 6px;
    }}
    .count {{
      font-size: 14px;
      color: #52606d;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .control-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .summary-card {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 16px;
    }}
    .summary-card span {{
      display: block;
      color: #697586;
      font-size: 13px;
      margin-bottom: 7px;
    }}
    .summary-card strong {{
      display: block;
      font-size: 15px;
      line-height: 1.35;
    }}
    .competition-intake, .submit-decision-handoff, .feedback-template-fill, .leaderboard-feedback-loop, .submission-handoff, .experiment-queue, .experiment-roadmap, .submission-decision, .promotion-gate, .experiment-board, .audit-log {{
      background: #ffffff;
      border: 1px solid #b9c7d6;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .empty-panel {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
    }}
    .empty-panel p {{
      color: #697586;
      margin-top: 6px;
    }}
    .competition-intake header, .submit-decision-handoff header, .feedback-template-fill header, .leaderboard-feedback-loop header, .submission-handoff header, .experiment-queue header, .experiment-roadmap header, .submission-decision header, .promotion-gate header, .experiment-board header, .audit-log header {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .handoff-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .handoff-grid div, .handoff-file {{
      background: #f7f8fa;
      border: 1px solid #edf0f4;
      border-radius: 6px;
      padding: 10px;
    }}
    .handoff-grid span, .handoff-file span, .handoff-command span {{
      display: block;
      color: #697586;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .handoff-grid strong, .handoff-file code {{
      font-size: 14px;
      word-break: break-word;
    }}
    .handoff-command {{
      margin-top: 12px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 6px;
      padding: 12px;
      overflow-x: auto;
    }}
    .handoff-notes {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .queue-next {{
      background: #f7f8fa;
      border: 1px solid #edf0f4;
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 12px;
    }}
    .queue-next span {{
      display: block;
      color: #697586;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .feedback-freshness, .package-verification {{
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 14px;
      margin: 12px 0;
      background: #fbfcfd;
    }}
    .freshness-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .freshness-head span {{
      color: #52606d;
      font-size: 13px;
    }}
    .freshness-head strong {{
      font-size: 15px;
      text-transform: uppercase;
    }}
    .queue-list {{
      list-style: none;
      padding-left: 0;
      display: grid;
      gap: 6px;
    }}
    .queue-list li {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid #edf0f4;
      padding-top: 6px;
    }}
    .status-summary div div {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 2px 0;
    }}
    .board-summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .board-summary div {{
      border: 1px solid #edf0f4;
      border-radius: 6px;
      background: #f7f8fa;
      padding: 10px;
    }}
    .board-summary span {{
      display: block;
      color: #697586;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .board-summary strong {{
      font-size: 14px;
      word-break: break-word;
    }}
    .experiment-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}
    .experiment-table th, .experiment-table td {{
      border-top: 1px solid #edf0f4;
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    .experiment-table th {{
      color: #697586;
      font-size: 12px;
      font-weight: 600;
      background: #fbfcfd;
    }}
    .experiment-table small {{
      display: block;
      color: #697586;
      margin-top: 3px;
    }}
    .experiment-table a {{
      display: inline-block;
      margin-right: 6px;
    }}
    .audit-group {{
      border: 1px solid #d8dee8;
      border-radius: 8px;
      margin-top: 12px;
      background: #fbfcfd;
      overflow: hidden;
    }}
    .audit-group summary {{
      cursor: pointer;
      padding: 12px 14px;
      color: #334e68;
      font-weight: 700;
      background: #f7f8fa;
      border-bottom: 1px solid #edf0f4;
    }}
    .audit-group:not([open]) summary {{
      border-bottom: 0;
    }}
    .audit-group .grid {{
      padding: 14px;
    }}
    .run-card {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 18px;
    }}
    .run-card header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .eyebrow {{
      color: #697586;
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    h2 {{
      font-size: 18px;
      line-height: 1.3;
    }}
    h3 {{
      font-size: 14px;
      margin: 14px 0 8px;
    }}
    .status {{
      border-radius: 999px;
      border: 1px solid #c8d1dc;
      padding: 4px 9px;
      font-size: 12px;
      color: #334e68;
      background: #f0f4f8;
    }}
    .status.pass, .status.validated, .status.ready_for_manual_submit, .status.ready_for_human_submit_decision, .status.ready {{
      border-color: #a7d8b8;
      background: #edf8f0;
      color: #1f6f43;
    }}
    .status.needs_review, .status.validation_failed {{
      border-color: #f4c27a;
      background: #fff7e8;
      color: #8a4b00;
    }}
    dl {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      margin: 0;
    }}
    dl div {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid #edf0f4;
      padding-top: 6px;
    }}
    dt {{
      color: #697586;
    }}
    dd {{
      margin: 0;
      text-align: right;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .scores {{
      padding-left: 0;
      list-style: none;
    }}
    .scores li {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 4px 0;
    }}
    .metric-strip {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }}
    .metric-strip div {{
      background: #f7f8fa;
      border: 1px solid #edf0f4;
      border-radius: 6px;
      padding: 10px;
    }}
    .metric-strip span {{
      display: block;
      color: #697586;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric-strip strong {{
      font-size: 14px;
      word-break: break-word;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }}
    a {{
      color: #075985;
      text-decoration: none;
      border-bottom: 1px solid #bae6fd;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <a class="breadcrumb" href="{html.escape(global_href)}">← 返回 AutoKaggle 总控制台</a>
    <section class="topbar">
      <div>
        <h1>Competition 控制台：{html.escape(competition_name)}</h1>
        <p class="summary">当前竞赛的 intake、baseline、Brain 计划、远端实验、提交交接与反馈闭环。</p>
      </div>
      <p class="count">已记录 {total} 次运行</p>
    </section>
    <nav class="competition-nav">
      {nav}
    </nav>
    {body}
  </main>
</body>
</html>
"""
