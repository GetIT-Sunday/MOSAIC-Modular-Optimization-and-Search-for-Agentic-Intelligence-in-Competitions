# AutoKaggle 用户配置说明

本项目推荐每个用户使用自己的本地控制端配置：

```text
本地 Mac / Linux 控制端
  -> SSH 到远端 Linux 执行服务器
  -> 远端固定 workspace 内运行实验
  -> 本地查看控制台 HTML
```

## 1. 复制私有配置

```bash
cp autokaggle_config.example.json autokaggle_config.json
```

`autokaggle_config.json` 已加入 `.gitignore`，不要提交。

## 2. 配置远端执行服务器

在 `autokaggle_config.json` 中填写：

```json
{
  "remote": {
    "ssh_alias": "dev",
    "workspace": "/home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac",
    "project_subdir": "workspaces/AutoKaggle",
    "conda_env": "mac"
  }
}
```

要求：

- `ssh_alias` 必须能直接运行，例如 `ssh dev`。
- `workspace` 必须是允许操作的远端硬边界。
- `project_subdir` 会拼成远端项目路径：`<workspace>/<project_subdir>`。
- `conda_env` 是远端 workspace 内使用的环境名。

## 3. 配置 LLM

在 `autokaggle_config.json` 中填写模型和 base URL：

```json
{
  "llm": {
    "openai_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
    "api_key_env": "AUTOKAGGLE_API_KEY",
    "api_key_file": "api_key.txt",
    "planner_model": "mimo-v2.5-pro",
    "coding_model": "mimo-v2.5",
    "cheap_model": "mimo-v2.5"
  }
}
```

API key 二选一：

```bash
export AUTOKAGGLE_API_KEY="..."
```

或在本地创建：

```text
api_key.txt
```

`api_key.txt` 已加入 `.gitignore`，不要提交，不会写入控制台 HTML。

## 4. 检查配置

```bash
python framework.py --config-check
```

然后重新生成控制台：

```bash
python framework.py --project-control-panel
```

打开：

```text
multi_agents/competition/console/config.html
```

## 5. 远端 Kaggle 认证

每个远端 workspace 都需要自己的 Kaggle auth。当前推荐在远端执行：

```bash
cd /home/dataset-local/data_local/wengchuangchuang/BioLLM/Mac
"$PWD/.conda/envs/mac/bin/kaggle" auth login
```

如果用户使用不同 workspace，请使用自己的 `workspace` 和 `conda_env`。

## 安全原则

- 不提交 `autokaggle_config.json`、`api_key.txt`、`.env`、`.kaggle/`。
- 控制台只显示密钥状态，不显示密钥值。
- 远端实验只允许在配置的 workspace 下运行。
- 真实 Kaggle submit 继续保留 Human Gate。
