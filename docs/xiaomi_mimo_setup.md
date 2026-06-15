# Xiaomi MiMo API Setup

AutoKaggle can use Xiaomi MiMo through its OpenAI-compatible endpoint.

## API Credentials

Fill credentials in either environment variables or `api_key.txt`.

Recommended local file:

```text
YOUR_API_KEY_HERE
https://token-plan-sgp.xiaomimimo.com/v1
```

The file path is:

```text
api_key.txt
```

You can also create a local `.env` file:

```sh
AUTOKAGGLE_API_KEY=YOUR_API_KEY_HERE
AUTOKAGGLE_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1
```

## Model Names

Set Xiaomi model names with environment variables:

```sh
AUTOKAGGLE_READER_MODEL=<xiaomi-chat-model>
AUTOKAGGLE_PLANNER_MODEL=<xiaomi-coding-or-planning-model>
AUTOKAGGLE_DEVELOPER_MODEL=<xiaomi-coding-model>
AUTOKAGGLE_REVIEWER_MODEL=<xiaomi-chat-model>
AUTOKAGGLE_SUMMARIZER_MODEL=<xiaomi-chat-model>
```

If Xiaomi does not provide an embedding endpoint compatible with OpenAI embeddings, disable tool RAG:

```sh
AUTOKAGGLE_DISABLE_TOOL_RAG=1
```

Some OpenAI-compatible providers expect `max_tokens` instead of `max_completion_tokens`:

```sh
AUTOKAGGLE_TOKEN_PARAM=max_tokens
```

## Remote Run

After editing local config, sync to the fixed remote workspace:

```sh
scripts/sync_to_dev.sh
```

Then test from the remote `mac` conda environment:

```sh
scripts/remote_dev.sh 'cd workspaces/AutoKaggle && python -m compileall -q framework.py api_handler.py utils.py multi_agents'
```
