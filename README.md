# chatgpt-to-claude-mcp

Convert a ChatGPT conversation export into a personal context profile, then install it as a system prompt for Claude and GitHub Copilot.

## What it does

1. **Streams** your `conversations-*.json` export (never loads full file into memory)
2. **Extracts** all your messages and builds a frequency + entropy signal map
3. **Surfaces** top proper nouns for you to review — flag irrelevant names as banned phrases
4. **Re-runs** with updated filters so the profile focuses on what actually matters
5. **Synthesises** a ~3k character personal context profile via the LLM
6. **Installs** it globally at `~/.config/profile.md` with symlinks for Claude (`~/.claude/CLAUDE.md`) and Copilot (`~/.copilot/copilot-instructions.md`)

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Install & run

### Via `uvx` (no install needed)

```bash
uvx chatgpt-to-claude-mcp
```

### Via pip

```bash
pip install chatgpt-to-claude-mcp
chatgpt-to-claude-mcp
```

## Add to Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chatgpt-to-claude": {
      "command": "uvx",
      "args": ["chatgpt-to-claude-mcp"]
    }
  }
}
```

## Add to GitHub Copilot (VS Code)

`.vscode/mcp.json` or user-level Copilot MCP config:

```json
{
  "servers": {
    "chatgpt-to-claude": {
      "type": "stdio",
      "command": "uvx",
      "args": ["chatgpt-to-claude-mcp"]
    }
  }
}
```

## Usage

Once connected, ask the LLM:

> "Build my profile from my ChatGPT export at ~/Downloads/chatgpt-export"

The tool will guide you through:
1. Running the pipeline
2. Reviewing top proper nouns (flag irrelevant ones as banned)
3. Synthesising the profile
4. Installing it globally

## `banned_phrases.json`

The tool persists your exclusion list to `{export_dir}/banned_phrases.json` so it survives across runs. You can also edit it manually.

## License

MIT
