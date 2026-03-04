# chatgpt-to-claude-mcp

> Turn your ChatGPT export into a personal context profile for Claude and GitHub Copilot.

Instead of manually writing a system prompt, this tool mines your actual conversation history — your word patterns, recurring topics, and habits of thought — and distils them into a ~3k character profile that meaningfully changes how an AI responds to you.

## How it works

The pipeline runs entirely locally. It never sends your conversation data anywhere.

1. **Stream** `conversations-*.json` using `ijson` — never loads the full file into memory
2. **Extract** user-only messages (assistant responses are just mirrors of your input)
3. **Score** lines by a weighted blend of word frequency relevance (60%) and Shannon entropy (40%), with context window expansion (±0–3 surrounding lines)
4. **Surface** the top proper nouns from your history — names, projects, companies — and ask which to exclude as noise (old employers, irrelevant contacts, etc.)
5. **Re-run** with updated banned phrases for a cleaner signal
6. **Synthesise** a profile via the LLM from the dense signal text
7. **Install** globally at `~/.config/profile.md` with symlinks for Claude (`~/.claude/CLAUDE.md`) and Copilot (`~/.copilot/copilot-instructions.md`)

Banned phrases are persisted to `{export_dir}/banned_phrases.json` so they survive reruns.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

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

Then restart Claude Desktop and say:

> "Build my profile from my ChatGPT export at ~/Downloads/chatgpt-export"

## Add to GitHub Copilot (VS Code)

`.vscode/mcp.json`:

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

## Install via pip

```bash
pip install chatgpt-to-claude-mcp
chatgpt-to-claude-mcp
```

## Tools

### `process_export(export_dir, banned_phrases?)`

Runs the full pipeline. Returns:
- `top_proper_nouns` — review these with the user, flag irrelevant ones
- `dense_signal` — scored excerpts for profile synthesis
- `synthesis_prompt` — prompt to use when synthesising the profile

### `save_profile(profile_content, export_dir)`

Writes the synthesised profile and installs symlinks.

## Getting your ChatGPT export

Settings → Data Controls → Export Data. You'll get a zip with `conversations-*.json` files.

## License

MIT
