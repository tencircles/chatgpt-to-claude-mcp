# chatgpt-to-claude-mcp

> Turn your ChatGPT export into a personal context profile for Claude and GitHub Copilot.

Instead of manually writing a system prompt, this tool mines your actual conversation history — your word patterns, recurring topics, and habits of thought — and distils them into a ~3k character profile that meaningfully changes how an AI responds to you.

## How it works

The pipeline runs entirely locally. It never sends your conversation data anywhere.

1. **Stream** `conversations-*.json` using `ijson` — never loads the full file into memory
2. **Extract** user-only messages (assistant responses are just mirrors of your input)
3. **Score** lines by a weighted blend of word frequency relevance (60%) and Shannon entropy (40%), with context window expansion (±0–3 surrounding lines)
4. **Surface** top proper nouns and bigrams from your history — review and flag noise (old employers, irrelevant contacts, etc.)
5. **Optionally add** domain-specific signal phrases (e.g. `"my game"`, `"my startup"`) to boost relevant context
6. **Re-run** with updated config until the signal looks clean
7. **Read** the dense signal text, then synthesise a profile via the LLM
8. **Install** globally at `~/.config/profile.md`

Banned phrases persist to `{export_dir}/banned_phrases.json` and signal phrases to `{export_dir}/signal_phrases.json` so they survive reruns.

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

### `process_export(export_dir, banned_phrases?, extra_signal_phrases?)`

Runs the full pipeline. Returns:
- `top_proper_nouns` — flag irrelevant ones to ban
- `top_phrases` — top bigrams, also available for banning
- `default_signal_phrases` — built-in signal markers
- `extra_signal_phrases` — your persisted domain-specific additions
- `signal_lines` / `signal_chars` — size of the dense signal for review
- `synthesis_prompt` — prompt to use when synthesising

### `read_signal(export_dir, max_lines?)`

Read the dense signal file after approving the config. Signal is scored highest-first — truncating from the bottom loses the least. Returns the text and a frequency summary.

### `save_profile(profile_content, export_dir)`

Writes the synthesised profile to `~/.config/profile.md` (with timestamped backup if content has changed) and saves a local copy in `{export_dir}/extracted/profile.md`.

## Getting your ChatGPT export

Settings → Data Controls → Export Data. You'll get a zip with `conversations-*.json` files.

## License

MIT
