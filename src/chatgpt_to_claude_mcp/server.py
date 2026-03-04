#!/usr/bin/env python3
"""
MCP server for chatgpt-to-claude.

Tools
-----
process_export(export_dir, banned_phrases, extra_signal_phrases)
    Run the full pipeline. Returns top proper nouns, top phrases, and
    current signal phrases for review. Does NOT return raw signal text.

read_signal(export_dir, max_lines)
    Read the dense signal file after the user has approved the config.
    Supports truncation via max_lines.

save_profile(profile_content, export_dir)
    Write the synthesised profile and install it globally.
"""

import shutil
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .scrape import run_scrape
from .extract import (
    run_extraction,
    load_banned_phrases, save_banned_phrases,
    load_signal_phrases, save_signal_phrases,
    SIGNAL_PHRASES,
)

mcp = FastMCP(
    "chatgpt-to-claude",
    instructions=(
        "Guide the user through building a personal context profile from their "
        "ChatGPT export. Use process_export first, then enter a review loop: "
        "show top_proper_nouns and top_phrases (to ban), then extra_signal_phrases "
        "(to add domain-specific signal). Re-run until the user approves. "
        "Then call read_signal, synthesise the profile, and call save_profile."
    ),
)

PROFILE_PATH = Path("~/.config/profile.md").expanduser()

_FORBIDDEN_PREFIXES = [
    Path("/etc"), Path("/usr"), Path("/bin"), Path("/sbin"),
    Path("/System"), Path("/private/etc"),
    Path.home() / ".ssh",
    Path.home() / ".gnupg",
    Path.home() / ".aws",
]

SYNTHESIS_PROMPT = """
Review the chat history excerpts below and synthesise a personal context profile.
Identify and capture:
- Core personality traits
- Communication style and register (sentence rhythm, humor, directness, signature phrases)
- Working style and habits
- Active projects and their current status
- Decision-making patterns and heuristics
- Preferences and hard dislikes
- Life circumstances (work situation, location, financial context, family if evident)

PRIVACY: Do NOT include full surnames, precise addresses, financial figures, third-party
personal details (family members beyond first names, colleagues, counterparties), or
anything that could identify or expose the user if this file were accidentally shared.
Generalise where needed (e.g. "based in Europe" not a city; "has a daughter" not her name).

OUTPUT REQUIREMENTS:
- ~3000 characters maximum. Hard limit. Every sentence must change how an AI responds.
- Specific and direct. Quote or paraphrase actual patterns from the text.
- Written in second person as a system prompt ("You are talking to [Name]...")
- Structure: Voice & Style / Beliefs & Mental Models / Active Context / Hard Rules
""".strip()


def _safe_resolve(path_str: str) -> Path | None:
    try:
        p = Path(path_str).expanduser().resolve()
    except Exception:
        return None
    for forbidden in _FORBIDDEN_PREFIXES:
        try:
            p.relative_to(forbidden)
            return None
        except ValueError:
            pass
    return p


def _read_top_freq(freq_path: Path, lines: int = 60) -> str:
    if not freq_path.exists():
        return ""
    return "\n".join(freq_path.read_text(encoding="utf-8").splitlines()[:lines])


@mcp.tool()
def process_export(
    export_dir: str,
    banned_phrases: list[str] | None = None,
    extra_signal_phrases: list[str] | None = None,
) -> dict:
    """
    Run the full pipeline on a ChatGPT export directory.

    Scrapes conversations-*.json → markdown, extracts user messages,
    builds frequency map, scores signal lines. Persists config to JSON files.

    Args:
        export_dir: Path to the directory containing conversations-*.json.
                    If unsure, ask — common location is ~/Downloads/chatgpt-export.
        banned_phrases: Proper nouns / phrases to suppress (old employers, irrelevant
                        contacts, noise tokens). Merged with banned_phrases.json.
        extra_signal_phrases: Domain-specific phrases to add as signal markers
                              (e.g. "my game", "my startup", "my client").
                              Merged with signal_phrases.json.
    """
    export_path = _safe_resolve(export_dir)
    if export_path is None:
        return {"error": f"Invalid or forbidden path: {export_dir}"}
    if not export_path.is_dir():
        return {"error": f"Directory not found: {export_path}"}

    json_files = list(export_path.glob("conversations-*.json"))
    if not json_files:
        return {
            "error": (
                f"No conversations-*.json files found in {export_path}. "
                "Export your ChatGPT data from Settings → Data Controls → Export Data, "
                "unzip, and provide that folder path."
            )
        }

    output_path    = export_path / "output"
    extracted_path = export_path / "extracted"

    # Merge + persist banned phrases
    existing_banned = load_banned_phrases(export_path)
    new_banned = {p.lower().strip() for p in (banned_phrases or []) if p.strip()}
    combined_banned = existing_banned | new_banned
    if combined_banned != existing_banned:
        save_banned_phrases(export_path, combined_banned)

    # Merge + persist extra signal phrases
    existing_signal = load_signal_phrases(export_path)
    new_signal = [p.lower().strip() for p in (extra_signal_phrases or []) if p.strip()]
    combined_signal = sorted(set(existing_signal) | set(new_signal))
    if set(combined_signal) != set(existing_signal):
        save_signal_phrases(export_path, combined_signal)

    scrape_stats  = run_scrape(export_path, output_path)
    extract_stats = run_extraction(output_path, extracted_path, combined_banned, combined_signal)

    return {
        "stats": {**scrape_stats, **extract_stats},
        "top_proper_nouns": extract_stats["top_proper_nouns"],
        "top_phrases": extract_stats["top_phrases"],
        "banned_phrases": sorted(combined_banned),
        "default_signal_phrases": SIGNAL_PHRASES,
        "extra_signal_phrases": combined_signal,
        "signal_lines": extract_stats["dense_signal_lines"],
        "signal_chars": extract_stats["dense_signal_chars"],
        "synthesis_prompt": SYNTHESIS_PROMPT,
        "next_step": (
            "REVIEW LOOP — complete all steps before proceeding:\n"
            "1. Show top_proper_nouns and top_phrases. Ask: 'Any to exclude? "
            "(irrelevant names, noise, private info)'\n"
            "2. Show extra_signal_phrases alongside default_signal_phrases. Ask: "
            "'Any domain-specific phrases to add as signal? "
            "(e.g. \"my game\", \"my startup\", \"my client\", project names)'\n"
            "3. If changes → call process_export again with updated lists. Repeat.\n"
            "4. When user approves → ask: 'Signal is {signal_lines} lines / "
            "{signal_chars} chars. Read all, truncate, or skip?' "
            "Then call read_signal, synthesise using synthesis_prompt, call save_profile."
        ),
    }


@mcp.tool()
def read_signal(
    export_dir: str,
    max_lines: int | None = None,
) -> dict:
    """
    Read the dense signal file for profile synthesis.

    Call after the user approves the ban list and signal phrases.
    Signal is scored highest-first — truncating from the bottom loses the least.

    Args:
        export_dir: Same directory passed to process_export.
        max_lines: Optional line limit. None = read all.
    """
    export_path = _safe_resolve(export_dir)
    if export_path is None:
        return {"error": f"Invalid or forbidden path: {export_dir}"}

    signal_path = export_path / "extracted" / "dense_signal.txt"
    if not signal_path.exists():
        return {"error": "dense_signal.txt not found. Run process_export first."}

    lines = signal_path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    if max_lines and max_lines < total:
        lines = lines[:max_lines]
        truncated = True
    else:
        truncated = False

    return {
        "dense_signal": "\n".join(lines),
        "lines_returned": len(lines),
        "lines_total": total,
        "truncated": truncated,
        "frequency_summary": _read_top_freq(export_path / "extracted" / "frequency.txt"),
        "synthesis_prompt": SYNTHESIS_PROMPT,
        "next_step": "Synthesise the profile from dense_signal using synthesis_prompt, then call save_profile.",
    }


@mcp.tool()
def save_profile(
    profile_content: str,
    export_dir: str,
) -> dict:
    """
    Save the synthesised profile and install it globally.

    Writes to:
    - {export_dir}/extracted/profile.md  (timestamped backup if exists)
    - ~/.config/profile.md               (global canonical)

    Args:
        profile_content: The synthesised profile text.
        export_dir: The export directory used in process_export.
    """
    export_path = _safe_resolve(export_dir)
    if export_path is None:
        return {"error": f"Invalid or forbidden path: {export_dir}"}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = export_path / "extracted"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_profile = local_dir / "profile.md"
    local_backup = None
    if local_profile.exists() and local_profile.read_text(encoding="utf-8") != profile_content:
        local_backup = local_dir / f"profile_{ts}.md"
        shutil.copy2(local_profile, local_backup)
    local_profile.write_text(profile_content, encoding="utf-8")

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    global_backup = None
    if PROFILE_PATH.exists() and not PROFILE_PATH.is_symlink() and PROFILE_PATH.read_text(encoding="utf-8") != profile_content:
        global_backup = PROFILE_PATH.with_stem(f"profile_{ts}")
        PROFILE_PATH.rename(global_backup)
    PROFILE_PATH.write_text(profile_content, encoding="utf-8")

    return {
        "saved_to": str(local_profile),
        "local_backup": str(local_backup) if local_backup else None,
        "global_backup": str(global_backup) if global_backup else None,
        "installed_at": str(PROFILE_PATH),
        "characters": len(profile_content),
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
