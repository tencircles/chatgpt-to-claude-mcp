#!/usr/bin/env python3
"""
MCP server for chatgpt-to-claude.
Exposes a guided pipeline that converts a ChatGPT export into a
personal context profile ready for use as a Claude/Copilot system prompt.

Tools
-----
process_export(export_dir, banned_phrases)
    Run the full scrape + extract pipeline. Returns top proper nouns
    and the dense signal text for the LLM to synthesise.

save_profile(profile_content, export_dir)
    Write the synthesised profile to the export dir and install it
    globally (~/.config/profile.md with symlinks for Claude + Copilot).
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .scrape import run_scrape
from .extract import run_extraction, load_banned_phrases, save_banned_phrases

mcp = FastMCP(
    "chatgpt-to-claude",
    instructions=(
        "Guide the user through building a personal context profile from their "
        "ChatGPT export. Use process_export first, then enter a review loop: "
        "show the user top_proper_nouns and top_phrases, ask them to flag anything "
        "irrelevant or sensitive, call process_export again with updated banned_phrases, "
        "repeat until the user approves the list. Then synthesise and call save_profile."
    ),
)

PROFILE_PATH = Path("~/.config/profile.md").expanduser()
COPILOT_INSTRUCTIONS = Path("~/.copilot/copilot-instructions.md").expanduser()
CLAUDE_INSTRUCTIONS = Path("~/.claude/CLAUDE.md").expanduser()

SYNTHESIS_PROMPT = """
Review the chat history excerpts below and synthesise a personal context profile.
Identify and capture:
- Core personality traits
- Communication style and register (how they actually write — sentence rhythm, humor, directness, signature phrases)
- Working style and habits
- Active projects and their current status
- Decision-making patterns and heuristics
- Preferences and hard dislikes
- Life circumstances (work situation, location, financial context, family if evident)

PRIVACY: Do NOT include full surnames, precise addresses, financial figures, third-party
personal details (family members' names beyond first names, colleagues, counterparties),
or anything that could identify or expose the user if this file were accidentally shared.
Generalise where needed (e.g. "based in Europe" not a specific city; "has a daughter" not her name).

OUTPUT REQUIREMENTS:
- ~3000 characters maximum. Hard limit. Every sentence must change how an AI responds.
- Specific and direct. Quote or paraphrase actual patterns from the text. No generic descriptions.
- Written in second person as a system prompt (e.g. "You are talking to [Name]...")
- Structure: Voice & Style / Beliefs & Mental Models / Active Context / Hard Rules
""".strip()


@mcp.tool()
def process_export(
    export_dir: str,
    banned_phrases: list[str] | None = None,
) -> dict:
    """
    Run the full pipeline on a ChatGPT export directory.

    Steps:
    1. Scrape conversations-*.json → markdown files
    2. Extract user messages, build frequency map, score signal lines
    3. Persist any new banned_phrases to banned_phrases.json

    Returns top proper nouns (for the user to review) and the dense
    signal text (for the LLM to synthesise the profile from).

    Args:
        export_dir: Path to the directory containing conversations-*.json files.
        banned_phrases: Names / phrases to suppress from scoring (e.g. old employers,
                        irrelevant contacts). Merged with any previously saved phrases.
    """
    export_path = Path(export_dir).expanduser().resolve()
    if not export_path.exists():
        return {"error": f"Directory not found: {export_path}"}

    output_path   = export_path / "output"
    extracted_path = export_path / "extracted"

    # Load persisted banned phrases and merge with any new ones
    existing_banned = load_banned_phrases(export_path)
    combined_banned = existing_banned | set(banned_phrases or [])
    if combined_banned != existing_banned:
        save_banned_phrases(export_path, combined_banned)

    # Run pipeline
    scrape_stats = run_scrape(export_path, output_path)
    extract_stats = run_extraction(output_path, extracted_path, combined_banned)

    # Read dense signal for synthesis
    dense_signal = (extracted_path / "dense_signal.txt").read_text(encoding="utf-8")
    freq_snippet = _read_top_freq(extracted_path / "frequency.txt", lines=60)

    return {
        "stats": {**scrape_stats, **extract_stats},
        "top_proper_nouns": extract_stats["top_proper_nouns"],
        "top_phrases": extract_stats["top_phrases"],
        "banned_phrases": sorted(combined_banned),
        "synthesis_prompt": SYNTHESIS_PROMPT,
        "dense_signal": dense_signal,
        "frequency_summary": freq_snippet,
        "next_step": (
            "REVIEW LOOP — do not skip:\n"
            "1. Show the user top_proper_nouns and top_phrases side by side.\n"
            "2. Ask: 'Do any of these look irrelevant, outdated, or sensitive? "
            "List anything you'd like to exclude, or say \"looks good\" to continue.'\n"
            "3. Also ask: 'Is there any private information you'd like kept out — "
            "full names, locations, financial details, people you'd rather not mention?'\n"
            "4. If the user adds bans → call process_export again with the updated "
            "banned_phrases list. Repeat from step 1.\n"
            "5. Only when the user explicitly approves → synthesise the profile from "
            "dense_signal using the synthesis_prompt, then call save_profile."
        ),
    }


@mcp.tool()
def save_profile(
    profile_content: str,
    export_dir: str,
) -> dict:
    """
    Save the synthesised profile and install it globally.

    Writes to:
    - {export_dir}/extracted/profile.md  (canonical)
    - ~/.config/profile.md               (global copy)
    - ~/.copilot/copilot-instructions.md (symlink → global)
    - ~/.claude/CLAUDE.md                (symlink → global)

    Args:
        profile_content: The synthesised profile text.
        export_dir: The export directory (for writing the local copy).
    """
    export_path = Path(export_dir).expanduser().resolve()
    local_profile = export_path / "extracted" / "profile.md"
    local_profile.parent.mkdir(parents=True, exist_ok=True)
    local_profile.write_text(profile_content, encoding="utf-8")

    # Global copy — backup existing file with timestamp before writing
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROFILE_PATH.exists() and not PROFILE_PATH.is_symlink():
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = PROFILE_PATH.with_stem(f"profile_{ts}")
        PROFILE_PATH.rename(backup)
        backed_up_to = str(backup)
    else:
        backed_up_to = None
    PROFILE_PATH.write_text(profile_content, encoding="utf-8")

    # Symlinks
    installed = [str(PROFILE_PATH)]
    for link in [COPILOT_INSTRUCTIONS, CLAUDE_INSTRUCTIONS]:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(PROFILE_PATH)
        installed.append(str(link))

    return {
        "saved_to": str(local_profile),
        "backed_up_to": backed_up_to,
        "installed_at": installed,
        "characters": len(profile_content),
    }


def _read_top_freq(freq_path: Path, lines: int = 60) -> str:
    if not freq_path.exists():
        return ""
    return "\n".join(freq_path.read_text(encoding="utf-8").splitlines()[:lines])


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
