#!/usr/bin/env python3
"""
Scrape ChatGPT conversation export JSON files into one markdown file per conversation.
Uses ijson for streaming parsing — never loads the full file into memory.
"""

import ijson
import re
import sys
from datetime import datetime
from pathlib import Path


def safe_filename(title: str, conv_id: str) -> str:
    title = title or "untitled"
    slug = re.sub(r"[^\w\s-]", "", title).strip()
    slug = re.sub(r"[\s]+", "_", slug)[:80]
    return f"{slug}__{conv_id[:8]}.md"


def format_ts(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def extract_conversations(filepath: Path):
    """
    Yields dicts: {id, title, create_time, messages: [{role, text, create_time}]}
    
    ijson path: each conversation is `item`, messages are inside `item.mapping.*.message`.
    We stream prefix/event/value triples and assemble state manually.
    """
    with open(filepath, "rb") as f:
        parser = ijson.parse(f)

        conv = None
        # Track nesting path segments
        path_stack = []

        # State for current message being assembled
        msg = None
        in_parts = False
        part_depth = 0

        for prefix, event, value in parser:
            # ── Conversation boundary ──────────────────────────────────────
            if prefix == "item" and event == "start_map":
                conv = {"id": None, "title": None, "create_time": None, "messages": []}

            elif prefix == "item" and event == "end_map":
                if conv:
                    yield conv
                conv = None

            # ── Top-level conversation fields ──────────────────────────────
            elif prefix == "item.id" and event == "string":
                if conv is not None:
                    conv["id"] = value

            elif prefix == "item.title" and event == "string":
                if conv is not None:
                    conv["title"] = value

            elif prefix == "item.create_time" and event in ("number", "string"):
                if conv is not None:
                    conv["create_time"] = value

            # ── Message start ──────────────────────────────────────────────
            elif prefix.endswith(".message") and event == "start_map":
                msg = {"role": None, "text": [], "create_time": None}

            elif prefix.endswith(".message") and event == "end_map":
                if msg and conv is not None and msg["role"] in ("user", "assistant"):
                    text = "".join(msg["text"]).strip()
                    if text:
                        conv["messages"].append({
                            "role": msg["role"],
                            "text": text,
                            "create_time": msg["create_time"],
                        })
                msg = None
                in_parts = False

            # ── Author role ────────────────────────────────────────────────
            elif prefix.endswith(".message.author.role") and event == "string":
                if msg is not None:
                    msg["role"] = value

            # ── Message create_time ────────────────────────────────────────
            elif prefix.endswith(".message.create_time") and event in ("number", "string"):
                if msg is not None:
                    msg["create_time"] = value

            # ── Parts array ────────────────────────────────────────────────
            elif prefix.endswith(".message.content.parts") and event == "start_array":
                in_parts = True

            elif prefix.endswith(".message.content.parts") and event == "end_array":
                in_parts = False

            elif in_parts and event == "string" and msg is not None:
                msg["text"].append(value)


def write_markdown(conv: dict, output_dir: Path):
    if not conv.get("id"):
        return
    filename = safe_filename(conv.get("title"), conv["id"])
    filepath = output_dir / filename

    lines = [f"# {conv.get('title') or 'Untitled'}\n"]
    if conv.get("create_time"):
        lines.append(f"*{format_ts(conv['create_time'])}*\n\n")
    lines.append("---\n\n")

    for msg in conv["messages"]:
        ts = f" *({format_ts(msg['create_time'])})*" if msg.get("create_time") else ""
        role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role_label}{ts}\n\n")
        lines.append(msg["text"])
        lines.append("\n\n---\n\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


def run_scrape(export_dir: Path, output_dir: Path) -> dict:
    """Scrape all conversations-*.json in export_dir → markdown files in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(export_dir.glob("conversations-*.json"))
    total_convs = total_msgs = 0
    for input_file in input_files:
        for conv in extract_conversations(input_file):
            if conv.get("messages"):
                write_markdown(conv, output_dir)
                total_msgs += len(conv["messages"])
                total_convs += 1
    return {"conversations": total_convs, "messages": total_msgs}


def main():
    export_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else export_dir / "output"
    stats = run_scrape(export_dir, output_dir)
    print(f"Done. {stats['conversations']} conversations, "
          f"{stats['messages']} messages → {output_dir}/")


if __name__ == "__main__":
    main()
