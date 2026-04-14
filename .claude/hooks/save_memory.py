#!/usr/bin/env python3
"""
LatentMind — Stop hook
Fires after every Claude response. Reads the latest exchange from the
transcript and saves it to the LatentMind store.

Receives JSON on stdin:
  {
    "session_id": "...",
    "transcript_path": "/path/to/conversation.jsonl",
    "hook_event_name": "Stop",
    "stop_hook_active": false
  }
"""

import json
import sys
from pathlib import Path


def extract_text(content) -> str:
    """Pull plain text out of a content field (string or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def main():
    raw = json.load(sys.stdin)

    if raw.get("stop_hook_active"):
        sys.exit(0)

    transcript_path = raw.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)

    lines = Path(transcript_path).read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        sys.exit(0)

    user_text      = None
    assistant_text = None

    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Transcript format: entry.type = "user"|"assistant", content at entry.message.content
        role    = entry.get("type", "")
        message = entry.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        text    = extract_text(content)

        if not text:
            continue

        if role == "assistant" and assistant_text is None:
            assistant_text = text
        elif role == "user" and user_text is None:
            user_text = text

        if user_text and assistant_text:
            break

    if not user_text and not assistant_text:
        sys.exit(0)

    try:
        from latentmind import LatentMind
    except ImportError:
        sys.exit(0)

    mind       = LatentMind()
    session_id = raw.get("session_id", "unknown")

    if user_text:
        mind.add(user_text, metadata={"role": "user", "session_id": session_id, "source": "hook"})

    if assistant_text:
        mind.add(assistant_text, metadata={"role": "assistant", "session_id": session_id, "source": "hook"})

    sys.exit(0)


if __name__ == "__main__":
    main()
