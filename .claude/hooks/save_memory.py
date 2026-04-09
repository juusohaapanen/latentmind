#!/usr/bin/env python3
"""
LatentMind — Stop hook
Fires after every Claude response. Reads the latest exchange from the
transcript and saves it to the LatentMind hot buffer (ChromaDB).

Receives JSON on stdin:
  {
    "session_id": "...",
    "transcript_path": "/path/to/conversation.jsonl",
    "cwd": "/path/to/project",
    "hook_event_name": "Stop",
    "stop_hook_active": false
  }
"""

import json
import sys
from pathlib import Path

def main():
    raw = json.load(sys.stdin)

    # Prevent infinite loop — if we already fired once this stop, exit cleanly
    if raw.get("stop_hook_active"):
        sys.exit(0)

    transcript_path = raw.get("transcript_path")
    cwd             = raw.get("cwd", ".")

    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)

    # Read the transcript — it's a .jsonl file (one JSON object per line)
    lines = Path(transcript_path).read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        sys.exit(0)

    # Find the last user message and the last assistant message
    user_text      = None
    assistant_text = None

    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        role    = entry.get("role", "")
        content = entry.get("content", "")

        # Content can be a string or a list of blocks
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        else:
            text = str(content).strip()

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

    # Import LatentMind — resolve path relative to cwd
    sys.path.insert(0, cwd)
    try:
        from latentmind import LatentMind
    except ImportError:
        # LatentMind not installed in this project — exit silently
        sys.exit(0)

    mind = LatentMind(Path(cwd) / ".latentmind")

    session_id = raw.get("session_id", "unknown")

    if user_text:
        mind.add(user_text, metadata={
            "role":       "user",
            "session_id": session_id,
            "source":     "hook",
        })

    if assistant_text:
        mind.add(assistant_text, metadata={
            "role":       "assistant",
            "session_id": session_id,
            "source":     "hook",
        })

    # Exit 0 — allow Claude to stop normally
    sys.exit(0)


if __name__ == "__main__":
    main()