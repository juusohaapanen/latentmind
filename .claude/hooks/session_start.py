#!/usr/bin/env python3
"""
LatentMind — SessionStart hook
Fires at the start of every Claude Code session.

1. Runs LSA consolidation (refit on full corpus)
2. Returns additionalContext so Claude is aware memories are available

Receives JSON on stdin:
  {
    "session_id": "...",
    "cwd": "/path/to/project",
    "hook_event_name": "SessionStart"
  }

Returns JSON on stdout:
  {
    "additionalContext": "..."
  }
"""

import json
import sys
from pathlib import Path


def main():
    raw = json.load(sys.stdin)
    cwd = raw.get("cwd", ".")

    sys.path.insert(0, cwd)
    try:
        from latentmind import LatentMind
    except ImportError:
        sys.exit(0)

    mind = LatentMind(Path(cwd) / ".latentmind")

    # Consolidate — refit LSA on everything accumulated so far
    mind.session_start_consolidation()

    count = mind.count()

    if count == 0:
        sys.exit(0)

    # Return a short context blurb so Claude knows it can search memories
    context = (
        f"LatentMind memory system is active with {count} memories. "
        f"You can search past context by calling the search() method "
        f"in latentmind.py, or ask the user if they'd like to recall "
        f"something specific from previous sessions."
    )

    json.dump({"additionalContext": context}, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()