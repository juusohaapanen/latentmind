#!/usr/bin/env python3
"""
LatentMind — SessionStart hook
Fires at the start of every Claude Code session.

1. Runs LSA consolidation (refit on full corpus)
2. Returns additionalContext so Claude is aware memories are available

Receives JSON on stdin:
  {
    "session_id": "...",
    "hook_event_name": "SessionStart"
  }

Returns JSON on stdout:
  {
    "additionalContext": "..."
  }
"""

import json
import sys


def main():
    json.load(sys.stdin)  # consume stdin (required by hook protocol)

    try:
        from latentmind import LatentMind
    except ImportError:
        sys.exit(0)

    mind = LatentMind()
    mind.session_start_consolidation()

    count = mind.count()
    if count == 0:
        sys.exit(0)

    context = (
        f"LatentMind memory system is active with {count} memories. "
        f"You can recall relevant context by searching memories in latentmind.py."
    )
    json.dump({"additionalContext": context}, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
