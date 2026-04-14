"""Command-line interface for LatentMind."""

import argparse
from ._mind import LatentMind


def cli():
    parser = argparse.ArgumentParser(
        prog="latentmind",
        description="LatentMind — semantic memory for Claude Code",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("count", help="Print the number of stored memories")

    p_recent = sub.add_parser("recent", help="Show the most recently added memories")
    p_recent.add_argument("-n", type=int, default=5, metavar="N",
                          help="Number of memories to show (default: 5)")

    p_search = sub.add_parser("search", help="Semantic search over memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", "--top-n", type=int, default=5, metavar="N",
                          help="Number of results (default: 5)")

    p_add = sub.add_parser("add", help="Add a new memory")
    p_add.add_argument("text", help="Memory text to store")

    args = parser.parse_args()
    mind = LatentMind()

    if args.cmd == "count":
        print(mind.count())

    elif args.cmd == "recent":
        memories = mind.recent(n=args.n)
        if not memories:
            print("No memories yet.")
        else:
            for i, text in enumerate(memories, 1):
                print(f"{i}. {text}")

    elif args.cmd == "search":
        results = mind.search(args.query, top_n=args.top_n)
        if not results:
            print("No memories found.")
        else:
            for score, text in results:
                print(f"{score:.3f}  {text}")

    elif args.cmd == "add":
        mem_id = mind.add(args.text)
        print(f"Saved as {mem_id} ({mind.count()} total memories).")
