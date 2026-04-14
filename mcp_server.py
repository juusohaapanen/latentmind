from mcp.server.fastmcp import FastMCP
from latentmind import LatentMind

mcp  = FastMCP("latentmind")
mind = LatentMind()


@mcp.tool()
def search(query: str, top_n: int = 5) -> str:
    """Semantic search over LatentMind memories. Returns the closest matches."""
    results = mind.search(query, top_n=top_n)
    if not results:
        return "No memories found."
    return "\n".join(f"{score:.3f}  {text}" for score, text in results)


@mcp.tool()
def add(text: str) -> str:
    """Add a new memory to LatentMind."""
    mem_id = mind.add(text)
    return f"Saved as {mem_id} ({mind.count()} total memories)."


@mcp.tool()
def recent(n: int = 5) -> str:
    """Return the n most recently added memories."""
    memories = mind.recent(n=n)
    if not memories:
        return "No memories yet."
    return "\n".join(f"{i}. {text}" for i, text in enumerate(memories, 1))


@mcp.tool()
def count() -> str:
    """Return the total number of stored memories."""
    return str(mind.count())


if __name__ == "__main__":
    mcp.run()
