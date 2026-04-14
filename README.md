# LatentMind

Semantic long-term memory for Claude Code. Every conversation is automatically stored and searchable across sessions using Latent Semantic Analysis (LSA) — no external embedding API required. 

## Motivation and architecture

Large language models suffer from early dementia: they can’t accurately remember what you’ve discussed with them, causing wrong answers and bad results for tasks they are doing. For example, in software development, they can forget important development and architecture decisions. For many agentic tasks, LLM requires proper knowledge and context. 

This is an experimental project to explore whether traditional or antique text-mining and information-retrieval methods can serve as a memory for AI agents. 

The system is built on two technologies: LSA (Latent Semantic Analysis), which handles all the math, and a storage layer built with ChromaDB. 

The key design decision: ChromaDB handles persistence and ANN search; LSA handles semantics. ChromaDB's own embedding model is bypassed and LatentMind injects its own vectors.

Refitting is necessary because LSA is a global decomposition — when new documents arrive, the latent space shifts. Every refit_every document, the model is retrained, and all stored vectors are updated in-place.


## How it works

After each Claude response, a hook extracts the latest exchange from the session transcript and saves it to a local ChromaDB store. At the start of the next session, the LSA model is refit on the full corpus and Claude is informed how many memories are available.

Retrieval is semantic rather than keyword-based: LSA projects all memories into a shared latent space (TF-IDF → TruncatedSVD → cosine similarity), so a query like *"Python packaging tools"* will surface memories about `uv` and `pyproject.toml` even if those exact words aren't in the query.

Memories are stored globally at `~/.latentmind/` and shared across all projects.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

Clone the repo and install it as a global tool with uv:

```sh
git clone <repo-url> latentmind
cd latentmind
uv tool install --editable .
```

Then copy the hooks to your global Claude config directory:

```sh
cp .claude/hooks/session_start.py ~/.claude/hooks/latentmind_session_start.py
cp .claude/hooks/save_memory.py   ~/.claude/hooks/latentmind_save_memory.py
```

Find the tool's Python interpreter path:

```sh
ls ~/.local/share/uv/tools/latentmind/bin/python
```

Finally, add the hooks to `~/.claude/settings.json` (replace the Python path if it differs):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/Users/<you>/.local/share/uv/tools/latentmind/bin/python ~/.claude/hooks/latentmind_session_start.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/Users/<you>/.local/share/uv/tools/latentmind/bin/python ~/.claude/hooks/latentmind_save_memory.py",
            "async": true
          }
        ]
      }
    ]
  }
}
```

## CLI

```sh
# How many memories are stored?
latentmind count

# Semantic search
latentmind search "Python packaging tools"
latentmind search "last week's debugging session" -n 10
```

## Python API

```python
from latentmind import LatentMind

mind = LatentMind()

mind.add("uv is a fast Python package manager written in Rust")
mind.add_batch(["memory one", "memory two"])

for score, text in mind.search("fast package manager", top_n=5):
    print(f"{score:.3f}  {text}")

print(mind.count())
```

## Configuration

Constants at the top of `latentmind.py`:

| Name | Default | Description |
|------|---------|-------------|
| `LATENTMIND_DIR` | `~/.latentmind` | Where ChromaDB is persisted |
| `K` | `8` | Number of LSA latent dimensions |
| `REFIT_EVERY` | `20` | Full corpus refit every N new memories |

## Development

```sh
uv run pytest tests/ -v
```
