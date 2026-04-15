# LatentMind

Semantic long-term memory for Claude Code. Every conversation is automatically stored and searchable across sessions using Latent Semantic Analysis (LSA). No external embedding API required. 

## Introduction

Large language models suffer from early dementia: they can’t accurately remember what you’ve discussed with them, causing wrong answers and bad results for tasks they are doing. For example, in software development, they can forget important development and architecture decisions. For many agentic tasks, LLM requires proper knowledge and context. 

This is an experimental project to explore whether traditional text-mining and information-retrieval methods can serve as a memory for AI agents. Apparently, traditional methods work pretty well. 

The system is built on two technologies: LSA (Latent Semantic Analysis), which handles all the math, and a storage layer built with ChromaDB. 

The key design decision: ChromaDB handles persistence and ANN search; LSA handles semantics. ChromaDB's own embedding model is bypassed, and LatentMind injects its own vectors.

Refitting is necessary because LSA is a global decomposition — when new documents arrive, the latent space shifts. Every refit_every document, the model is retrained, and all stored vectors are updated in-place.



## Latent Semantic Analysis

Latent Semantic Analysis (LSA) is a method to extract information from large language corpora using matrix algebra. The underlying idea (according to Landauer et al.) is “that the aggregate of all the word contexts in which a given word does and does not appear provides a set of mutual constraints that largely determines the similarity of meaning of words”. (Landauer, Foltz & Laham, 1998) This means, in practice, that textual corpora are presented as a term-document matrix, and singular value decomposition is performed to extract latent meanings of words from the data. For example, the model would learn that the terms “car” and “vehicle” mean the same. This is a very useful feature for information retrieval. 

Unlike modern transformer-based models, LSA doesn’t care about word order, which is elegantly solved in transformer architectures using positional encoding. But for AI memory, it doesn't seem to be a particularly important feature: the model can still find relevant entries in memory.

The biggest challenge with LSA, especially on the LongMemEval benchmark, is that there aren’t enough documents for the model to learn from. 


## Does it work?

Yes. The system was evaluated with the LongMemEval benchmark. Without neural networks or an external API for indexing, it achieved 95.7% accuracy and an overall end-to-end accuracy of 53.8%.

| Question Type | R@5 | A@5 | R@10 | A@10 | n |
|---|---|---|---|---|---|
| knowledge-update | 64.1% | 79.5% | 70.5% | 83.3% | 78 |
| multi-session | 19.5% | 38.3% | 31.6% | 44.4% | 133 |
| single-session-assistant | 17.9% | 37.5% | 19.6% | 35.7% | 56 |
| single-session-preference | 0.0% | 50.0% | 0.0% | 46.7% | 30 |
| single-session-user | 80.0% | 95.7% | 81.4% | 94.3% | 70 |
| temporal-reasoning | 21.1% | 30.8% | 24.8% | 33.8% | 133 |
| **OVERALL** | **34.0%** | **51.4%** | **39.6%** | **53.8%** | **500** |


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
