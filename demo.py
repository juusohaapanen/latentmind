"""
Smoke-test / demo script for LatentMind.

Seeds the store with a small synthetic corpus (Tampere geography, AI/ML,
Python tooling, food & coffee), then runs a handful of semantic queries.

Run with:
    python demo.py
"""

from pathlib import Path
from latentmind import LatentMind

# ---------------------------------------------------------------------------
# Synthetic seed corpus
# Topics: Tampere geography | AI/ML | Python/tools | Food/coffee
# Bridge sentences deliberately cross two topics to exercise LSA.
# ---------------------------------------------------------------------------

SEED_MEMORIES = [
    # Tampere geography
    "Rahola is a neighbourhood in the western part of Tampere",
    "Kaarila is close to Rahola and has a nice small harbour",
    "Siirtolapuutarha is a garden allotment area between Rahola and Kaarila",
    "Walking from Rahola to Kaarila takes about twenty minutes",
    "Pyynikki ridge is a popular nature area near the centre of Tampere",
    "Tampere has two large lakes, Näsijärvi and Pyhäjärvi",

    # AI / ML concepts
    "Latent semantic analysis uses singular value decomposition to find topics",
    "Word embeddings represent words as dense vectors in a high dimensional space",
    "Cosine similarity measures the angle between two vectors in semantic space",
    "TF-IDF weights terms by frequency in a document versus the whole corpus",
    "Vector spaces allow semantically similar concepts to cluster together",
    "SVD reduces a high dimensional matrix into a lower dimensional representation",

    # Python / tools
    "uv is a fast Python package manager written in Rust",
    "scikit-learn provides TfidfVectorizer and TruncatedSVD for LSA pipelines",
    "numpy is the foundation for numerical computing in Python",
    "Python 3.13 introduced several performance improvements to the interpreter",
    "uv init creates a new project with a pyproject toml and virtual environment",

    # Food / coffee
    "A good cup of coffee helps when reading about machine learning papers",
    "Tampere is famous for mustamakkara, a local black sausage",
    "Pyynikki donuts are the most famous pastry in Tampere",

    # Bridge sentences
    "I was drinking coffee while reading about word embeddings and vector spaces",
    "Walking around Pyynikki is a good way to think about clustering algorithms",
    "uv makes it easy to set up a numpy environment for experimenting with LSA",
    "Eating mustamakkara by Näsijärvi while thinking about semantic similarity",
]

QUERIES = [
    "walking in Tampere neighbourhoods",
    "machine learning and vector representations",
    "Python packaging tools",
    "food and drinks in Tampere",
    "coffee and thinking about AI",
]

if __name__ == "__main__":
    mind = LatentMind()

    if mind.count() == 0:
        print("Seeding LatentMind with initial memories...")
        mind.add_batch(SEED_MEMORIES)
        print(f"Seeded {mind.count()} memories.\n")
    else:
        mind.session_start_consolidation()

    for q in QUERIES:
        print(f"Query: {q!r}")
        for score, text in mind.search(q, top_n=3):
            print(f"  {score:.3f}  {text}")
        print()

    print("Adding a new memory...")
    mind.add(
        "Claude Code hooks let you run scripts at the end of each session",
        metadata={"topic": "tools", "source": "manual"},
    )
    print(f"LatentMind now has {mind.count()} memories.\n")

    print("Searching for the new memory:")
    for score, text in mind.search("Claude Code session hooks", top_n=3):
        print(f"  {score:.3f}  {text}")
