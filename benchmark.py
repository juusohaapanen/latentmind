"""
LongMemEval benchmark for LatentMind.

Two evaluation modes:
  - Recall@k   (default)    : did LatentMind retrieve the answer in top-k docs?
  - End-to-end (--llm)      : LatentMind retrieves, Claude generates, answer scored.

Usage:
    latentmind-bench [--split oracle|s|m] [--limit N] [--k K [K ...]]
                     [--llm] [--model MODEL]
                     [--output FILE]
                     [--components K] [--refit-every N]
                     [--min-df N] [--max-df F] [--ngram-max N] [--max-features N]

Dataset: xiaowu0162/longmemeval-cleaned (HuggingFace, auto-downloaded)
Install:  uv tool install --editable . --with datasets
LLM mode: also requires  uv tool install --editable . --with anthropic
"""

import json
import argparse
import tempfile
from pathlib import Path
from typing import Optional

from latentmind import LatentMind
from latentmind._lsa import LSAModel

try:
    from datasets import load_dataset as _hf_load
except ImportError:
    _hf_load = None

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s":      "longmemeval_s_cleaned.json",
    "m":      "longmemeval_m_cleaned.json",
}

DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
LOCAL_CACHE  = Path.home() / ".cache" / "latentmind"


def load_longmemeval(split: str = "oracle") -> list[dict]:
    cache_file = LOCAL_CACHE / f"longmemeval_{split}.json"

    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        return json.loads(cache_file.read_text())

    if _hf_load is None:
        raise RuntimeError(
            "datasets package not installed.\n"
            "Run: uv tool install --editable . --with datasets"
        )
    filename = SPLIT_FILES[split]
    print(f"Downloading {DATASET_REPO} / {filename} ...")
    ds = _hf_load(
        DATASET_REPO,
        data_files={"train": filename},
        split="train",
    )
    items = list(ds)
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(items))
    print(f"Cached to {cache_file}")
    return items


def is_abstention(answer) -> bool:
    if answer is None:
        return True
    if isinstance(answer, str) and not answer.strip():
        return True
    if isinstance(answer, list) and all(
        a is None or (isinstance(a, str) and not a.strip()) for a in answer
    ):
        return True
    return False


def normalise_answer(answer) -> list[str]:
    if isinstance(answer, list):
        return [str(a).lower().strip() for a in answer if a is not None]
    return [str(answer).lower().strip()]


# ---------------------------------------------------------------------------
# Memory ingestion
# ---------------------------------------------------------------------------

def ingest_sessions(mind: LatentMind, sessions: list) -> int:
    texts = []
    for idx, session in enumerate(sessions):
        for turn in session:
            role    = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if content:
                texts.append(f"[Session {idx} / {role}] {content}")
    if texts:
        mind.add_batch(texts)
    return len(texts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    mind: LatentMind,
    question: str,
    k: int,
) -> list[tuple[float, str]]:
    return mind.search(question, top_n=k)


# ---------------------------------------------------------------------------
# Answer generation (LLM mode)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a user's past conversation memories. "
    "Answer the question using only the provided memories. Be concise — one sentence "
    "or a short phrase is ideal. "
    "If the answer cannot be determined from the memories, reply exactly: I don't know."
)


def generate_answer(client, question: str, docs: list[tuple[float, str]], model: str) -> str:
    context = "\n\n".join(f"[relevance {s:.2f}] {t}" for s, t in docs)
    response = client.messages.create(
        model=model,
        max_tokens=128,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Memories:\n{context}\n\nQuestion: {question}",
        }],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_ABSTENTION_PHRASES = frozenset({
    "i don't know", "i do not know", "not mentioned", "not provided",
    "unknown", "cannot answer", "no information", "not present", "not available",
})

_JUDGE_PROMPTS: dict[str, str] = {
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. In addition, "
        "do not penalize off-by-one errors for the number of days. If the question asks for "
        "the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still correct."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly."
    ),
}

_JUDGE_PROMPT_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no."
)


def _judge_prompt_for(qtype: str) -> str:
    return _JUDGE_PROMPTS.get(qtype, _JUDGE_PROMPT_DEFAULT)


def _answer_label(qtype: str) -> str:
    return "Rubric" if qtype == "single-session-preference" else "Correct answer"


def score_retrieval(docs: list[tuple[float, str]], ans_strs: list[str]) -> bool:
    """True if any answer string appears in any retrieved doc."""
    return any(a in doc.lower() for _, doc in docs for a in ans_strs)


def judge_answer(
    client,
    qtype: str,
    question: str,
    expected: str,
    generated: str,
    model: str,
    docs: list[tuple[float, str]] | None = None,
    judge_top_n: int = 5,
) -> bool:
    """Ask Claude to judge whether the generated answer is correct. Returns True/False."""
    label  = _answer_label(qtype)
    prompt = _judge_prompt_for(qtype)

    memories_section = ""
    if docs:
        top = docs[:judge_top_n]
        memories_section = "Retrieved memories:\n" + "\n".join(
            f"  [{i+1}] (relevance {s:.2f}) {t}" for i, (s, t) in enumerate(top)
        ) + "\n\n"

    user_msg = (
        f"{prompt}\n\n"
        f"Question: {question}\n"
        f"{memories_section}"
        f"{label}: {expected}\n"
        f"Model response: {generated}\n\n"
        f"Answer (yes/no):"
    )
    response = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": user_msg}],
    )
    verdict = response.content[0].text.strip().lower()
    return verdict.startswith("yes")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    split: str,
    limit: Optional[int],
    ks: list[int],
    output_path: Optional[Path],
    use_llm: bool = False,
    llm_model: str = "claude-opus-4-6",
    judge_top_n: int = 5,
    n_components: int = 8,
    refit_every: int = 20,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = None,
) -> None:
    if use_llm:
        if _anthropic is None:
            raise RuntimeError(
                "anthropic package not installed.\n"
                "Run: uv tool install --editable . --with anthropic"
            )
        llm_client = _anthropic.Anthropic()
    else:
        llm_client = None

    items = load_longmemeval(split)
    if limit:
        items = items[:limit]

    eval_items = [(i, item) for i, item in enumerate(items) if not is_abstention(item["answer"])]
    n_skipped  = len(items) - len(eval_items)
    total      = len(eval_items)
    max_k      = max(ks)

    mode = f"retrieval + {llm_model} (judge_top_n={judge_top_n})" if use_llm else "retrieval-only"
    print(
        f"Questions: {total} evaluated, {n_skipped} abstention skipped  |  "
        f"split={split}  mode={mode}  k={ks}  components={n_components}  "
        f"min_df={min_df}  max_df={max_df}  ngram={ngram_range}  "
        f"max_features={max_features}\n"
    )

    # recall_hits[k][qtype], answer_hits[k][qtype]
    recall_hits: dict[int, dict[str, list[bool]]] = {k: {} for k in ks}
    answer_hits: dict[int, dict[str, list[bool]]] = {k: {} for k in ks}
    results: list[dict] = []

    for rank, (_, item) in enumerate(eval_items):
        qid      = item["question_id"]
        qtype    = item.get("question_type", "unknown")
        question = item["question"]
        answer   = item["answer"]
        sessions = item.get("haystack_sessions", [])
        ans_strs = normalise_answer(answer)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSAModel(
                n_components=n_components,
                refit_every=refit_every,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                max_features=max_features,
            )
            mind     = LatentMind(Path(tmpdir), model=model)
            n_stored = ingest_sessions(mind, sessions)

            item_recall = {}
            item_answer = {}
            item_docs   = {}
            item_generated = {}

            for k in ks:
                docs = retrieve(mind, question, k=k)
                item_docs[k] = [{"score": round(s, 3), "text": t} for s, t in docs]

                r_hit = score_retrieval(docs, ans_strs)
                item_recall[k] = r_hit
                recall_hits[k].setdefault(qtype, []).append(r_hit)

                if use_llm:
                    generated = generate_answer(llm_client, question, docs, llm_model)
                    expected_str = answer if isinstance(answer, str) else (
                        "; ".join(str(a) for a in answer) if isinstance(answer, list) else str(answer)
                    )
                    a_hit = judge_answer(
                        llm_client, qtype, question, expected_str, generated, llm_model,
                        docs=docs, judge_top_n=judge_top_n,
                    )
                    item_answer[k] = a_hit
                    item_generated[k] = generated
                    answer_hits[k].setdefault(qtype, []).append(a_hit)

        if use_llm:
            marks = "  ".join(
                f"R@{k}={'✓' if item_recall[k] else '✗'} A@{k}={'✓' if item_answer[k] else '✗'}"
                for k in ks
            )
        else:
            marks = "  ".join(f"R@{k}={'✓' if item_recall[k] else '✗'}" for k in ks)
        print(f"[{rank+1:3}/{total}]  {marks}  {qtype:<35}  memories={n_stored}")

        record = {
            "question_id":   qid,
            "question_type": qtype,
            "question":      question,
            "answer":        answer,
            **{f"recall_at_{k}": item_recall[k] for k in ks},
            "retrieved": item_docs,
        }
        if use_llm:
            record.update({
                **{f"answer_correct_at_{k}": item_answer[k] for k in ks},
                "generated": item_generated,
            })
        results.append(record)

    # --- summary ---
    all_recall = {k: [] for k in ks}
    all_answer = {k: [] for k in ks}
    qtypes     = sorted({qt for k in ks for qt in recall_hits[k]})
    col_w      = 6

    print("\n" + "=" * 80)
    header = f"  {'Question type':<38}"
    for k in ks:
        header += f"  R@{k:<3}"
        if use_llm:
            header += f"  A@{k:<3}"
    header += "      n"
    print(header)
    print("-" * 80)

    for qtype in qtypes:
        row = f"  {qtype:<38}"
        for k in ks:
            rh = recall_hits[k].get(qtype, [])
            all_recall[k].extend(rh)
            row += f"  {sum(rh)/len(rh):>4.1%}" if rh else f"  {'N/A':>4}"
            if use_llm:
                ah = answer_hits[k].get(qtype, [])
                all_answer[k].extend(ah)
                row += f"  {sum(ah)/len(ah):>4.1%}" if ah else f"  {'N/A':>4}"
        n = len(recall_hits[ks[0]].get(qtype, []))
        row += f"  {n:>6}"
        print(row)

    print("-" * 80)
    overall = f"  {'OVERALL':<38}"
    for k in ks:
        rh = all_recall[k]
        overall += f"  {sum(rh)/len(rh):>4.1%}" if rh else f"  {'N/A':>4}"
        if use_llm:
            ah = all_answer[k]
            overall += f"  {sum(ah)/len(ah):>4.1%}" if ah else f"  {'N/A':>4}"
    overall += f"  {total:>6}"
    print(overall)

    if output_path:
        output_path.write_text("\n".join(json.dumps(r) for r in results) + "\n")
        print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="latentmind-bench",
        description="LongMemEval benchmark for LatentMind: Recall@k and optional end-to-end accuracy",
    )
    parser.add_argument("--split", choices=["oracle", "s", "m"], default="oracle")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[5, 10], metavar="K",
        help="k values for Recall@k (default: 5 10)",
    )
    parser.add_argument("--output", type=Path, default=None, metavar="FILE")

    # LLM mode
    llm = parser.add_argument_group("LLM (end-to-end) mode")
    llm.add_argument(
        "--llm", action="store_true",
        help="Enable end-to-end evaluation: retrieve → Claude generates → score answer",
    )
    llm.add_argument(
        "--model", default="claude-opus-4-6", metavar="MODEL",
        help="Claude model for answer generation (default: claude-opus-4-6)",
    )
    llm.add_argument(
        "--judge-top-n", type=int, default=5, metavar="N",
        help="Number of top retrieved memories to include in the judge prompt (default: 5)",
    )

    # LSA
    lsa = parser.add_argument_group("LSA parameters")
    lsa.add_argument("--components", type=int, default=8, metavar="K")
    lsa.add_argument("--refit-every", type=int, default=20, metavar="N")

    # TF-IDF
    tfidf = parser.add_argument_group("TF-IDF parameters")
    tfidf.add_argument("--min-df", type=int, default=1, metavar="N")
    tfidf.add_argument("--max-df", type=float, default=0.95, metavar="F")
    tfidf.add_argument("--ngram-max", type=int, default=1, metavar="N")
    tfidf.add_argument("--max-features", type=int, default=None, metavar="N")

    args = parser.parse_args()

    run_benchmark(
        split=args.split,
        limit=args.limit,
        ks=sorted(set(args.k)),
        output_path=args.output,
        use_llm=args.llm,
        llm_model=args.model,
        judge_top_n=args.judge_top_n,
        n_components=args.components,
        refit_every=args.refit_every,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
