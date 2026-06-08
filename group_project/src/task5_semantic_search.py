"""Task 5 — Semantic Search Module.

Improved local semantic fallback using normalized TF-IDF cosine similarity with
Vietnamese accent folding. This helps queries like "Miu Lê bị bắt" match content
that may be searched as "Miu Le bi bat" and reduces brittle exact-token behavior.
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from .task4_chunking_indexing import get_chunks
from .task6_lexical_search import tokenize

_CORPUS = None
_DOC_VECS = None
_IDF = None


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return text


def _tokens(text: str) -> list[str]:
    # Include both original tokenizer and accent-folded tokenizer.
    return tokenize(text) + re.findall(r"\w+", normalize_text(text), re.UNICODE)


def _build_index():
    global _CORPUS, _DOC_VECS, _IDF
    _CORPUS = get_chunks()
    tokenized = [_tokens(c["content"]) for c in _CORPUS]
    N = len(tokenized)
    df = {}
    for toks in tokenized:
        for token in set(toks):
            df[token] = df.get(token, 0) + 1
    _IDF = {token: math.log((N + 1) / (freq + 1)) + 1 for token, freq in df.items()}
    _DOC_VECS = []
    for toks in tokenized:
        cnt = Counter(toks)
        vec = {token: (1 + math.log(tf)) * _IDF.get(token, 1.0) for token, tf in cnt.items()}
        _DOC_VECS.append(vec)


def _cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(token, 0.0) for token, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _query_vec(query):
    cnt = Counter(_tokens(query))
    return {token: (1 + math.log(tf)) * (_IDF.get(token, 1.0) if _IDF else 1.0) for token, tf in cnt.items()}


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    global _CORPUS
    if _CORPUS is None:
        _build_index()
    qv = _query_vec(query)
    scores = [_cos(qv, doc_vec) for doc_vec in _DOC_VECS]
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"content": _CORPUS[i]["content"], "score": float(scores[i]), "metadata": _CORPUS[i].get("metadata", {})}
        for i in idxs
    ]


if __name__ == "__main__":
    print(semantic_search("hình phạt ma tuý", 3))
