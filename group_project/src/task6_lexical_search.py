"""Task 6 — Lexical Search Module.

Default `lexical_search()` uses BM25 for Task 6. Bonus alternative: TF-IDF cosine.
Both methods now use accent-folded Vietnamese tokens to improve retrieval quality.
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from .task4_chunking_indexing import get_chunks

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    """Tokenize original + accent-folded text for robust Vietnamese matching."""
    original = _TOKEN_RE.findall((text or "").lower())
    folded = _TOKEN_RE.findall(normalize_text(text))
    return original + folded


class SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.df: dict[str, int] = {}
        for doc in tokenized_corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
        self.N = len(tokenized_corpus)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc, dl in zip(self.corpus, self.doc_len):
            tf = Counter(doc)
            score = 0.0
            for q in query_tokens:
                if q not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df.get(q, 0) + 0.5) / (self.df.get(q, 0) + 0.5))
                denom = tf[q] + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1)))
                score += idf * (tf[q] * (self.k1 + 1)) / denom
            scores.append(float(score))
        return scores


class SimpleTFIDF:
    """Alternative lexical index: TF-IDF + cosine similarity.

    TF counts query/document terms; IDF rewards rare terms; cosine similarity
    measures vector angle. This is simpler than BM25 and is included for bonus.
    """

    def __init__(self, tokenized_corpus: list[list[str]]):
        self.N = len(tokenized_corpus)
        self.df: dict[str, int] = {}
        for doc in tokenized_corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
        self.idf = {token: math.log((self.N + 1) / (df + 1)) + 1 for token, df in self.df.items()}
        self.doc_vectors = [self._vectorize(tokens) for tokens in tokenized_corpus]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {token: (1 + math.log(tf)) * self.idf.get(token, 1.0) for token, tf in counts.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(token, 0.0) for token, value in a.items())
        na = math.sqrt(sum(value * value for value in a.values()))
        nb = math.sqrt(sum(value * value for value in b.values()))
        return float(dot / (na * nb)) if na and nb else 0.0

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        qv = self._vectorize(query_tokens)
        return [self._cosine(qv, dv) for dv in self.doc_vectors]


CORPUS: list[dict] = get_chunks()
_BM25: SimpleBM25 | None = None
_TFIDF: SimpleTFIDF | None = None


def build_bm25_index(corpus: list[dict]) -> SimpleBM25:
    return SimpleBM25([tokenize(doc["content"]) for doc in corpus])


def build_tfidf_index(corpus: list[dict]) -> SimpleTFIDF:
    return SimpleTFIDF([tokenize(doc["content"]) for doc in corpus])


def _format_results(scores: list[float], top_k: int) -> list[dict]:
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in idxs:
        if scores[i] <= 0 and results:
            continue
        item = CORPUS[i]
        results.append({"content": item["content"], "score": float(scores[i]), "metadata": item.get("metadata", {})})
    return results[:top_k]


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    global _BM25, CORPUS
    if not CORPUS:
        CORPUS = get_chunks()
    if _BM25 is None:
        _BM25 = build_bm25_index(CORPUS)
    return _format_results(_BM25.get_scores(tokenize(query)), top_k)


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    global _TFIDF, CORPUS
    if not CORPUS:
        CORPUS = get_chunks()
    if _TFIDF is None:
        _TFIDF = build_tfidf_index(CORPUS)
    return _format_results(_TFIDF.get_scores(tokenize(query)), top_k)


def lexical_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict]:
    method = method.lower().strip()
    if method == "bm25":
        return bm25_search(query, top_k)
    if method in {"tfidf", "tf-idf"}:
        return tfidf_search(query, top_k)
    raise ValueError(f"Unknown lexical search method: {method}. Use 'bm25' or 'tfidf'.")


if __name__ == "__main__":
    print(lexical_search("ma tuy", 3))
    print(lexical_search("ma tuy", 3, method="tfidf"))
