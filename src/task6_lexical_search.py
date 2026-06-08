"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
from functools import lru_cache
from pathlib import Path

INDEX_PATH = Path(__file__).parent.parent / "data" / "vector_store" / "drug_law_docs_index.json"
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase, giữ dấu và số điều luật."""
    return TOKEN_PATTERN.findall(text.lower())


@lru_cache(maxsize=1)
def load_corpus() -> tuple[dict, ...]:
    """Load chunks đã tạo ở Task 4 làm corpus cho BM25."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vector index: {INDEX_PATH}. "
            "Hãy chạy Task 4 trước: .venv/bin/python src/task4_chunking_indexing.py"
        )

    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return tuple(
        {
            "content": chunk["content"],
            "metadata": {
                **chunk["metadata"],
                "chunk_id": chunk.get("id"),
            },
        }
        for chunk in data["chunks"]
    )


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def get_bm25_index():
    """Cache BM25 index để nhiều lần search không phải build lại."""
    corpus = load_corpus()
    return build_bm25_index(list(corpus))


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    query_tokens = tokenize(query)
    if not query_tokens or top_k <= 0:
        return []

    corpus = load_corpus()
    bm25 = get_bm25_index()
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        results.append(
            {
                "content": corpus[index]["content"],
                "score": score,
                "metadata": corpus[index]["metadata"],
            }
        )

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
