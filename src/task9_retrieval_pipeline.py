"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from concurrent.futures import ThreadPoolExecutor

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
    from .bonus_hyde import hyde_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search
    from bonus_hyde import hyde_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"
RETRIEVAL_MULTIPLIER = 4
USE_HYDE = True


def mark_source(results: list[dict], source: str) -> list[dict]:
    """Attach pipeline source label without mutating caller-owned results."""
    marked = []
    for item in results:
        marked.append(
            {
                **item,
                "metadata": {
                    **item.get("metadata", {}),
                    "pipeline_source": source,
                },
                "source": source,
            }
        )
    return marked


def fallback_pageindex(query: str, top_k: int, fallback_reason: str) -> list[dict]:
    """Run PageIndex fallback; return [] when PageIndex is not configured."""
    try:
        results = pageindex_search(query, top_k=top_k)
    except Exception as error:
        print(f"  ! PageIndex fallback unavailable: {error}")
        return []

    marked = mark_source(results, "pageindex")
    for item in marked:
        item["metadata"] = {
            **item.get("metadata", {}),
            "fallback_reason": fallback_reason,
        }
    return marked


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = USE_HYDE,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không
        use_hyde: Có thêm HyDE retrieval channel hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    candidate_k = max(top_k * RETRIEVAL_MULTIPLIER, top_k)

    # Step 1: dense + sparse + optional HyDE retrieval run in parallel because they are
    # independent retrieval channels.
    with ThreadPoolExecutor(max_workers=3) as executor:
        dense_future = executor.submit(semantic_search, query, candidate_k)
        sparse_future = executor.submit(lexical_search, query, candidate_k)
        hyde_future = executor.submit(hyde_search, query, candidate_k) if use_hyde else None
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()
        hyde_results = hyde_future.result() if hyde_future else []

    # Step 2: RRF merge. RRF uses ranks, so BM25 and cosine scales do not need
    # manual normalization before fusion.
    ranked_lists = [dense_results, sparse_results]
    if hyde_results:
        ranked_lists.append(hyde_results)

    merged = rerank_rrf(ranked_lists, top_k=candidate_k)
    merged = mark_source(merged, "hybrid")

    # Step 3: MMR rerank balances relevance with diversity using Task 4 vectors.
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = mark_source(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    # Step 4: fallback if hybrid has no confident result.
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        reason = f"hybrid_score={best_score:.3f}<threshold={score_threshold:.3f}"
        print(f"  ! Hybrid weak ({reason}). Fallback -> PageIndex")
        fallback_results = fallback_pageindex(query, top_k=top_k, fallback_reason=reason)
        if fallback_results:
            return fallback_results

        # Keep degraded hybrid results when PageIndex is not available locally.
        for item in final_results:
            item["metadata"] = {
                **item.get("metadata", {}),
                "fallback_attempted": True,
                "fallback_available": False,
                "fallback_reason": reason,
            }

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
