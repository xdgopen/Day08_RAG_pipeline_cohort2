"""Task 9 — Retrieval Pipeline hoàn chỉnh.

Hybrid = semantic TF-IDF cosine + lexical BM25, merge bằng RRF, rerank lightweight,
fallback sang PageIndex offline khi score dưới threshold.
"""
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf, _query_entities
from .task8_pageindex_vectorless import pageindex_search
from .task6_lexical_search import normalize_text

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def hyde_expand_query(query: str) -> str:
    """Bonus HyDE (Hypothetical Document Embeddings) offline.

    Sinh một đoạn tài liệu giả định ngắn từ query rồi nối vào query để semantic/BM25
    có thêm thuật ngữ ngữ cảnh. Khi dùng LLM thật, phần này có thể thay bằng một
    prompt sinh hypothetical answer/document trước khi embedding.
    """
    return (
        f"{query}. Tài liệu giả định liên quan pháp luật ma túy, nghệ sĩ Việt Nam, "
        "bị bắt, bị truy tố, tổ chức sử dụng ma túy, tàng trữ, mua bán, nguồn báo chí, "
        "Bộ luật Hình sự, Luật Phòng chống ma túy."
    )


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, score_threshold: float = SCORE_THRESHOLD, use_reranking: bool = True, use_hyde: bool = True) -> list[dict]:
    # Run both original query and HyDE-expanded query. Original query preserves
    # named entities (Miu Lê, Chi Dân...), HyDE adds domain terms. Combining both
    # prevents generic drug-law chunks from outranking the target person/article.
    search_query = hyde_expand_query(query) if use_hyde else query
    search_limit = max(top_k * 6, 20)
    dense_results = semantic_search(query, top_k=search_limit) + semantic_search(search_query, top_k=search_limit)
    sparse_results = lexical_search(query, top_k=search_limit) + lexical_search(search_query, top_k=search_limit)
    merged = rerank_rrf([dense_results, sparse_results], top_k=max(top_k * 6, 20))
    for item in merged:
        item["source"] = "hybrid"
    final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking and merged else merged[:top_k]
    # For celebrity/person-specific questions, hide unrelated case chunks from the
    # final source list. This improves both answer quality and UI source display.
    person_entities = [
        ent for ent in _query_entities(query)
        if ent in {"miu le", "chi dan", "an tay", "truc phuong", "long nhat", "son ngoc minh", "son ngoc"}
    ]
    if person_entities:
        known_people = {"miu le", "chi dan", "an tay", "truc phuong", "long nhat", "son ngoc minh", "son ngoc"}
        entity_filtered = []
        for item in final_results:
            md = item.get("metadata", {}) or {}
            title_norm = normalize_text(str(md.get("title", "")) + " " + str(md.get("source", "")))
            title_people = {p for p in known_people if p in title_norm}
            # If this is a news article about other named people, do not keep it
            # just because a related-news/footer snippet mentions the query person.
            if title_people and not any(ent in title_norm for ent in person_entities):
                continue
            haystack = normalize_text(item.get("content", "") + " " + str(md.get("source", "")) + " " + str(md.get("path", "")))
            if any(ent in haystack for ent in person_entities):
                entity_filtered.append(item)
        if entity_filtered:
            final_results = entity_filtered[:top_k]

    for item in final_results:
        item.setdefault("metadata", {})["hyde_enabled"] = use_hyde
    for item in final_results:
        item["source"] = "hybrid"
    if not final_results or final_results[0].get("score",0.0) < score_threshold:
        return pageindex_search(query, top_k=top_k)
    return final_results[:top_k]

if __name__ == "__main__":
    print(retrieve("hình phạt ma tuý",3))
