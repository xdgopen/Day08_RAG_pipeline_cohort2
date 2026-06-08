"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

INDEX_PATH = Path(__file__).parent.parent / "data" / "vector_store" / "drug_law_docs_index.json"
DEFAULT_METHOD = "mmr"


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity cho vector embedding."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=1)
def load_vector_index() -> dict:
    """Load local vector index đã tạo ở Task 4."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vector index: {INDEX_PATH}. "
            "Hãy chạy Task 4 trước: .venv/bin/python src/task4_chunking_indexing.py"
        )

    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str):
    """Load cùng embedding model đã dùng ở Task 4."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_chunk_embeddings() -> dict[str, list[float]]:
    """Map chunk_id -> embedding để MMR dùng lại vectors đã index."""
    index = load_vector_index()
    return {
        chunk["id"]: chunk["embedding"]
        for chunk in index["chunks"]
        if chunk.get("id") and chunk.get("embedding")
    }


def embed_query(query: str) -> list[float]:
    """Embed query bằng đúng model/config của Task 4."""
    index = load_vector_index()
    embedding_config = index["embedding"]
    model = load_embedding_model(embedding_config["model"])
    return model.encode(
        query,
        normalize_embeddings=embedding_config.get("normalized", False),
    ).tolist()


def attach_candidate_embeddings(candidates: list[dict]) -> list[dict]:
    """Attach embedding từ vector index vào candidates dựa trên metadata.chunk_id."""
    embedding_map = get_chunk_embeddings()
    enriched = []

    for candidate in candidates:
        chunk_id = candidate.get("metadata", {}).get("chunk_id")
        embedding = candidate.get("embedding") or embedding_map.get(chunk_id)
        if embedding is None:
            continue

        enriched.append({**candidate, "embedding": embedding})

    return enriched


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    raise NotImplementedError(
        "Cross-encoder reranking cần API key/model riêng. "
        "Module này dùng MMR mặc định để chạy offline."
    )


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_mmr_score = float("-inf")
        best_relevance = 0.0
        best_diversity_penalty = 0.0

        for idx in remaining:
            candidate_embedding = candidates[idx]["embedding"]
            relevance = cosine_sim(query_embedding, candidate_embedding)
            diversity_penalty = 0.0

            if selected:
                diversity_penalty = max(
                    cosine_sim(candidate_embedding, candidates[selected_idx]["embedding"])
                    for selected_idx in selected
                )

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * diversity_penalty
            )

            if mmr_score > best_mmr_score:
                best_idx = idx
                best_mmr_score = mmr_score
                best_relevance = relevance
                best_diversity_penalty = diversity_penalty

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        candidates[best_idx]["score"] = float(best_mmr_score)
        candidates[best_idx]["metadata"] = {
            **candidates[best_idx].get("metadata", {}),
            "rerank_method": "mmr",
            "relevance_score": float(best_relevance),
            "diversity_penalty": float(best_diversity_penalty),
            "lambda_param": lambda_param,
        }

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item.pop("embedding", None)
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    candidate_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("metadata", {}).get("chunk_id") or item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            candidate_map[key] = item

    results = []
    for key, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        candidate = candidate_map[key].copy()
        candidate["score"] = float(score)
        candidate["metadata"] = {
            **candidate.get("metadata", {}),
            "rerank_method": "rrf",
        }
        results.append(candidate)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = DEFAULT_METHOD,  # "cross_encoder" | "mmr"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    query = query.strip()
    if not query or top_k <= 0 or not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        query_embedding = embed_query(query)
        enriched_candidates = attach_candidate_embeddings(candidates)
        return rerank_mmr(query_embedding, enriched_candidates, top_k=top_k)
    elif method == "rrf":
        raise NotImplementedError("RRF cần nhiều ranked lists; gọi rerank_rrf(...) trực tiếp")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    from task5_semantic_search import semantic_search

    query = "hình phạt tàng trữ trái phép chất ma túy"
    candidates = semantic_search(query, top_k=10)
    results = rerank(query, candidates, top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
