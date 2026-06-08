"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
import math
from functools import lru_cache
from pathlib import Path

INDEX_PATH = Path(__file__).parent.parent / "data" / "vector_store" / "drug_law_docs_index.json"


@lru_cache(maxsize=1)
def load_vector_index() -> dict:
    """Load local vector index được tạo ở Task 4."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vector index: {INDEX_PATH}. "
            "Hãy chạy Task 4 trước: .venv/bin/python src/task4_chunking_indexing.py"
        )

    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str):
    """Load đúng embedding model đã dùng khi index."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_name)


def cosine_similarity(a: list[float], b: list[float], normalized: bool = False) -> float:
    """Tính cosine similarity; nếu vector đã normalize thì dot product là đủ."""
    if normalized:
        return sum(x * y for x, y in zip(a, b))

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    return semantic_search_by_text(query, top_k=top_k, retrieval_method="semantic")


def semantic_search_by_text(
    search_text: str,
    top_k: int = 10,
    retrieval_method: str = "semantic",
) -> list[dict]:
    """
    Dense retrieval bằng một text bất kỳ.

    Hàm này dùng chung cho semantic search thường và HyDE, trong đó HyDE embed
    hypothetical document thay vì embed trực tiếp query gốc.
    """
    search_text = search_text.strip()
    if not search_text:
        return []

    if top_k <= 0:
        return []

    index = load_vector_index()
    embedding_config = index["embedding"]
    model_name = embedding_config["model"]
    expected_dim = embedding_config["dimension"]
    normalized = embedding_config.get("normalized", False)

    model = load_embedding_model(model_name)
    query_embedding = model.encode(search_text, normalize_embeddings=normalized).tolist()
    if len(query_embedding) != expected_dim:
        raise ValueError(
            f"Query embedding dim {len(query_embedding)} không khớp index dim {expected_dim}"
        )

    scored_results = []
    for chunk in index["chunks"]:
        score = cosine_similarity(query_embedding, chunk["embedding"], normalized=normalized)
        scored_results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": {
                    **chunk["metadata"],
                    "chunk_id": chunk.get("id"),
                    "retrieval_method": retrieval_method,
                },
            }
        )

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    return scored_results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
