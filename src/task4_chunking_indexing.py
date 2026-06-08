"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import json
from datetime import datetime
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "drug_law_docs_index.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy:
# - RecursiveCharacterTextSplitter giữ đoạn văn Markdown tương đối nguyên vẹn
#   bằng cách ưu tiên tách theo paragraph, dòng, câu rồi mới tới khoảng trắng.
# - Phù hợp corpus hiện tại vì file pháp luật dài, file báo ngắn, heading không
#   đồng đều; recursive splitter an toàn hơn MarkdownHeaderTextSplitter.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# Embedding model:
# - all-MiniLM-L6-v2 nhẹ, nhanh, đã có cache local trong máy nên index ổn định
#   không cần tải model lớn. Dimension 384 đủ nhỏ để lưu local và demo nhanh.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector store:
# - Lưu dense vectors vào JSON local để Task 4 index thành công ngay trong repo,
#   không phụ thuộc Docker/Weaviate server. Task 5 có thể load file này để search.
VECTOR_STORE = "local_json"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": str(relative_path),
                    "source_name": md_file.name,
                    "doc_type": doc_type,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    if CHUNKING_METHOD != "recursive":
        raise ValueError(f"Unsupported chunking method: {CHUNKING_METHOD}")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            chunk_id = f"{doc['metadata']['source']}::chunk_{chunk_index:04d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "chunk_count": len(splits),
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except TypeError:
        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        vector = embedding.tolist()
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dim {EMBEDDING_DIM}, got {len(vector)}"
            )
        chunk["embedding"] = vector

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE != "local_json":
        raise ValueError(f"Unsupported vector store: {VECTOR_STORE}")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_name": "DrugLawDocs",
        "created_at": datetime.now().isoformat(),
        "vector_store": VECTOR_STORE,
        "chunking": {
            "method": CHUNKING_METHOD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIM,
            "normalized": True,
        },
        "documents_indexed": len({chunk["metadata"]["source"] for chunk in chunks}),
        "chunks_indexed": len(chunks),
        "chunks": chunks,
    }

    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved local vector index: {INDEX_PATH}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
