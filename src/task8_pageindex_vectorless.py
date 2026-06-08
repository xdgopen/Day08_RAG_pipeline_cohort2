"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_POLL_SECONDS = float(os.getenv("PAGEINDEX_POLL_SECONDS", "2"))
PAGEINDEX_MAX_POLLS = int(os.getenv("PAGEINDEX_MAX_POLLS", "30"))
PAGEINDEX_DOC_IDS = [
    doc_id.strip()
    for doc_id in os.getenv("PAGEINDEX_DOC_IDS", "").split(",")
    if doc_id.strip()
]
PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
PAGEINDEX_DIR = PROJECT_ROOT / "data" / "pageindex"
REGISTRY_PATH = PAGEINDEX_DIR / "pageindex_docs.json"
UPLOAD_EXTENSIONS = {".pdf", ".doc", ".docx", ".md"}


def has_pageindex_api_key() -> bool:
    """Return True khi .env chứa API key thật, không phải placeholder."""
    return bool(PAGEINDEX_API_KEY and PAGEINDEX_API_KEY not in {"pi_xxx", "xxx"})


def get_pageindex_client():
    """Create PageIndex SDK client."""
    if not has_pageindex_api_key():
        raise RuntimeError(
            "PAGEINDEX_API_KEY chưa được set hoặc vẫn là placeholder. "
            "Hãy cập nhật .env bằng API key thật từ https://pageindex.ai/"
        )

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def load_registry() -> dict:
    """Load local mapping source file -> PageIndex doc_id."""
    if not REGISTRY_PATH.exists():
        return {"documents": []}

    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(registry: dict):
    """Persist local PageIndex upload registry."""
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_upload_files() -> list[Path]:
    """
    Return documents to upload.

    PageIndex is strongest on document files (PDF/DOC). We upload legal source
    files from data/landing/legal and Markdown news/legal files from
    data/standardized so the fallback can cover the same corpus.
    """
    files = []
    for base_dir in (LANDING_DIR / "legal", STANDARDIZED_DIR):
        if not base_dir.exists():
            continue

        for file_path in sorted(base_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in UPLOAD_EXTENSIONS:
                files.append(file_path)

    return files


def get_registered_doc_ids() -> list[str]:
    """Get doc IDs from env override or upload registry."""
    if PAGEINDEX_DOC_IDS:
        return PAGEINDEX_DOC_IDS

    registry = load_registry()
    return [
        item["doc_id"]
        for item in registry.get("documents", [])
        if item.get("doc_id")
    ]


def extract_doc_id(response: dict) -> str | None:
    """Support slight response-shape differences from PageIndex."""
    if not isinstance(response, dict):
        return None

    return (
        response.get("doc_id")
        or response.get("id")
        or response.get("document_id")
        or response.get("document", {}).get("id")
    )


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = get_pageindex_client()
    registry = load_registry()
    existing_by_source = {
        item["source"]: item
        for item in registry.get("documents", [])
        if item.get("source") and item.get("doc_id")
    }

    uploaded = []
    skipped = []
    failed = []

    for file_path in iter_upload_files():
        source = str(file_path.relative_to(PROJECT_ROOT))
        if source in existing_by_source:
            skipped.append(existing_by_source[source])
            print(f"  - Skipped existing: {source}")
            continue

        print(f"  Uploading: {source}")
        try:
            response = client.submit_document(str(file_path))
            doc_id = extract_doc_id(response)
            if not doc_id:
                raise RuntimeError(f"Không lấy được doc_id từ response: {response}")

            item = {
                "source": source,
                "filename": file_path.name,
                "doc_type": file_path.parent.name,
                "doc_id": doc_id,
                "status": "uploaded",
            }
            uploaded.append(item)
            registry.setdefault("documents", []).append(item)
            save_registry(registry)
            print(f"  ✓ Uploaded: {source} -> {doc_id}")
        except Exception as error:
            failed.append({"source": source, "error": str(error)})
            print(f"  ! Failed: {source} ({error})")

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "failed": failed,
        "registry_path": str(REGISTRY_PATH),
    }


def extract_retrieval_id(response: dict) -> str | None:
    """Support slight response-shape differences for retrieval submit."""
    if not isinstance(response, dict):
        return None

    return (
        response.get("retrieval_id")
        or response.get("id")
        or response.get("retrieval", {}).get("id")
    )


def wait_for_retrieval(client, retrieval_id: str) -> dict:
    """Poll PageIndex retrieval until ready or timeout."""
    last_response = {}
    for _ in range(PAGEINDEX_MAX_POLLS):
        last_response = client.get_retrieval(retrieval_id)
        status = str(last_response.get("status", "")).lower()
        if status in {"completed", "complete", "ready", "succeeded", "success"}:
            return last_response
        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex retrieval failed: {last_response}")

        if any(key in last_response for key in ("results", "blocks", "nodes", "answer")):
            return last_response

        time.sleep(PAGEINDEX_POLL_SECONDS)

    raise TimeoutError(f"Timed out waiting for retrieval {retrieval_id}: {last_response}")


def result_text(item: dict) -> str:
    """Extract text from common PageIndex result shapes."""
    return (
        item.get("text")
        or item.get("content")
        or item.get("markdown")
        or item.get("answer")
        or item.get("node", {}).get("text")
        or item.get("node", {}).get("content")
        or ""
    )


def normalize_retrieval_response(response: dict, doc_id: str) -> list[dict]:
    """Convert PageIndex response variants into retrieval result dicts."""
    raw_results = (
        response.get("results")
        or response.get("blocks")
        or response.get("nodes")
        or response.get("retrieval_results")
        or []
    )

    if isinstance(raw_results, dict):
        raw_results = list(raw_results.values())

    normalized = []
    for rank, item in enumerate(raw_results, start=1):
        if not isinstance(item, dict):
            continue

        content = result_text(item).strip()
        if not content:
            continue

        normalized.append(
            {
                "content": content,
                "score": float(item.get("score") or item.get("relevance_score") or 1 / rank),
                "metadata": {
                    "doc_id": doc_id,
                    "rank": rank,
                    "page": item.get("page") or item.get("page_index"),
                    "node_id": item.get("node_id") or item.get("id"),
                    "raw_metadata": item.get("metadata", {}),
                },
                "source": "pageindex",
            }
        )

    answer = response.get("answer")
    if answer and not normalized:
        normalized.append(
            {
                "content": str(answer).strip(),
                "score": 1.0,
                "metadata": {"doc_id": doc_id, "rank": 1},
                "source": "pageindex",
            }
        )

    return normalized


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    client = get_pageindex_client()
    doc_ids = get_registered_doc_ids()
    if not doc_ids:
        raise RuntimeError(
            "Chưa có PageIndex doc_id. Hãy chạy upload_documents() trước "
            "hoặc set PAGEINDEX_DOC_IDS trong .env."
        )

    results = []
    for doc_id in doc_ids:
        submitted = client.submit_query(doc_id=doc_id, query=query, thinking=True)
        retrieval_id = extract_retrieval_id(submitted)
        if not retrieval_id:
            raise RuntimeError(f"Không lấy được retrieval_id từ response: {submitted}")

        response = wait_for_retrieval(client, retrieval_id)
        results.extend(normalize_retrieval_response(response, doc_id))

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not has_pageindex_api_key():
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
