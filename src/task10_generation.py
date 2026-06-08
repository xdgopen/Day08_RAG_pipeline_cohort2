"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PLACEHOLDER_KEYS = {"", "sk-xxx", "xxx"}
MIN_CONTEXT_SCORE = 0.05
CITATION_DEFAULT_YEAR = os.getenv("CITATION_DEFAULT_YEAR", "2026")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context, state
'I cannot verify this information' rather than guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = []
    back = []
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            front.append(chunk)
        else:
            back.append(chunk)

    return front + list(reversed(back))


def infer_year(source: str, content: str) -> str:
    """Infer citation year from source path/name or chunk content."""
    match = re.search(r"(20\d{2}|19\d{2})", f"{source}\n{content}")
    if match:
        return match.group(1)
    return CITATION_DEFAULT_YEAR


def source_label(chunk: dict, index: int) -> str:
    """Build citation label dạng [Nguồn, Năm]."""
    metadata = chunk.get("metadata", {})
    source = (
        metadata.get("source")
        or metadata.get("source_name")
        or metadata.get("doc_id")
        or f"Source {index}"
    )
    source_name = str(source).split("/")[-1].replace(".md", "").replace(".pdf", "")
    year = infer_year(str(source), chunk.get("content", ""))
    return f"{source_name}, {year}"


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        citation = source_label(chunk, i)
        source = metadata.get("source") or metadata.get("source_name") or "unknown"
        doc_type = metadata.get("doc_type") or metadata.get("type") or "unknown"
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Document {i} | Citation: {citation} | Source: {source} | "
            f"Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '').strip()}\n"
        )
    return "\n---\n".join(context_parts)


def has_openai_api_key() -> bool:
    """Return True when .env contains a real-looking OpenAI API key."""
    return OPENAI_API_KEY not in PLACEHOLDER_KEYS and OPENAI_API_KEY.startswith("sk-")


def build_user_message(query: str, context: str) -> str:
    """Build prompt payload containing evidence and question."""
    return f"""Context:
{context}

---

Question: {query}

Answer in Vietnamese. Use citations exactly as provided in each Document header."""


def extractive_answer(query: str, chunks: list[dict]) -> str:
    """
    Offline fallback answer.

    It does not invent facts: it quotes/summarizes the top retrieved chunks and
    attaches source-year citations so demos still satisfy citation behavior when
    no LLM key is configured.
    """
    if not chunks or chunks[0].get("score", 0.0) < MIN_CONTEXT_SCORE:
        return "I cannot verify this information"

    sentences = []
    for i, chunk in enumerate(chunks[:3], 1):
        content = " ".join(chunk.get("content", "").split())
        if not content:
            continue

        excerpt = content[:450].rstrip()
        if len(content) > 450:
            excerpt += "..."
        sentences.append(f"- {excerpt} [{source_label(chunk, i)}]")

    if not sentences:
        return "I cannot verify this information"

    return (
        "Dựa trên các nguồn đã truy xuất, các điểm liên quan nhất là:\n"
        + "\n".join(sentences)
    )


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip()
    if not query:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
        }

    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=top_k)
    chunks = chunks[:top_k]
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = build_user_message(query, context)

    if not chunks:
        answer = "I cannot verify this information"
    elif has_openai_api_key():
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
    else:
        answer = extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "prompt": user_message,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
