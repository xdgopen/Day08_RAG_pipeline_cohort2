"""
Bonus — HyDE (Hypothetical Document Embeddings).

HyDE tạo một tài liệu giả định từ query, embed tài liệu đó, rồi dùng embedding
này để tìm kiếm semantic. Ý tưởng: tài liệu giả định thường gần với văn bản cần
truy xuất hơn là câu hỏi ngắn ban đầu.
"""

try:
    from .task5_semantic_search import semantic_search_by_text
except ImportError:
    from task5_semantic_search import semantic_search_by_text


def generate_hypothetical_document(query: str) -> str:
    """
    Sinh hypothetical document offline, không cần API.

    Với corpus pháp luật + tin tức hiện tại, template này mở rộng query thành
    đoạn văn kiểu tài liệu có các cụm từ thường xuất hiện trong nguồn: quy định,
    hành vi, trách nhiệm, hình phạt, xử lý, người liên quan, ma túy.
    """
    query = query.strip()
    if not query:
        return ""

    return (
        "Tài liệu giả định trả lời câu hỏi về pháp luật và tin tức ma túy. "
        f"Nội dung trọng tâm: {query}. "
        "Tài liệu có thể đề cập đến quy định của Luật Phòng, chống ma túy, "
        "Bộ luật Hình sự, nghị định hướng dẫn, hành vi sử dụng trái phép chất "
        "ma túy, tàng trữ, vận chuyển, mua bán, tổ chức sử dụng, xử lý vi phạm, "
        "trách nhiệm của cơ quan chức năng, biện pháp cai nghiện, quản lý người "
        "sử dụng trái phép chất ma túy, hoặc các bài báo về cá nhân liên quan "
        "đến ma túy. Cần tìm các đoạn nguồn nêu trực tiếp căn cứ, sự kiện, "
        "hình phạt, quy trình hoặc tên người liên quan."
    )


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search bằng HyDE.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    hypothetical_document = generate_hypothetical_document(query)
    if not hypothetical_document or top_k <= 0:
        return []

    results = semantic_search_by_text(
        hypothetical_document,
        top_k=top_k,
        retrieval_method="hyde",
    )

    for result in results:
        result["metadata"] = {
            **result.get("metadata", {}),
            "hyde_query": query,
            "hypothetical_document": hypothetical_document,
        }

    return results


if __name__ == "__main__":
    results = hyde_search("hình phạt tàng trữ trái phép chất ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
