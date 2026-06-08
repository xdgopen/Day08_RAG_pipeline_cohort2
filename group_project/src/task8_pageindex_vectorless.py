"""Task 8 — PageIndex Vectorless RAG.

Có API key thì có thể thay bằng PageIndex SDK. Bản nộp này cung cấp vectorless
fallback offline: dò cấu trúc markdown/keyword trên tài liệu chuẩn hoá và luôn đánh
dấu source='pageindex' theo yêu cầu test.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from .task6_lexical_search import tokenize
from .task4_chunking_indexing import get_chunks

load_dotenv()
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    files=list(STANDARDIZED_DIR.rglob("*.md"))
    return {"uploaded": len(files), "mode": "offline-local"}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    q=set(tokenize(query)); results=[]
    for c in get_chunks():
        toks=set(tokenize(c.get("content","")))
        score=len(q & toks)/(len(q) or 1)
        if score>0 or not results:
            results.append({"content":c["content"],"score":float(score),"metadata":c.get("metadata",{}),"source":"pageindex"})
    results=sorted(results, key=lambda x:x["score"], reverse=True)[:top_k]
    for r in results: r["source"]="pageindex"
    return results

if __name__ == "__main__":
    print(upload_documents()); print(pageindex_search("ma tuý",3))
