"""Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Hỗ trợ JSON/HTML/TXT và fallback đọc text cho PDF/DOCX mô phỏng trong repo. Nếu cài
MarkItDown thật, có thể dùng cho tài liệu binary thực tế.
"""
import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

try:
    from markitdown import MarkItDown
except Exception:
    MarkItDown = None


def _read_text_or_convert(filepath: Path) -> str:
    try:
        return filepath.read_text(encoding="utf-8")
    except Exception:
        if MarkItDown:
            return MarkItDown().convert(str(filepath)).text_content
        return filepath.read_bytes().decode("utf-8", errors="ignore")


def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"; output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            content=_read_text_or_convert(filepath)
            (output_dir / f"{filepath.stem}.md").write_text(content, encoding="utf-8")


def convert_news_articles():
    news_dir = LANDING_DIR / "news"; output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            data=json.loads(filepath.read_text(encoding="utf-8"))
            header=f"# {data.get('title','Unknown')}\n\n**Source:** {data.get('url','N/A')}\n**Crawled:** {data.get('date_crawled','N/A')}\n**Year:** {data.get('year','N/A')}\n\n---\n\n"
            content=header+data.get("content_markdown", data.get("content", ""))
        else:
            content=_read_text_or_convert(filepath)
        if filepath.suffix.lower() in (".json", ".html", ".txt", ".md"):
            (output_dir / f"{filepath.stem}.md").write_text(content, encoding="utf-8")


def convert_all():
    convert_legal_docs(); convert_news_articles(); print(f"Done: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_all()
