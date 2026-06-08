"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}
PDF_DOC_FALLBACK_URLS = {
    "nghi-dinh-105-2021-huong-dan-thi-hanh-luat-phong-chong-ma-tuy": (
        "https://g7.cdnchinhphu.vn/api/download/stream?"
        "Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0"
        "Gvvr483flbRcmqDlxAHtAi4-m42ig5_ghIhfpCvGkqVGWDgKCY1g9ldqtt"
        "WlRC5DVQZYsDIez2YyFa-Du25zssfc_BNQjzGOg~~&file_name="
        "2021_1047+%2B+1048_105-2021-N%C4%90-CP.doc"
    )
}


def load_markitdown():
    """Load MarkItDown lazily so JSON conversion still works without it."""
    try:
        from markitdown import MarkItDown
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Chưa cài markitdown. Chạy: pip install markitdown"
        ) from error

    return MarkItDown()


def write_markdown(output_path: Path, content: str):
    """Write markdown content with a trailing newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.rstrip() + "\n", encoding="utf-8")


def convert_doc_with_textutil(filepath: Path) -> str:
    """Fallback for old .doc files on macOS."""
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(filepath)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def download_official_doc(url: str, filename: str) -> Path:
    """Download an official DOC fallback to a temporary file."""
    output_path = Path(tempfile.gettempdir()) / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )

    with urlopen(request, timeout=30) as response:
        output_path.write_bytes(response.read())

    return output_path


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = None

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in LEGAL_EXTENSIONS:
            continue

        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"

        try:
            if md is None:
                md = load_markitdown()

            result = md.convert(str(filepath))
            content = result.text_content
            if not content.strip() and filepath.stem in PDF_DOC_FALLBACK_URLS:
                print("  ! PDF không có text, tải DOC chính thức từ Công báo")
                doc_path = download_official_doc(
                    PDF_DOC_FALLBACK_URLS[filepath.stem],
                    f"{filepath.stem}.doc",
                )
                content = convert_doc_with_textutil(doc_path)
        except Exception:
            if filepath.suffix.lower() != ".doc":
                raise

            print("  ! MarkItDown không convert được .doc, dùng textutil fallback")
            content = convert_doc_with_textutil(filepath)

        write_markdown(output_path, content)
        print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            write_markdown(output_path, content)
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
