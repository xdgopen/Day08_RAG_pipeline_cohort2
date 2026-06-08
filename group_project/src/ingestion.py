"""Document ingestion utilities for the RAG chatbot.

Supports adding new knowledge at runtime from:
- Uploaded PDF/DOCX/DOC/MD/TXT/JSON files
- Web links/URLs

Pipeline:
1. Save raw input to data/landing/uploads/
2. Convert/extract to Markdown in data/standardized/uploads/
3. Rebuild local chunk DB data/local_chunks.json
4. Reset in-memory retrieval caches so later queries can retrieve the new docs

The code is dependency-light. If `markitdown` is installed it is used for PDF/DOCX;
otherwise the module falls back to safe text extraction best-effort.
"""

from __future__ import annotations

from datetime import datetime
from html import unescape
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import re
import zipfile

import requests

PROJECT_DIR = Path(__file__).parent.parent
LANDING_UPLOAD_DIR = PROJECT_DIR / "data" / "landing" / "uploads"
STANDARDIZED_UPLOAD_DIR = PROJECT_DIR / "data" / "standardized" / "uploads"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt", ".json", ".html", ".htm"}


def ensure_upload_dirs() -> None:
    LANDING_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STANDARDIZED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_len: int = 80) -> str:
    text = (text or "document").strip().lower()
    text = re.sub(r"[^\w\s.-]+", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    text = text.strip("-.") or "document"
    return text[:max_len]


def _safe_decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "cp1258", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def _clean_html(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    return " ".join(unescape(html).split())


def _meta(html: str, key: str) -> str:
    patterns = [
        fr"<meta[^>]+property=[\"']{re.escape(key)}[\"'][^>]+content=[\"']([^\"']+)[\"']",
        fr"<meta[^>]+name=[\"']{re.escape(key)}[\"'][^>]+content=[\"']([^\"']+)[\"']",
    ]
    for pat in patterns:
        m = re.search(pat, html, re.I)
        if m:
            return unescape(m.group(1)).strip()
    return ""


def extract_html_to_markdown(html: str, url: str = "") -> tuple[str, str]:
    """Extract title + readable markdown-ish content from HTML."""
    title = _meta(html, "og:title") or _meta(html, "twitter:title")
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        title = _clean_html(m.group(1)) if m else (urlparse(url).netloc or "Web document")
    description = _meta(html, "og:description") or _meta(html, "description")
    published = _meta(html, "article:published_time") or _meta(html, "pubdate")

    paragraphs: list[str] = []
    for m in re.finditer(r"<(p|h1|h2|h3|li)[^>]*>(.*?)</\1>", html, re.I | re.S):
        txt = _clean_html(m.group(2))
        if len(txt) >= 35 and txt not in paragraphs:
            paragraphs.append(txt)
    if description and description not in paragraphs:
        paragraphs.insert(0, description)
    if not paragraphs:
        paragraphs = [_clean_html(html)[:6000]]

    md = (
        f"# {title}\n\n"
        f"**Source:** Web upload\n"
        f"**URL:** {url or 'N/A'}\n"
        f"**Published:** {published or 'N/A'}\n"
        f"**Ingested:** {datetime.now().isoformat(timespec='seconds')}\n\n"
        "---\n\n"
        + "\n\n".join(paragraphs[:80])
        + "\n"
    )
    return title, md


def _extract_docx_text(path: Path) -> str:
    """Best-effort DOCX extraction without external dependencies."""
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"<w:tab[^>]*/>", "\t", xml)
        text = re.sub(r"</w:p>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        return unescape(text)
    except Exception:
        return ""


def convert_file_to_markdown(raw_path: Path, original_name: str | None = None) -> tuple[str, str]:
    """Convert a raw uploaded file to (title, markdown)."""
    suffix = raw_path.suffix.lower()
    display_name = original_name or raw_path.name
    title = Path(display_name).stem

    if suffix == ".md":
        content = raw_path.read_text(encoding="utf-8", errors="ignore")
        return title, content if content.lstrip().startswith("#") else f"# {title}\n\n{content}"

    if suffix == ".json":
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        title = data.get("title") or title
        body = data.get("content_markdown") or data.get("markdown") or data.get("content") or json.dumps(data, ensure_ascii=False, indent=2)
        return title, f"# {title}\n\n**Source:** Uploaded JSON\n**Ingested:** {datetime.now().isoformat(timespec='seconds')}\n\n{body}"

    if suffix in {".html", ".htm"}:
        return extract_html_to_markdown(raw_path.read_text(encoding="utf-8", errors="ignore"))

    if suffix in {".txt"}:
        text = raw_path.read_text(encoding="utf-8", errors="ignore")
        return title, f"# {title}\n\n**Source:** Uploaded text file\n**Ingested:** {datetime.now().isoformat(timespec='seconds')}\n\n{text}"

    # Try MarkItDown for PDF/DOCX/DOC and any other supported binary.
    try:
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(raw_path))
        text = getattr(result, "text_content", "") or str(result)
        if text and len(text.strip()) > 20:
            return title, f"# {title}\n\n**Source:** Uploaded {suffix.upper()}\n**Ingested:** {datetime.now().isoformat(timespec='seconds')}\n\n{text}"
    except Exception:
        pass

    if suffix == ".docx":
        text = _extract_docx_text(raw_path)
        if text.strip():
            return title, f"# {title}\n\n**Source:** Uploaded DOCX\n**Ingested:** {datetime.now().isoformat(timespec='seconds')}\n\n{text}"

    # Last fallback: decode bytes. This works for pseudo-PDF/DOCX text files and
    # gives a clear message for encrypted/binary files.
    text = _safe_decode(raw_path.read_bytes())
    if not text.strip():
        text = "Unable to extract text from this file. Please install MarkItDown or upload a text/markdown version."
    return title, f"# {title}\n\n**Source:** Uploaded {suffix.upper()}\n**Ingested:** {datetime.now().isoformat(timespec='seconds')}\n\n{text}"


def _write_standardized_markdown(title: str, markdown: str, prefix: str) -> Path:
    ensure_upload_dirs()
    digest = hashlib.md5((title + markdown[:500]).encode("utf-8", errors="ignore")).hexdigest()[:8]
    md_path = STANDARDIZED_UPLOAD_DIR / f"{prefix}-{slugify(title)}-{digest}.md"
    md_path.write_text(markdown, encoding="utf-8")
    return md_path


def ingest_uploaded_file(filename: str, content: bytes, refresh: bool = True) -> dict:
    """Save + convert uploaded file, then optionally refresh local index."""
    ensure_upload_dirs()
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    raw_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(Path(filename).stem)}{ext}"
    raw_path = LANDING_UPLOAD_DIR / raw_name
    raw_path.write_bytes(content)

    title, markdown = convert_file_to_markdown(raw_path, filename)
    md_path = _write_standardized_markdown(title, markdown, "upload")
    stats = refresh_index() if refresh else {}
    return {"title": title, "raw_path": str(raw_path), "markdown_path": str(md_path), "stats": stats}


def ingest_url(url: str, refresh: bool = True) -> dict:
    """Fetch a web URL, save raw HTML, convert to Markdown, optionally refresh index."""
    ensure_upload_dirs()
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must start with http:// or https://")

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    html = response.text
    title, markdown = extract_html_to_markdown(html, url)

    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    raw_path = LANDING_UPLOAD_DIR / f"url-{slugify(parsed.netloc)}-{digest}.html"
    raw_path.write_text(html, encoding="utf-8", errors="ignore")
    md_path = _write_standardized_markdown(title, markdown, "url")
    stats = refresh_index() if refresh else {}
    return {"title": title, "raw_path": str(raw_path), "markdown_path": str(md_path), "stats": stats}


def refresh_index() -> dict:
    """Rebuild local chunk DB and clear retrieval caches in the current process."""
    from . import task4_chunking_indexing as t4

    docs = t4.load_documents()
    chunks = t4.embed_chunks(t4.chunk_documents(docs))
    t4.index_to_vectorstore(chunks)

    # Reset semantic cache.
    try:
        from . import task5_semantic_search as t5

        t5._CORPUS = None
        t5._DOC_VECS = None
        t5._IDF = None
    except Exception:
        pass

    # Reset lexical cache.
    try:
        from . import task6_lexical_search as t6

        t6.CORPUS = chunks
        t6._BM25 = None
        t6._TFIDF = None
    except Exception:
        pass

    return {"documents": len(docs), "chunks": len(chunks), "index_path": str(t4.INDEX_PATH)}


if __name__ == "__main__":
    print(refresh_index())
