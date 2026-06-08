"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
from html.parser import HTMLParser
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://baovephapluat.vn/cong-to-kiem-sat-tu-phap/truy-to/truy-to-ca-si-chi-dan-va-226-bi-can-trong-vu-an-ma-tuy-lien-quan-den-tiep-vien-hang-khong-196299.html",
    "https://vov.vn/phap-luat/cu-truot-dai-cua-ca-si-chi-dan-khi-ru-re-nhom-ban-su-dung-ma-tuy-post1287890.vov",
    "https://vov.vn/giai-tri/long-nhat-son-ngoc-minh-bi-bat-vi-ma-tuy-nsut-hanh-thuy-len-tieng-canh-bao-post1293528.vov",
    "https://tuoitre.vn/rapper-binh-gold-bi-bat-vi-cuop-tai-san-duong-tinh-voi-ma-tuy-20250726185902989.htm",
    "https://thanhnien.vn/ntk-nguyen-cong-tri-bi-bat-vi-ma-tuy-dung-khoa-lap-cho-sai-pham-bang-tai-nang-185250724101540772.htm",
]


class ArticleHTMLParser(HTMLParser):
    """Extract title and readable article text from basic news HTML."""

    CONTENT_TAGS = {"h1", "h2", "h3", "p", "li"}
    SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__()
        self.title = ""
        self._current_tag = None
        self._skip_depth = 0
        self._title_parts = []
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth:
            return

        if tag == "title" or tag in self.CONTENT_TAGS:
            self._current_tag = tag

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return

        if tag == "title":
            self.title = " ".join(self._title_parts).strip()

        if tag == self._current_tag:
            self._current_tag = None

    def handle_data(self, data):
        if self._skip_depth or not self._current_tag:
            return

        text = " ".join(data.split())
        if not text:
            return

        if self._current_tag == "title":
            self._title_parts.append(text)
        else:
            self._text_parts.append(text)

    def to_markdown(self):
        paragraphs = []
        seen = set()

        for text in self._text_parts:
            if len(text) < 20 or text in seen:
                continue
            paragraphs.append(text)
            seen.add(text)

        return "\n\n".join(paragraphs)


def crawl_article_with_urllib(url: str) -> dict:
    """Fallback crawler using only the Python standard library."""
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
        html = response.read().decode("utf-8", errors="replace")

    parser = ArticleHTMLParser()
    parser.feed(html)

    return {
        "url": url,
        "title": parser.title or "Unknown",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": parser.to_markdown(),
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except (ModuleNotFoundError, TypeError) as error:
        print(f"  ! Không dùng được crawl4ai ({error}), dùng urllib fallback")
        return await asyncio.to_thread(crawl_article_with_urllib, url)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return {
            "url": url,
            "title": result.metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
