"""Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Có Crawl4AI thì crawl URL thật; khi offline, tạo article JSON mẫu có đầy đủ metadata
để pipeline và tests hoạt động.
"""
import asyncio, json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
ARTICLE_URLS = [
    # Các bài thật/mới từ Tuổi Trẻ Online về nghệ sĩ/người nổi tiếng Việt Nam
    # bị bắt/truy tố/liên quan ma túy. Danh sách này được dùng để crawl lại khi cần.
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://tuoitre.vn/long-nhat-bi-bat-vi-ma-tuy-la-ba-tam-showbiz-tung-co-tinh-tao-loat-scandal-de-noi-tieng-2026052012470757.htm",
    "https://tuoitre.vn/ca-si-son-ngoc-minh-truoc-khi-bi-bat-vi-ma-tuy-nguoi-nha-mat-lien-lac-bo-be-ca-hat-nhieu-nam-20260520133233973.htm",
    "https://tuoitre.vn/ca-si-miu-le-bi-bat-qua-tang-su-dung-ma-tuy-o-hai-phong-20260511172700149.htm",
    "https://tuoitre.vn/chuyen-an-vn10-truy-to-227-bi-can-trong-do-co-ca-si-chi-dan-an-tay-2026040308051239.htm",
    "https://tuoitre.vn/vu-4-tiep-vien-hang-khong-xach-tay-ma-tuy-tiec-sinh-nhat-bay-lac-cua-co-tien-truc-phuong-20260413135309212.htm",
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
]

def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

async def crawl_article(url: str) -> dict:
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return {"url":url,"title":getattr(result,"metadata",{}).get("title",url),"date_crawled":datetime.now().isoformat(),"content_markdown":getattr(result,"markdown","")}
    except Exception:
        # Fallback không dùng LLM/API: tải HTML thật và trích xuất meta + các đoạn <p>.
        # Vẫn giữ URL thật, tiêu đề thật nếu site trả metadata.
        import re
        from html import unescape
        import requests

        def clean_html(s: str) -> str:
            s = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", s, flags=re.I)
            s = re.sub(r"<[^>]+>", " ", s)
            return " ".join(unescape(s).split())

        def get_meta(html: str, key: str) -> str:
            patterns = [
                fr"<meta[^>]+property=[\"']{re.escape(key)}[\"'][^>]+content=[\"']([^\"']+)[\"']",
                fr"<meta[^>]+name=[\"']{re.escape(key)}[\"'][^>]+content=[\"']([^\"']+)[\"']",
            ]
            for pat in patterns:
                m = re.search(pat, html, re.I)
                if m:
                    return unescape(m.group(1)).strip()
            return ""

        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25).text
        title = get_meta(html, "og:title") or url.rstrip('/').split('/')[-1].replace('-', ' ').title()
        published = get_meta(html, "article:published_time")
        paragraphs = []
        for m in re.finditer(r"<p[^>]*>(.*?)</p>", html, re.S | re.I):
            txt = clean_html(m.group(1))
            if len(txt) > 40 and txt not in paragraphs:
                paragraphs.append(txt)
        content = "\n\n".join(paragraphs[:30])
        return {
            "url": url,
            "title": title,
            "source": "Tuổi Trẻ Online",
            "date_published": published,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"# {title}\n\n**Source:** Tuổi Trẻ Online\n**URL:** {url}\n**Published:** {published}\n\n{content}",
        }

async def crawl_all():
    setup_directory()
    for i,url in enumerate(ARTICLE_URLS,1):
        article=await crawl_article(url)
        (DATA_DIR/f"article_{i:02d}.json").write_text(json.dumps(article,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(crawl_all())
