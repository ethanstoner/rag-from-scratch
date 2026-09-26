"""Fetch the Lilian Weng posts and turn each into clean text plus metadata.

Raw HTML is cached under data/raw/, parsed documents under data/docs.jsonl.
"""
import json
import re
from dataclasses import asdict, dataclass

import httpx
from bs4 import BeautifulSoup, NavigableString

from ragfs.core.config import DATA_DIR, POST_URL, POSTS

BLOCK_TAGS = ["p", "li", "pre", "blockquote", "figcaption", "tr", "h1", "h2", "h3", "h4", "h5", "div"]

# Straight quotes keep hand-written eval evidence quotes typeable.
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


@dataclass
class Document:
    id: str
    title: str
    url: str
    date: str
    tags: list[str]
    reading_minutes: int
    text: str

    @property
    def word_count(self):
        return len(self.text.split())

    def metadata(self):
        return {"title": self.title, "url": self.url, "date": self.date, "tags": self.tags,
                "reading_minutes": self.reading_minutes, "word_count": self.word_count}


def clean_text(text: str) -> str:
    text = text.translate(QUOTES)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def parse_post(html: str, slug: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one(".post-title").get_text(" ", strip=True)
    date = soup.select_one('meta[property="article:published_time"]')["content"][:10]
    tags = [a.get_text(strip=True) for a in soup.select(".post-tags a")]
    meta = soup.select_one(".post-meta").get_text(" ", strip=True)
    m = re.search(r"(\d+)\s*min", meta)
    body = soup.select_one(".post-content")
    for bad in body.select("script, style, .toc, a.anchor"):
        bad.decompose()
    for level in range(1, 5):
        for h in body.find_all(f"h{level}"):
            h.insert(0, NavigableString("#" * level + " "))
    for el in body.find_all(BLOCK_TAGS):
        el.insert_before(NavigableString("\n"))
        el.append(NavigableString("\n"))
    return Document(id=slug, title=title, url=POST_URL.format(slug=slug), date=date, tags=tags,
                    reading_minutes=int(m.group(1)) if m else 0, text=clean_text(body.get_text("")))


def fetch_raw(slug: str) -> str:
    path = DATA_DIR / "raw" / f"{slug}.html"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(POST_URL.format(slug=slug), timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def load_corpus(refresh: bool = False) -> list[Document]:
    path = DATA_DIR / "docs.jsonl"
    if path.exists() and not refresh:
        return [Document(**json.loads(ln)) for ln in path.read_text(encoding="utf-8").splitlines()]
    docs = [parse_post(fetch_raw(s), s) for s in POSTS]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(d)) for d in docs) + "\n", encoding="utf-8")
    return docs
