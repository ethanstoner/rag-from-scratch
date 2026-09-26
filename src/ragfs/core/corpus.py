"""The MultiHop-RAG news corpus (Tang & Yang, 2024; ODC-BY): 609 articles, Sep-Dec 2023."""
import json
import re
from dataclasses import dataclass

import httpx

from ragfs.core.config import DATA_DIR, DATASET_URL

FILES = ("corpus.json", "MultiHopRAG.json")

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Straight quotes, single spaces, at most one blank line. Paragraph breaks survive for the chunker."""
    text = re.sub(r"[ \t ]+", " ", text.translate(QUOTES))
    return re.sub(r"\s*\n\s*\n\s*", "\n\n", text).strip()


def locate(fact: str, text: str) -> tuple[int, int] | None:
    """Character span of `fact` in `text`, tolerant of whitespace differences."""
    words = normalize(fact).split()
    m = re.search(r"\s+".join(map(re.escape, words)), text)
    return (m.start(), m.end()) if m else None


@dataclass
class Document:
    id: str
    title: str
    url: str
    source: str
    category: str
    author: str
    published_at: str
    text: str

    @property
    def date(self):
        return self.published_at[:10]

    def metadata(self):
        return {"title": self.title, "source": self.source, "category": self.category,
                "date": self.date, "url": self.url}


def download(refresh: bool = False):
    folder = DATA_DIR / "multihop"
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = folder / name
        if refresh or not path.exists():
            r = httpx.get(DATASET_URL.format(file=name), timeout=300.0, follow_redirects=True)
            r.raise_for_status()
            path.write_bytes(r.content)
    return folder


def load_raw(name: str):
    return json.loads((download() / name).read_text(encoding="utf-8"))


def load_corpus() -> list[Document]:
    return [Document(id=f"a{i:03d}", title=a["title"], url=a["url"], source=a["source"],
                     category=a["category"], author=a["author"], published_at=a["published_at"],
                     text=normalize(a["body"]))
            for i, a in enumerate(load_raw("corpus.json"))]
