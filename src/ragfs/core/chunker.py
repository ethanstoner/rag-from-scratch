"""Recursive character splitter, the same algorithm the course gets from LangChain.

Split on the coarsest separator that works, recurse into pieces that are still too
long, then greedily pack pieces into chunks of at most `size` characters, carrying
up to `overlap` characters of trailing pieces into the next chunk.
"""
import re

from ragfs.core.config import CHUNK_OVERLAP, CHUNK_SIZE
from ragfs.core.types import Chunk

SEPARATORS = ["\n\n", "\n", " ", ""]


def _split(text: str, separators: list[str], size: int) -> list[str]:
    sep = next((s for s in separators if s == "" or s in text), "")
    rest = separators[separators.index(sep) + 1:] if sep else []
    parts = list(text) if sep == "" else text.split(sep)
    pieces = []
    for i, p in enumerate(parts):
        # Keep the separator attached so chunks join back into the original text.
        piece = p + sep if sep and i < len(parts) - 1 else p
        if not piece:
            continue
        if len(piece) > size and rest:
            pieces.extend(_split(piece, rest, size))
        else:
            pieces.append(piece)
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks, current, length = [], [], 0
    for p in pieces:
        if current and length + len(p) > size:
            chunks.append("".join(current))
            while current and (length > overlap or length + len(p) > size):
                length -= len(current.pop(0))
        current.append(p)
        length += len(p)
    if current:
        chunks.append("".join(current))
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return _merge(_split(text, SEPARATORS, size), size, overlap)


HEADING = re.compile(r"^#{1,6} (.+)$", re.M)


def chunk_document(doc_id: str, text: str, metadata: dict | None = None,
                   size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk one document. Each chunk records every section heading its span falls under."""
    headings = [(m.start(), m.group(1).strip()) for m in HEADING.finditer(text)]
    chunks, cursor = [], 0
    for i, piece in enumerate(split_text(text, size, overlap)):
        start = text.find(piece[:80], max(0, cursor - overlap - 1))
        if start == -1:
            start = cursor
        cursor = start + len(piece)
        opening = next((h for pos, h in reversed(headings) if pos <= start), "")
        inner = [h for pos, h in headings if start < pos < cursor]
        sections = [opening] * bool(opening) + [h for h in inner if h != opening]
        chunks.append(Chunk(id=f"{doc_id}#{i}", text=piece, doc_id=doc_id,
                            metadata={**(metadata or {}), "sections": sections, "start": start}))
    return chunks
