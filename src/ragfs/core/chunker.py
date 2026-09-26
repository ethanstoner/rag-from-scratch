"""Recursive character splitter, the same algorithm the course gets from LangChain.

Split on the coarsest separator that works, recurse into pieces that are still too
long, then greedily pack pieces into chunks of at most `size` characters, carrying
up to `overlap` characters of trailing pieces into the next chunk. Pieces carry
their offsets so every chunk knows exactly where it sits in the source text.
"""
import re

from ragfs.core.config import CHUNK_OVERLAP, CHUNK_SIZE
from ragfs.core.types import Chunk

SEPARATORS = ["\n\n", "\n", " ", ""]


def _split(text: str, offset: int, separators: list[str], size: int) -> list[tuple[int, str]]:
    sep = next((s for s in separators if s == "" or s in text), "")
    rest = separators[separators.index(sep) + 1:] if sep else []
    parts = list(text) if sep == "" else text.split(sep)
    pieces, pos = [], offset
    for i, p in enumerate(parts):
        # Keep the separator attached so chunks join back into the original text.
        piece = p + sep if sep and i < len(parts) - 1 else p
        if piece:
            if len(piece) > size and rest:
                pieces.extend(_split(piece, pos, rest, size))
            else:
                pieces.append((pos, piece))
        pos += len(piece)
    return pieces


def _merge(pieces: list[tuple[int, str]], size: int, overlap: int) -> list[tuple[int, str]]:
    chunks, current, length = [], [], 0

    def emit():
        raw = "".join(p for _, p in current)
        stripped = raw.strip()
        if stripped:
            chunks.append((current[0][0] + len(raw) - len(raw.lstrip()), stripped))

    for pos, p in pieces:
        if current and length + len(p) > size:
            emit()
            while current and (length > overlap or length + len(p) > size):
                length -= len(current.pop(0)[1])
        current.append((pos, p))
        length += len(p)
    if current:
        emit()
    return chunks


def split_spans(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    return _merge(_split(text, 0, SEPARATORS, size), size, overlap)


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return [t for _, t in split_spans(text, size, overlap)]


HEADING = re.compile(r"^#{1,6} (.+)$", re.M)


def chunk_document(doc_id: str, text: str, metadata: dict | None = None,
                   size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk one document. Each chunk records its span and any section headings it falls under."""
    headings = [(m.start(), m.group(1).strip()) for m in HEADING.finditer(text)]
    chunks = []
    for i, (start, piece) in enumerate(split_spans(text, size, overlap)):
        end = start + len(piece)
        opening = next((h for pos, h in reversed(headings) if pos <= start), "")
        inner = [h for pos, h in headings if start < pos < end]
        sections = [opening] * bool(opening) + [h for h in inner if h != opening]
        chunks.append(Chunk(id=f"{doc_id}#{i}", text=piece, doc_id=doc_id,
                            metadata={**(metadata or {}), "sections": sections, "start": start, "end": end}))
    return chunks
