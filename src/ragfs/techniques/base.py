"""Shared technique interface and the grounded-answer prompt every technique ends with."""
from ragfs.core.config import TOP_K
from ragfs.core.kit import Kit
from ragfs.core.types import Chunk, Result

REFUSAL = "Insufficient information."

# MultiHop-RAG gold answers are a single entity, Yes/No, or the refusal string, so the
# prompt asks for exactly that shape; exact match is then a fair metric.
ANSWER_PROMPT = """Answer the question using only the news excerpts below.
Reply with the answer only: a name or short phrase, or Yes / No for yes-no questions. No explanation.
If the excerpts do not contain enough information, reply exactly: {refusal}

Excerpts:
{context}

Question: {question}
Answer:"""


def format_context(chunks: list[Chunk]) -> str:
    def header(c):
        m = c.metadata
        return f"[{m.get('source', '?')}, {m.get('date', '?')}] {m.get('title', c.doc_id)}"
    return "\n\n---\n\n".join(f"{header(c)}\n{c.text}" for c in chunks)


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith("insufficient information")


class Technique:
    name = "base"
    lesson = ""
    summary = ""

    def __init__(self, kit: Kit, k: int = TOP_K):
        self.kit, self.k = kit, k
        self.llm = kit.llm
        self.trace: dict = {}

    def prepare(self):
        """Build any extra index. Runs before timing starts; its cost is reported separately."""

    def retrieve(self, question: str) -> list[Chunk]:
        raise NotImplementedError

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        return self.llm.complete(ANSWER_PROMPT.format(
            refusal=REFUSAL, context=format_context(contexts), question=question))

    def answer(self, question: str) -> Result:
        self.trace = {}
        contexts = self.retrieve(question)
        return Result(answer=self.generate(question, contexts), contexts=contexts, trace=self.trace)

    def dense(self, query: str, k: int | None = None, where=None) -> list[Chunk]:
        return [c for c, _ in self.kit.store.search(query, k or self.k, where)]
