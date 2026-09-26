"""Shared technique interface and the grounded-answer prompt every technique ends with."""
from ragfs.core.config import TOP_K
from ragfs.core.kit import Kit
from ragfs.core.types import Chunk, Result

REFUSAL = "I don't know."

ANSWER_PROMPT = """Answer the question using only the context below. Be concise.
If the context does not contain the answer, reply exactly: {refusal}

Context:
{context}

Question: {question}"""


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n---\n\n".join(f"[{c.metadata.get('title', c.doc_id)}]\n{c.text}" for c in chunks)


def is_refusal(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith("i don't know") or a.startswith("i do not know")


class Technique:
    name = "base"
    lesson = ""
    summary = ""

    def __init__(self, kit: Kit, k: int = TOP_K):
        self.kit, self.k = kit, k
        self.llm = kit.llm
        self.trace: dict = {}

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
