"""Lessons 16-18: retrieval with self-checks, as plain loops instead of LangGraph graphs.

The course's CRAG falls back to Tavily web search; here the fallback is a rewritten
query against the same corpus, because the benchmark only credits the corpus.
Regeneration is deterministic (temperature 0) and tells the model its previous answer
failed a check, instead of resampling.
"""
from ragfs.core.config import TOP_K
from ragfs.core.fusion import unique
from ragfs.core.types import Result
from ragfs.techniques.base import ANSWER_PROMPT, REFUSAL, Technique, format_context, is_refusal
from ragfs.techniques.query_translation import Decomposition

YES_NO = {"type": "object", "required": ["score"], "properties": {"score": {"type": "string", "enum": ["yes", "no"]}}}

GRADE_DOC = """You are grading whether a retrieved news excerpt is relevant to a question.
If it contains keywords or information related to the question, grade it relevant.
Return JSON {{"score": "yes"}} or {{"score": "no"}}.

Excerpt:
{doc}

Question: {question}"""

GRADE_GROUNDED = """Is the answer below supported by the news excerpts? Return JSON {{"score": "yes"}} or {{"score": "no"}}.

Excerpts:
{context}

Answer: {answer}"""

GRADE_USEFUL = """Does the answer below resolve the question? Return JSON {{"score": "yes"}} or {{"score": "no"}}.

Question: {question}
Answer: {answer}"""

REWRITE = """Rewrite this question into a better search query for finding news articles that answer it.
Keep every named outlet, person, company and date. Reply with the query only.

Question: {question}"""


class Graders:
    """Mixin: yes/no LLM checks. A malformed grade counts as 'yes' so a parse error never discards evidence."""

    def _yes(self, prompt):
        try:
            return self.llm.json(prompt, YES_NO)["score"] == "yes"
        except Exception:
            self.trace["grade_errors"] = self.trace.get("grade_errors", 0) + 1
            return True

    def relevant(self, question, chunks):
        return [c for c in chunks if self._yes(GRADE_DOC.format(doc=c.text, question=question))]

    def rewrite(self, question):
        return self.llm.complete(REWRITE.format(question=question)).splitlines()[0].strip() or question


class CRAG(Graders, Technique):
    name = "crag"
    lesson = "16"
    summary = "Grade each retrieved chunk; if any fail, rewrite the query and retrieve again"

    def answer(self, question):
        self.trace = {}
        docs = self.dense(question)
        keep = self.relevant(question, docs)
        self.trace["relevant_first_pass"] = len(keep)
        if len(keep) < len(docs):
            query = self.rewrite(question)
            self.trace["rewritten"] = query
            extra = [c for c in self.dense(query) if c.id not in {d.id for d in keep}]
            keep = unique(keep + self.relevant(question, extra))
        contexts = keep[:self.k]
        return Result(answer=self.generate(question, contexts), contexts=contexts, trace=self.trace)


class SelfRAG(Graders, Technique):
    name = "self_rag"
    lesson = "17"
    summary = "Retrieve, grade, generate, then check groundedness and usefulness; loop on failure"
    max_rounds = 3

    RETRY_NOTE = "\n\nNote: a previous answer ({prev}) was rejected because it {why}. Answer again carefully."

    def answer(self, question):
        self.trace = {"rounds": []}
        query, answer, contexts = question, REFUSAL, []
        for _ in range(self.max_rounds):
            docs = self.relevant(question, self.dense(query))
            round_ = {"query": query, "relevant": len(docs)}
            self.trace["rounds"].append(round_)
            if not docs:
                query = self.rewrite(query)
                continue
            contexts = docs
            prompt = ANSWER_PROMPT.format(refusal=REFUSAL, context=format_context(docs), question=question)
            answer = self.llm.complete(prompt)
            if not is_refusal(answer) and not self._yes(GRADE_GROUNDED.format(
                    context=format_context(docs), answer=answer)):
                round_["grounded"] = False
                answer = self.llm.complete(prompt + self.RETRY_NOTE.format(
                    prev=answer, why="was not supported by the excerpts"))
            if not is_refusal(answer) and self._yes(GRADE_USEFUL.format(question=question, answer=answer)):
                round_["useful"] = True
                break
            round_["useful"] = False
            query = self.rewrite(query)
        return Result(answer=answer, contexts=contexts, trace=self.trace)


class Adaptive(Technique):
    """Route by complexity: single-article questions go to plain RAG, multi-article to decomposition."""
    name = "adaptive"
    lesson = "18"
    summary = "Classify question complexity; route to single-shot RAG or decomposition"

    PROMPT = """Can this question be answered from a single news article, or does it need facts combined
from several articles? Return JSON {{"route": "single"}} or {{"route": "multi"}}.

Question: {question}"""
    SCHEMA = {"type": "object", "required": ["route"], "properties": {"route": {"type": "string", "enum": ["single", "multi"]}}}

    def __init__(self, kit, k=TOP_K):
        super().__init__(kit, k)
        self.multi = Decomposition(kit, k=k)

    def answer(self, question):
        self.trace = {}
        try:
            route = self.llm.json(self.PROMPT.format(question=question), self.SCHEMA)["route"]
        except Exception:
            route = "single"
        self.trace["route"] = route
        if route == "multi":
            res = self.multi.answer(question)
            res.trace = {"route": route, **res.trace}
            return res
        contexts = self.dense(question)
        return Result(answer=self.generate(question, contexts), contexts=contexts, trace=self.trace)
