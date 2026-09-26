"""Lessons 5-9: rewrite the question before retrieving.

Every variant hands the generator at most `k` chunks, the same budget as the baseline,
so a gain reflects better retrieval rather than more context.
"""
from ragfs.core.fusion import rrf, unique
from ragfs.core.types import Result
from ragfs.techniques.base import REFUSAL, Technique, format_context

QUERIES_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                  "required": ["queries"]}


def interleave(rankings):
    """Round-robin merge: rank 1 of every list, then rank 2, ... (deduplicated)."""
    out = []
    for i in range(max(map(len, rankings), default=0)):
        out.extend(r[i] for r in rankings if i < len(r))
    return unique(out)


class MultiQuery(Technique):
    name = "multi_query"
    lesson = "5"
    summary = "LLM writes alternative phrasings; union of their results"
    n = 4

    PROMPT = """You are helping search a collection of news articles. Write {n} different versions of the
question below, each phrased differently, to retrieve relevant articles from a vector database.
Return JSON: {{"queries": [...]}}

Question: {question}"""

    def rewrites(self, question):
        try:
            qs = self.llm.json(self.PROMPT.format(n=self.n, question=question), QUERIES_SCHEMA)["queries"]
        except Exception as e:
            self.trace["rewrite_error"] = str(e)
            qs = []
        qs = [q for q in qs if isinstance(q, str) and q.strip()][:self.n]
        self.trace["queries"] = qs
        return [question] + qs

    def retrieve(self, question):
        return interleave([self.dense(q) for q in self.rewrites(question)])[:self.k]


class RagFusion(MultiQuery):
    name = "rag_fusion"
    lesson = "6"
    summary = "Alternative phrasings, fused with reciprocal rank fusion"

    def retrieve(self, question):
        return [c for c, _ in rrf([self.dense(q, k=20) for q in self.rewrites(question)])][:self.k]


SUBQ_PROMPT = """Break the question below into 2 to 4 simpler sub-questions that can each be answered from a
single news article, and whose answers together answer the original question.
Return JSON: {{"queries": [...]}}

Question: {question}"""


class Decomposition(Technique):
    """Answer sub-questions in sequence, each seeing the earlier Q&A pairs, then the whole question."""
    name = "decomposition"
    lesson = "7"
    summary = "Sub-questions answered recursively; final answer builds on them"

    SUB_ANSWER = """Answer the question using the context and the background Q&A pairs. Be brief (one or two sentences).
If they do not contain the answer, say so.

Background Q&A:
{qa}

Context:
{context}

Question: {question}"""

    FINAL = """Use the sub-question answers and the excerpts to answer the main question.
Reply with the answer only: a name or short phrase, or Yes / No for yes-no questions. No explanation.
If there is not enough information, reply exactly: {refusal}

Sub-question answers:
{qa}

Excerpts:
{context}

Question: {question}
Answer:"""

    def subquestions(self, question):
        try:
            qs = self.llm.json(SUBQ_PROMPT.format(question=question), QUERIES_SCHEMA)["queries"]
        except Exception as e:
            self.trace["decompose_error"] = str(e)
            qs = []
        qs = [q for q in qs if isinstance(q, str) and q.strip()][:4] or [question]
        self.trace["subquestions"] = qs
        return qs

    def per_sub_k(self, n):
        return max(2, self.k // n)

    def answer_subs(self, subs):
        qa, rankings = [], []
        for sq in subs:
            ctx = self.dense(sq, k=self.per_sub_k(len(subs)))
            rankings.append(ctx)
            a = self.llm.complete(self.SUB_ANSWER.format(
                qa="\n".join(f"Q: {q}\nA: {a}" for q, a in qa) or "(none)", context=format_context(ctx), question=sq))
            qa.append((sq, a))
        return qa, rankings

    def answer(self, question):
        self.trace = {}
        qa, rankings = self.answer_subs(self.subquestions(question))
        self.trace["sub_answers"] = [a for _, a in qa]
        contexts = interleave(rankings)[:self.k]
        ans = self.llm.complete(self.FINAL.format(
            refusal=REFUSAL, qa="\n".join(f"Q: {q}\nA: {a}" for q, a in qa),
            context=format_context(contexts), question=question))
        return Result(answer=ans, contexts=contexts, trace=self.trace)


class DecompositionIndividual(Decomposition):
    """Answer each sub-question on its own, then synthesise."""
    name = "decomposition_individual"
    lesson = "7"
    summary = "Sub-questions answered independently, then synthesised"

    def answer_subs(self, subs):
        qa, rankings = [], []
        for sq in subs:
            ctx = self.dense(sq, k=self.per_sub_k(len(subs)))
            rankings.append(ctx)
            qa.append((sq, self.llm.complete(self.SUB_ANSWER.format(
                qa="(none)", context=format_context(ctx), question=sq))))
        return qa, rankings


class StepBack(Technique):
    name = "step_back"
    lesson = "8"
    summary = "Also retrieve for a more generic 'step-back' question"

    PROMPT = """You are an expert at world knowledge. Rewrite the question below as a more generic
step-back question that is easier to answer, the way these examples do:

Q: Could the members of The Police perform lawful arrests?
Step-back: What can the members of The Police do?

Q: Did the Verge report that Apple's new iPhone sold out in October 2023?
Step-back: What did news outlets report about Apple's iPhone sales in late 2023?

Q: {question}
Step-back:"""

    def retrieve(self, question):
        step = self.llm.complete(self.PROMPT.format(question=question)).splitlines()[0].strip()
        self.trace["step_back"] = step
        half = self.k // 2
        return unique(self.dense(question, k=self.k - half) + self.dense(step, k=self.k))[:self.k]


class HyDE(Technique):
    name = "hyde"
    lesson = "9"
    summary = "Embed a hypothetical answer passage instead of the question"

    PROMPT = """Write a short news article passage (about 100 words) that answers the question below.
Invent plausible details if you need to.

Question: {question}
Passage:"""

    def retrieve(self, question):
        passage = self.llm.complete(self.PROMPT.format(question=question))
        self.trace["hypothetical"] = passage[:300]
        vec = self.kit.embedder.embed_documents([passage])[0]
        return [c for c, _ in self.kit.store.search_vector(vec, self.k)]
