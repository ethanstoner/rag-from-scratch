"""Lesson 14 (ColBERT) and lesson 15's re-ranking, on local GPU models via transformers.

ColBERT here is the colbertv2.0 checkpoint driven by hand: token embeddings from its
BERT encoder + 128-d projection, and late-interaction MaxSim scoring written in torch
(no colbert-ai / RAGatouille). Search is exhaustive over every document token.
"""
import string

import torch
from torch.nn import functional as F

from ragfs.core.config import CACHE_DIR
from ragfs.core.fusion import rrf
from ragfs.techniques.base import Technique

COLBERT = "colbert-ir/colbertv2.0"
RERANKER = "BAAI/bge-reranker-v2-m3"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ColBERTEncoder:
    Q_MARKER, D_MARKER = "[unused0]", "[unused1]"

    def __init__(self, query_maxlen=32, doc_maxlen=300):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, BertConfig, BertModel
        self.tok = AutoTokenizer.from_pretrained(COLBERT)
        weights = load_file(hf_hub_download(COLBERT, "model.safetensors"))
        self.bert = BertModel(BertConfig.from_pretrained(COLBERT), add_pooling_layer=False)
        self.bert.load_state_dict({k[5:]: v for k, v in weights.items()
                                   if k.startswith("bert.") and not k.startswith("bert.pooler")}, strict=False)
        self.linear = torch.nn.Linear(768, 128, bias=False)
        self.linear.weight.data = weights["linear.weight"]
        self.dev = _device()
        self.bert.to(self.dev).eval().half()
        self.linear.to(self.dev).half()
        self.qlen, self.dlen = query_maxlen, doc_maxlen
        self.q_id, self.d_id = self.tok.convert_tokens_to_ids([self.Q_MARKER, self.D_MARKER])
        self.skip = {self.tok.convert_tokens_to_ids(p) for p in string.punctuation} | {self.tok.pad_token_id}

    def _tokenize(self, texts, marker, maxlen, pad_to_max):
        # ". " reserves position 1 for the marker token, as colbert-ai does.
        enc = self.tok([". " + t for t in texts], padding="max_length" if pad_to_max else "longest",
                       truncation=True, max_length=maxlen, return_tensors="pt")
        enc["input_ids"][:, 1] = marker
        return enc

    @torch.inference_mode()
    def _encode(self, enc):
        enc = {k: v.to(self.dev) for k, v in enc.items()}
        h = self.bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        return F.normalize(self.linear(h).float(), dim=-1)

    def queries(self, texts):
        enc = self._tokenize(texts, self.q_id, self.qlen, pad_to_max=True)
        # Query augmentation: pad with [MASK]; the masks are embedded but not attended to.
        pad = enc["attention_mask"] == 0
        enc["input_ids"][pad] = self.tok.mask_token_id
        return self._encode(enc)

    def docs(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = self._tokenize(texts[i:i + batch], self.d_id, self.dlen, pad_to_max=False)
            emb = self._encode(enc)
            keep = enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], torch.tensor(sorted(self.skip)))
            for e, k in zip(emb, keep.to(emb.device)):
                out.append(e[k].half())
        return out


class ColBERTIndex:
    """All document tokens in one matrix; MaxSim via a segmented max over token scores."""

    def __init__(self, encoder, chunks):
        self.enc, self.chunks = encoder, chunks
        path = CACHE_DIR / "colbert_index.pt"
        ids = [c.id for c in chunks]
        if path.exists():
            saved = torch.load(path)
            if saved["ids"] == ids:
                self.tokens, self.owner = saved["tokens"].to(encoder.dev), saved["owner"].to(encoder.dev)
                return
        per_doc = encoder.docs([c.text for c in chunks])
        self.tokens = torch.cat(per_doc)
        self.owner = torch.cat([torch.full((len(e),), i, dtype=torch.long, device=e.device)
                                for i, e in enumerate(per_doc)])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids, "tokens": self.tokens.cpu(), "owner": self.owner.cpu()}, path)

    @torch.inference_mode()
    def search(self, query, k):
        q = self.enc.queries([query])[0].half()                   # (32, 128)
        sims = (q @ self.tokens.T).float()                         # (32, n_tokens)
        best = torch.full((q.shape[0], len(self.chunks)), -1e4, device=sims.device)
        best = best.scatter_reduce(1, self.owner.expand_as(sims), sims, reduce="amax")
        scores = best.sum(0)                                       # MaxSim, summed over query tokens
        top = torch.topk(scores, k)
        return [(self.chunks[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())]


class ColBERT(Technique):
    name = "colbert"
    lesson = "14"
    summary = "Token-level late interaction (ColBERTv2 MaxSim) instead of one vector per chunk"

    def prepare(self):
        if "colbert" not in self.kit.extras:
            self.kit.extras["colbert"] = ColBERTIndex(ColBERTEncoder(), self.kit.chunks)

    def retrieve(self, question):
        self.prepare()
        return [c for c, _ in self.kit.extras["colbert"].search(question, self.k)]


class CrossEncoder:
    def __init__(self, model=RERANKER):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, dtype=torch.float16)
        self.dev = _device()
        self.model.to(self.dev).eval()

    @torch.inference_mode()
    def scores(self, query, texts, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok([[query, t] for t in texts[i:i + batch]], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.dev)
            out.extend(self.model(**enc).logits.view(-1).float().tolist())
        return out


class Rerank(Technique):
    """Hybrid (dense + BM25) candidates, re-ordered by a cross-encoder."""
    name = "rerank"
    lesson = "15"
    summary = "Cross-encoder (bge-reranker-v2-m3) re-ranks 30 hybrid candidates"
    candidates = 30

    def prepare(self):
        if "reranker" not in self.kit.extras:
            self.kit.extras["reranker"] = CrossEncoder()

    def retrieve(self, question):
        self.prepare()
        dense = self.dense(question, k=self.candidates)
        sparse = [c for c, _ in self.kit.bm25.search(question, self.candidates)]
        pool = [c for c, _ in rrf([dense, sparse])][:self.candidates]
        scores = self.kit.extras["reranker"].scores(question, [c.text for c in pool])
        ranked = sorted(zip(pool, scores), key=lambda x: -x[1])
        return [c for c, _ in ranked[:self.k]]
