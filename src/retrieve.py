"""Ba cách truy xuất: vector, BM25, hybrid (RRF)."""
import json, re, hashlib
from pathlib import Path
import numpy as np

def load_chunks(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def tokenize(text, segment=True):
    text = text.lower()
    if segment:
        try:
            from underthesea import word_tokenize
            toks = word_tokenize(text)
        except Exception:
            toks = text.split()
    else:
        toks = text.split()
    return [t for t in (re.sub(r"[^\w]+", "", t.replace(" ", "_")) for t in toks) if t]

class Retriever:
    def __init__(self, chunks, cfg):
        self.chunks, self.cfg = chunks, cfg
        self._bm25 = self._emb = self._model = None

    # ---- BM25 ----
    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        seg = self.cfg["bm25"]["word_segment"]
        self._bm25 = BM25Okapi([tokenize(c["text"], seg) for c in self.chunks])

    def bm25(self, query, k):
        if self._bm25 is None: self._build_bm25()
        scores = self._bm25.get_scores(tokenize(query, self.cfg["bm25"]["word_segment"]))
        return self._topk(scores, k)

    # ---- Vector ----
    def _build_emb(self):
        from sentence_transformers import SentenceTransformer
        e = self.cfg["embedding"]
        self._model = SentenceTransformer(e["model"])
        key = hashlib.md5((e["model"] + "|" + "".join(c["id"] + c["text"] for c in self.chunks)).encode()).hexdigest()
        cache = Path(e["cache_dir"]); cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"{key}.npy"
        if f.exists():
            self._emb = np.load(f)
        else:
            texts = [e["passage_prefix"] + c["text"] for c in self.chunks]
            self._emb = self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            np.save(f, self._emb)

    def vector(self, query, k):
        if self._emb is None: self._build_emb()
        q = self._model.encode([self.cfg["embedding"]["query_prefix"] + query], normalize_embeddings=True)[0]
        return self._topk(self._emb @ q, k)

    # ---- Hybrid: Reciprocal Rank Fusion ----
    def hybrid(self, query, k, pool=20):
        rrf_k = self.cfg["hybrid"]["rrf_k"]
        fused = {}
        for ranked in (self.vector(query, pool), self.bm25(query, pool)):
            for rank, (idx, _) in enumerate(ranked, 1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]

    def search(self, method, query, k):
        res = {"vector": self.vector, "bm25": self.bm25, "hybrid": self.hybrid}[method](query, k)
        return [(self.chunks[i], float(s)) for i, s in res]

    @staticmethod
    def _topk(scores, k):
        scores = np.asarray(scores)
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]
