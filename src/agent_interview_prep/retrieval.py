"""Deterministic TF-IDF retrieval over document chunks.

Why not LlamaIndex's built-in indexes in mock mode? KeywordTableIndex pulls
NLTK stopwords data over the network, and VectorStoreIndex needs a real
embedding model to rank anything — MockEmbedding returns near-identical
vectors, so its "ranking" is meaningless. So mock mode uses this small
pure-Python TF-IDF index behind the same retrieve(query, top_k) interface.
Live mode (live.py) swaps in a LlamaIndex VectorStoreIndex with Gemini
embeddings. The README states this plainly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .documents import Chunk

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "their this to was were will with you your we our they them then than".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*")


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("./-")
        if tok and tok not in _STOPWORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity, 0..1


class TfidfRetriever:
    """TF-IDF + cosine similarity over a fixed set of chunks. Deterministic."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._doc_tokens = [tokenize(c.text) for c in self.chunks]
        n_docs = max(len(self.chunks), 1)
        df: Counter[str] = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        self._idf = {t: math.log((n_docs + 1) / (count + 1)) + 1.0 for t, count in df.items()}
        self._doc_vecs = [self._vectorize(toks) for toks in self._doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        vec = {t: (c / len(tokens)) * self._idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def retrieve(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        qvec = self._vectorize(tokenize(query))
        if not qvec:
            return []
        scored = []
        for chunk, dvec in zip(self.chunks, self._doc_vecs):
            score = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            scored.append(ScoredChunk(chunk=chunk, score=round(score, 4)))
        scored.sort(key=lambda s: (-s.score, s.chunk.index))
        return scored[:top_k]
