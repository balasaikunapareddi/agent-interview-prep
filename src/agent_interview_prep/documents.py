"""Document loading and chunking.

Chunks are LlamaIndex TextNode objects, but the splitter itself is local:
LlamaIndex's SentenceSplitter pulls NLTK tokenizer data over the network
on first use, which breaks the offline mock-mode contract. So chunking is
a small deterministic sentence packer in pure Python, producing the same
TextNode shape the rest of the pipeline (and live mode) consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llama_index.core.schema import TextNode

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n\s*\n|\n(?=[-•*]\s)")


@dataclass(frozen=True)
class Chunk:
    """One retrievable snippet of a document."""

    text: str
    source: str  # "resume" or "jd"
    index: int  # position within the source document


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [" ".join(p.split()) for p in parts if p and p.strip()]


def chunk_text(text: str, source: str, max_chars: int = 280, overlap_sentences: int = 1) -> list[Chunk]:
    """Pack sentences into overlapping chunks (deterministic, offline)."""
    sentences = _sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            node = TextNode(text=" ".join(current), metadata={"source": source})
            chunks.append(Chunk(text=node.text, source=source, index=len(chunks)))
        current = current[-overlap_sentences:] if overlap_sentences else []
        current_len = sum(len(s) + 1 for s in current)

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            flush()
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        node = TextNode(text=" ".join(current), metadata={"source": source})
        chunks.append(Chunk(text=node.text, source=source, index=len(chunks)))
    return chunks
