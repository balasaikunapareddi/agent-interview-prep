"""Gap analysis: match JD requirements against resume evidence.

Rules (in code, not in a prompt):
- matched: an alias term appears in the resume. Evidence = the retrieved
  chunk that contains an alias (best score wins).
- partial: no alias, but a related term appears. Evidence = the chunk with
  the related term — the adjacent experience to bridge from.
- missing: neither. We still attach the closest retrieved chunk as
  `adjacent` so bridge questions can reference the nearest experience.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import ScoredChunk, TfidfRetriever
from .skills import Requirement, count_mentions

MATCHED = "matched"
PARTIAL = "partial"
MISSING = "missing"


@dataclass(frozen=True)
class SkillGap:
    requirement: Requirement
    status: str  # matched | partial | missing
    evidence: ScoredChunk | None  # chunk supporting matched/partial
    matched_terms: tuple[str, ...]  # alias or related terms found in resume
    adjacent: ScoredChunk | None  # closest chunk (for missing: bridge anchor)

    @property
    def skill(self):
        return self.requirement.skill


def _best_chunk_with_terms(retriever: TfidfRetriever, query: str, terms: list[str]) -> ScoredChunk | None:
    if not terms:
        return None
    lowered_terms = [t.lower() for t in terms]
    for scored in retriever.retrieve(query, top_k=len(retriever.chunks)):
        text = scored.chunk.text.lower()
        if any(t in text for t in lowered_terms):
            return scored
    return None


def analyze_gaps(requirements: list[Requirement], resume_text: str, retriever: TfidfRetriever) -> list[SkillGap]:
    gaps: list[SkillGap] = []
    for req in requirements:
        skill = req.skill
        query = skill.retrieval_query()
        alias_count, alias_found = count_mentions(resume_text, skill.aliases)
        top = retriever.retrieve(query, top_k=1)
        # Only keep the "closest" chunk if it is actually close; a near-zero
        # TF-IDF score means there is no meaningful adjacent experience.
        adjacent = top[0] if top and top[0].score >= 0.08 else None
        if alias_count > 0:
            evidence = _best_chunk_with_terms(retriever, query, alias_found) or adjacent
            gaps.append(SkillGap(req, MATCHED, evidence, tuple(alias_found), adjacent))
            continue
        related_count, related_found = count_mentions(resume_text, skill.related)
        if related_count > 0:
            evidence = _best_chunk_with_terms(retriever, query, related_found) or adjacent
            gaps.append(SkillGap(req, PARTIAL, evidence, tuple(related_found), adjacent))
            continue
        gaps.append(SkillGap(req, MISSING, None, (), adjacent))
    return gaps


def summarize(gaps: list[SkillGap]) -> dict[str, list[SkillGap]]:
    out: dict[str, list[SkillGap]] = {MATCHED: [], PARTIAL: [], MISSING: []}
    for g in gaps:
        out[g.status].append(g)
    return out
