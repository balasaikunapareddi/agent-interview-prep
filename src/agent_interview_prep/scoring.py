"""Deterministic answer scoring rubric.

Four dimensions, each 0–5, total /20:
- relevance:   does the answer talk about the skill that was asked about?
- specificity: numbers, metrics, named technologies, concrete detail.
- structure:   STAR shape — situation, task, action, result markers.
- evidence:    overlap with the resume chunk the question was mapped to
               (is the candidate telling *their* story?).

Every rule is a token/marker check in code, so the same answer always gets
the same score. Live mode can swap in an LLM judge; the rubric dimensions
stay the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .questions import Question
from .retrieval import tokenize
from .skills import BY_NAME

_NUMBER_RE = re.compile(r"\d")
_METRIC_WORDS = ("%", "ms", "seconds", "requests/day", "latency", "reduced", "increased", "improved", "cut", "saved", "shipped", "auc", "accuracy")

_STAR_MARKERS = {
    "situation": ("when ", "we had", "the problem", "at the time", "our team", "the system"),
    "task": ("i needed", "my job", "the goal", "i was responsible", "we needed", "had to"),
    "action": ("i built", "i decided", "i implemented", "i led", "i designed", "i wrote", "i introduced", "i set up", "i trained", "i optimized"),
    "result": ("result", "reduced", "increased", "improved", "shipped", "cut ", "saved", "went from", "ended up", "outcome"),
}

WEAK_THRESHOLD = 12.0  # totals below this land in the study plan


@dataclass(frozen=True)
class AnswerScore:
    relevance: float
    specificity: float
    structure: float
    evidence: float
    feedback: tuple[str, ...]

    @property
    def total(self) -> float:
        return round(self.relevance + self.specificity + self.structure + self.evidence, 1)


def _overlap_ratio(answer_tokens: set[str], target_tokens: set[str]) -> float:
    if not target_tokens:
        return 0.0
    return len(answer_tokens & target_tokens) / len(target_tokens)


def score_answer(question: Question, answer: str) -> AnswerScore:
    lowered = answer.lower()
    answer_tokens = set(tokenize(answer))
    skill = BY_NAME.get(question.skill_name)
    feedback: list[str] = []

    # relevance: skill terms + question content words
    skill_terms: set[str] = set()
    if skill is not None:
        for term in (*skill.aliases, *skill.related, skill.label):
            skill_terms.update(tokenize(term))
    relevance = round(min(5.0, _overlap_ratio(answer_tokens, skill_terms) * 8.0), 1)
    if relevance < 2.5:
        feedback.append("Answer drifts off the skill that was asked about — name the technology and the work directly.")

    # specificity: numbers/metrics + named tech + sane length
    numbers = len(_NUMBER_RE.findall(answer))
    metrics = sum(1 for w in _METRIC_WORDS if w in lowered)
    tech_hits = 0
    if skill is not None:
        tech_hits = sum(1 for t in skill.aliases if t in lowered)
    specificity = round(min(5.0, numbers * 0.9 + metrics * 0.8 + tech_hits * 0.7 + (1.0 if len(answer) > 220 else 0.0)), 1)
    if specificity < 2.5:
        feedback.append("Too vague — add numbers: scale, latency, percentages, timelines.")

    # structure: STAR markers, one point-ish per component present
    components = sum(1 for markers in _STAR_MARKERS.values() if any(m in lowered for m in markers))
    structure = round(min(5.0, components * 1.25), 1)
    if structure < 2.5:
        feedback.append("Missing STAR shape — set the scene, your task, what you did, and the result.")

    # evidence: overlap with the mapped resume chunk
    if question.experience is not None:
        chunk_tokens = set(tokenize(question.experience.chunk.text))
        evidence = round(min(5.0, _overlap_ratio(answer_tokens, chunk_tokens) * 10.0), 1)
    else:
        evidence = 0.0
    if evidence < 2.5:
        feedback.append("Answer doesn't connect to your actual resume experience — tell the real story, with its details.")

    if not feedback:
        feedback.append("Strong answer: on-topic, specific, structured, and grounded in your experience.")
    return AnswerScore(relevance, specificity, structure, evidence, tuple(feedback))
