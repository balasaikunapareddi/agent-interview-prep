"""Question generation: likely interview questions mapped to real experience.

Rules (in code):
- matched skill  -> technical or behavioral question, mapped to the resume
  chunk that evidences the skill.
- partial skill  -> question that starts from the adjacent experience and
  asks how the candidate extends it, plus an honest-framing suggestion.
- missing skill  -> "bridge" question: names the gap, asks for a ramp-up
  plan, and suggests honest framing anchored on the nearest experience.

Templates rotate deterministically by requirement order, so output is
stable run to run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gaps import MATCHED, MISSING, PARTIAL, SkillGap
from .retrieval import ScoredChunk

TECHNICAL = "technical"
BEHAVIORAL = "behavioral"
BRIDGE = "bridge"

_TECHNICAL_TEMPLATES = (
    "Walk me through how you've used {label} in production — what did you build, and what broke?",
    "How would you design a system that relies on {label} at scale? What are the failure modes you'd plan for?",
    "Tell me about a hard problem you solved with {label}. How did you debug it?",
)

_BEHAVIORAL_TEMPLATES = (
    "Tell me about a time your {label} work didn't go as planned. What did you do?",
    "Describe a project where {label} was central. What was your specific contribution, and what was the outcome?",
)


@dataclass(frozen=True)
class Question:
    text: str
    kind: str  # technical | behavioral | bridge
    skill_name: str
    skill_label: str
    experience: ScoredChunk | None  # resume chunk this question maps to
    framing: str = ""  # suggested honest framing (bridge questions)


def _snippet(scored: ScoredChunk | None, limit: int = 110) -> str:
    if scored is None:
        return ""
    text = scored.chunk.text
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def generate_questions(gaps: list[SkillGap], max_questions: int | None = None) -> list[Question]:
    questions: list[Question] = []
    for i, gap in enumerate(gaps):
        skill = gap.skill
        if gap.status == MATCHED:
            if i % 2 == 0:
                text = _TECHNICAL_TEMPLATES[i % len(_TECHNICAL_TEMPLATES)].format(label=skill.label)
                kind = TECHNICAL
            else:
                text = _BEHAVIORAL_TEMPLATES[i % len(_BEHAVIORAL_TEMPLATES)].format(label=skill.label)
                kind = BEHAVIORAL
            questions.append(Question(text, kind, skill.name, skill.label, gap.evidence))
        elif gap.status == PARTIAL:
            related = ", ".join(gap.matched_terms)
            text = (
                f"Your resume shows {related} experience. This role needs {skill.label} — "
                f"how would you extend what you've done to cover it?"
            )
            framing = (
                f"Lead with the adjacent work (\"{_snippet(gap.evidence)}\"), be explicit about "
                f"where {skill.label} differs, and name one concrete thing you'd learn first."
            )
            questions.append(Question(text, BRIDGE, skill.name, skill.label, gap.evidence, framing))
        else:  # MISSING
            text = (
                f"This role leans heavily on {skill.label}, which I don't see on your resume. "
                f"What's your plan to ramp up, and what adjacent experience transfers?"
            )
            anchor = _snippet(gap.adjacent)
            framing = (
                f"Don't bluff. Say you haven't shipped {skill.label} yet"
                + (f", bridge from your nearest experience (\"{anchor}\")" if anchor else "")
                + ", then give a specific 30-day ramp-up plan."
            )
            questions.append(Question(text, BRIDGE, skill.name, skill.label, gap.adjacent, framing))
    if max_questions is not None:
        questions = questions[:max_questions]
    return questions
