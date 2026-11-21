"""One-command demo: bundled backend-engineer resume vs AI/ML-engineer JD.

    python -m agent_interview_prep.demo
    python -m agent_interview_prep.demo --resume my_resume.txt --jd job.txt
    python -m agent_interview_prep.demo --interactive   # you answer the questions

Prints the gap analysis, the generated questions with their mapped resume
experience, a scored mock interview, and the prioritized study plan.
Mock mode: no API key, no network.
"""

from __future__ import annotations

import argparse
import os

from .agent import InterviewPrepAgent
from .documents import load_text
from .gaps import MATCHED, MISSING, PARTIAL, summarize

_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "samples")


def _sample_path(name: str) -> str:
    # Works from a source checkout; falls back to cwd-relative samples/.
    candidate = os.path.normpath(os.path.join(_SAMPLES_DIR, name))
    if os.path.exists(candidate):
        return candidate
    return os.path.join("samples", name)


def _short(text: str, limit: int = 130) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _excerpt(text: str, terms: tuple[str, ...], limit: int = 150) -> str:
    """Show the part of the chunk around the first matched term."""
    lowered = text.lower()
    pos = -1
    for term in terms:
        pos = lowered.find(term.lower())
        if pos >= 0:
            break
    if pos < 0:
        return _short(text, limit)
    start = max(0, pos - 40)
    end = min(len(text), start + limit)
    snippet = text[start:end]
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interview prep: gaps, questions, mock interview, study plan.")
    parser.add_argument("--resume", default=_sample_path("resume_backend_engineer.txt"))
    parser.add_argument("--jd", default=_sample_path("jd_ai_ml_engineer.txt"))
    parser.add_argument("--interactive", action="store_true", help="answer the mock-interview questions yourself")
    parser.add_argument("--max-questions", type=int, default=12, help="questions to generate/ask (default 12; 0 = all)")
    args = parser.parse_args(argv)

    resume_text = load_text(args.resume)
    jd_text = load_text(args.jd)

    agent = InterviewPrepAgent()
    report = agent.run(
        resume_text,
        jd_text,
        max_questions=args.max_questions or None,
        interactive=args.interactive,
    )

    print("=" * 72)
    print("GAP ANALYSIS — JD requirements vs resume evidence")
    print("=" * 72)
    summary = summarize(report.gaps)
    for status, title in ((MATCHED, "MATCHED"), (PARTIAL, "PARTIAL (adjacent experience)"), (MISSING, "MISSING")):
        print(f"\n{title} ({len(summary[status])}):")
        for gap in summary[status]:
            line = f"  - {gap.skill.label} (JD mentions: {gap.requirement.jd_mentions})"
            if gap.matched_terms:
                line += f"  [{', '.join(gap.matched_terms)}]"
            print(line)
            if gap.evidence is not None:
                print(f"      evidence: \"{_excerpt(gap.evidence.chunk.text, gap.matched_terms)}\"")

    print("\n" + "=" * 72)
    print("LIKELY QUESTIONS — mapped to your experience")
    print("=" * 72)
    for i, q in enumerate(report.questions, 1):
        print(f"\n{i}. [{q.kind}] {q.text}")
        if q.experience is not None:
            print(f"   your story: \"{_short(q.experience.chunk.text)}\"")
        if q.framing:
            print(f"   framing: {q.framing}")

    print("\n" + "=" * 72)
    print("MOCK INTERVIEW — scored answers (relevance/specificity/structure/evidence, /20)")
    print("=" * 72)
    for i, turn in enumerate(report.turns, 1):
        s = turn.score
        print(f"\n{i}. {turn.question.skill_label}: {s.total}/20  "
              f"(rel {s.relevance}, spec {s.specificity}, struct {s.structure}, evid {s.evidence})")
        print(f"   A: {_short(turn.answer, 180)}")
        for fb in s.feedback:
            print(f"   -> {fb}")

    print("\n" + "=" * 72)
    print("STUDY PLAN — in priority order")
    print("=" * 72)
    for item in report.plan:
        print(f"\n{item.rank}. {item.skill_label} — {item.why}")
        for action in item.actions:
            print(f"   - {action}")
        print(f"   practice: {item.practice_question}")


if __name__ == "__main__":
    main()
