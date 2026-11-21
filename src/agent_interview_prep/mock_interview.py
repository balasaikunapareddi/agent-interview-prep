"""Mock interview: ask each generated question, score each answer.

Demo mode uses a deterministic MockCandidate with scripted answers keyed by
skill — a mix of strong answers (specific, STAR-shaped, grounded in the
sample resume) and weak ones (vague, no metrics), so the study plan has
real signal to work with. Interactive CLI mode swaps in input().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .questions import Question
from .scoring import AnswerScore, score_answer

# Scripted answers for the bundled sample candidate. Keyed by skill name.
_SCRIPTED: dict[str, str] = {
    "python": (
        "When I joined Ledgerly, the payments API was a Flask monolith that timed out under load. "
        "I needed to rebuild the hot paths, so I designed FastAPI services in Python with async I/O "
        "and connection pooling. I built the payment-processing APIs serving 2M requests/day at p99 "
        "latency under 180ms, and the result was a 43% cut in p95 read latency after adding Redis caching."
    ),
    "machine-learning": (
        "At Cartway we had a retention problem and no signal on which customers were about to churn. "
        "I was responsible for a side project with the data team, so I trained a scikit-learn "
        "gradient-boosting model on customer activity data and validated it on a holdout set (AUC 0.81). "
        "I served predictions behind a FastAPI endpoint, and the retention team used it to target "
        "save offers — that was the outcome we shipped."
    ),
    "rag": (
        "Honestly, I haven't shipped a RAG system. The closest I've come is a weekend TF-IDF search "
        "tool over my own notes — keyword retrieval, no embeddings. In my first month I'd build a "
        "small RAG prototype over a sample corpus, learn chunking and evaluation, and pair with "
        "whoever owns the current pipeline."
    ),
    "llm": (
        "I haven't worked with LLMs in production yet. My ML experience is classical — a churn model "
        "with scikit-learn. I'd ramp up by building a small extraction prototype, studying prompt "
        "design and evaluation, and shadowing the team that owns the current LLM features."
    ),
    "kubernetes": (
        "I've deployed with Docker on AWS ECS, not Kubernetes directly. When we migrated the billing "
        "monolith into 5 microservices, I containerized everything and set up CI/CD, which cut deploy "
        "time from 40 minutes to 6. The concepts transfer — I'd learn the cluster-specific parts "
        "(Helm, autoscaling, ingress) in the first few weeks."
    ),
    "aws": "I've used AWS quite a bit — ECS, RDS, S3. It works fine for us, no big issues.",
    "sql": "I write SQL pretty often. Mostly joins and some optimization when queries get slow.",
}

_GENERIC_STRONG = (
    "At Ledgerly we had a reliability problem in the payments path, and I was responsible for the "
    "fix. I designed the change, implemented it in Python, and rolled it out behind dashboards and "
    "alerts. The result was measurable: latency down 43% and mean time to detection from 25 minutes "
    "to 4 minutes."
)

_GENERIC_WEAK = "I've done some of that. It went okay overall, I think. We didn't really measure it."


class MockCandidate:
    """Deterministic scripted candidate for demo mode."""

    def __init__(self, scripted: dict[str, str] | None = None) -> None:
        self.scripted = dict(_SCRIPTED if scripted is None else scripted)

    def answer(self, question: Question) -> str:
        if question.skill_name in self.scripted:
            return self.scripted[question.skill_name]
        # Deterministic fallback: bridge questions get the honest weak-ish
        # answer; matched skills get the strong generic story.
        if question.kind == "bridge":
            return _GENERIC_WEAK
        return _GENERIC_STRONG


@dataclass(frozen=True)
class InterviewTurn:
    question: Question
    answer: str
    score: AnswerScore


def run_mock_interview(
    questions: list[Question],
    answer_fn: Callable[[Question], str],
) -> list[InterviewTurn]:
    turns = []
    for q in questions:
        answer = answer_fn(q)
        turns.append(InterviewTurn(question=q, answer=answer, score=score_answer(q, answer)))
    return turns
