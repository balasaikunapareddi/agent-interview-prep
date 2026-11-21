"""Study plan assembly: gaps + weak answers -> prioritized actions.

Priority rules (in code):
1. Missing skills, ordered by how often the JD mentions them.
2. Partial skills, same ordering.
3. Weak mock-interview answers (total < WEAK_THRESHOLD) on skills not
   already in the plan — the knowledge may exist, the story doesn't.
Each item carries concrete actions and a practice question.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gaps import MATCHED, MISSING, PARTIAL, SkillGap
from .mock_interview import InterviewTurn
from .scoring import WEAK_THRESHOLD

# Concrete actions per skill; generic fallback below.
_ACTION_BANK: dict[str, tuple[str, ...]] = {
    "llm": (
        "Build a small LLM extraction prototype (structured output from messy text) and write up what broke.",
        "Study prompt design patterns: few-shot, chain-of-thought, output schemas, guardrails.",
        "Learn how LLM features are evaluated: golden sets, LLM-as-judge, regression checks.",
    ),
    "rag": (
        "Build a minimal RAG pipeline over a sample corpus: chunk, embed, retrieve, answer with citations.",
        "Learn chunking strategies and retrieval metrics (recall@k, MRR) — then measure your TF-IDF baseline against embeddings.",
        "Study failure modes: stale index, bad chunks, hallucinated citations, and how evals catch them.",
    ),
    "embeddings": (
        "Compare TF-IDF vs embedding retrieval on the same corpus and measure the difference.",
        "Learn embedding model trade-offs: dimension, latency, multilingual support, fine-tuned vs off-the-shelf.",
    ),
    "vector-databases": (
        "Stand up pgvector locally; index 10k documents and benchmark query latency vs brute force.",
        "Learn ANN index types (HNSW, IVF) and when each wins.",
    ),
    "fine-tuning": (
        "Fine-tune a small open model with LoRA on a toy extraction task; compare against prompting-only.",
        "Learn dataset curation for fine-tuning: format, dedup, contamination, held-out eval.",
    ),
    "pytorch": (
        "Rebuild your scikit-learn churn model as a PyTorch MLP and compare AUC and training time.",
        "Work through tensors, autograd, and a full training loop with validation.",
    ),
    "deep-learning": (
        "Study transformer architecture end to end: attention, positional encoding, training objective.",
        "Train one small neural net from scratch and instrument loss/accuracy curves.",
    ),
    "prompt-engineering": (
        "Rewrite three production-style prompts with schemas and few-shot examples; diff the outputs.",
        "Learn prompt versioning and regression testing for prompt changes.",
    ),
    "kubernetes": (
        "Deploy one of your Docker services to a local kind/minikube cluster with a Helm chart.",
        "Learn autoscaling, readiness/liveness probes, and ingress — map each to your ECS experience.",
    ),
    "spark": (
        "Rewrite one nightly ETL job in PySpark and compare runtime and code shape.",
        "Learn partitioning, shuffles, and skew — the three things that break Spark jobs.",
    ),
    "airflow": (
        "Model your nightly ETL as an Airflow DAG with retries, backfills, and SLAs.",
        "Learn idempotent task design — the difference between a rerun and a duplicate.",
    ),
    "mlops": (
        "Add a model registry + drift check to your churn-model serving setup.",
        "Design a rollback path: versioned artifacts, shadow deployment, canary traffic.",
    ),
    "ab-testing": (
        "Design an A/B test for a model change: primary metric, guardrails, sample size, stopping rule.",
        "Learn common traps: peeking, novelty effects, underpowered tests.",
    ),
    "statistics": (
        "Re-derive the stats behind your churn model's AUC and confidence intervals.",
        "Study hypothesis testing basics: p-values, power, and multiple comparisons.",
    ),
}

_GENERIC_ACTIONS = (
    "Build one small end-to-end project that uses this skill and write up the trade-offs.",
    "Prepare two STAR stories from your experience that touch this area, with numbers.",
    "Study the failure modes interviewers probe for this skill.",
)

_WEAK_ANSWER_ACTIONS = (
    "Rewrite your answer in STAR form with at least two concrete numbers.",
    "Rehearse the story out loud twice; time it to under two minutes.",
    "Add the outcome: what changed because of your work?",
)


@dataclass(frozen=True)
class StudyItem:
    rank: int
    skill_name: str
    skill_label: str
    why: str
    actions: tuple[str, ...]
    practice_question: str


def build_study_plan(gaps: list[SkillGap], turns: list[InterviewTurn]) -> list[StudyItem]:
    items: list[StudyItem] = []
    seen: set[str] = set()

    def add(gap: SkillGap, why: str) -> None:
        skill = gap.skill
        seen.add(skill.name)
        actions = _ACTION_BANK.get(skill.name, _GENERIC_ACTIONS)
        practice = next(
            (t.question.text for t in turns if t.question.skill_name == skill.name),
            f"Explain how you would approach {skill.label} in this role, with an example.",
        )
        items.append(
            StudyItem(
                rank=len(items) + 1,
                skill_name=skill.name,
                skill_label=skill.label,
                why=why,
                actions=actions,
                practice_question=practice,
            )
        )

    for status, label in ((MISSING, "missing from resume"), (PARTIAL, "only adjacent experience on resume")):
        ordered = sorted(
            (g for g in gaps if g.status == status),
            key=lambda g: (-g.requirement.jd_mentions, g.skill.name),
        )
        for gap in ordered:
            add(gap, f"{label}; JD mentions it {gap.requirement.jd_mentions}×")

    weak = sorted(
        (t for t in turns if t.score.total < WEAK_THRESHOLD and t.question.skill_name not in seen),
        key=lambda t: t.score.total,
    )
    gap_by_skill = {g.skill.name: g for g in gaps}
    for turn in weak:
        gap = gap_by_skill.get(turn.question.skill_name)
        if gap is None or gap.status != MATCHED:
            continue
        seen.add(gap.skill.name)
        items.append(
            StudyItem(
                rank=len(items) + 1,
                skill_name=gap.skill.name,
                skill_label=gap.skill.label,
                why=f"skill is on the resume but the mock answer scored {turn.score.total}/20",
                actions=_WEAK_ANSWER_ACTIONS,
                practice_question=turn.question.text,
            )
        )
    return items
