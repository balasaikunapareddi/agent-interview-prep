"""Skill lexicon and JD skill extraction.

Extraction is deliberately boring: a curated lexicon of skills, each with
aliases (terms that count as direct evidence) and related terms (adjacent
experience that counts as partial). The JD is scanned for alias mentions;
mention count becomes the requirement's importance weight. No prompt
decides what a skill is — the lexicon is the rulebook, and it is testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str  # canonical id, e.g. "llm"
    label: str  # display name, e.g. "LLMs"
    aliases: tuple[str, ...]  # direct evidence terms
    related: tuple[str, ...] = ()  # adjacent experience -> partial
    query: str = ""  # retrieval query for evidence; defaults to label

    def retrieval_query(self) -> str:
        return self.query or self.label


def _s(name, label, aliases, related=(), query=""):
    return Skill(name=name, label=label, aliases=tuple(aliases), related=tuple(related), query=query)


LEXICON: tuple[Skill, ...] = (
    _s("python", "Python", ["python"]),
    _s("sql", "SQL", ["sql"], ["postgresql", "postgres", "mysql"]),
    _s("postgresql", "PostgreSQL", ["postgresql", "postgres"], ["sql", "database"]),
    _s("rest-apis", "REST APIs", ["rest api", "rest apis", "restful"], ["api", "fastapi", "flask", "django"]),
    _s("fastapi", "FastAPI", ["fastapi"], ["flask", "django", "rest api"]),
    _s("docker", "Docker", ["docker", "containerized", "containers"], ["kubernetes"]),
    _s("kubernetes", "Kubernetes", ["kubernetes", "k8s"], ["docker", "ecs", "containers"]),
    _s("aws", "AWS", ["aws", "amazon web services", "ecs", "s3", "rds"], ["cloud", "gcp", "azure"]),
    _s("ci-cd", "CI/CD", ["ci/cd", "continuous integration", "continuous delivery", "github actions", "jenkins"], ["deployment", "pipelines"]),
    _s("microservices", "Microservices", ["microservices", "microservice"], ["distributed systems", "service boundaries"]),
    _s("kafka", "Kafka", ["kafka"], ["streaming", "event pipeline"]),
    _s("redis", "Redis", ["redis"], ["caching", "cache"]),
    _s("testing", "Testing", ["pytest", "unit test", "unit tests", "test coverage", "contract testing"], ["ci/cd"]),
    _s("system-design", "System design", ["system design", "distributed systems"], ["microservices", "scalability", "latency"]),
    _s(
        "machine-learning",
        "Machine learning",
        ["machine learning", "ml model", "gradient boosting", "scikit-learn", "xgboost", "random forest", "trained a model", "model training", "churn-prediction", "churn prediction"],
        ["data analysis", "statistics", "predictions"],
        query="machine learning model training scikit-learn predictions",
    ),
    _s("deep-learning", "Deep learning", ["deep learning", "neural network", "neural networks"], ["machine learning", "pytorch", "tensorflow"]),
    _s("pytorch", "PyTorch", ["pytorch"], ["tensorflow", "deep learning", "machine learning"]),
    _s("tensorflow", "TensorFlow", ["tensorflow"], ["pytorch", "deep learning"]),
    _s("llm", "LLMs", ["llm", "llms", "large language model", "large language models", "gpt", "transformer", "transformers"], ["nlp", "language model"]),
    _s("rag", "RAG", ["rag", "retrieval-augmented", "retrieval augmented"], ["search", "retrieval", "tf-idf", "embeddings"], query="retrieval augmented generation search documents"),
    _s("prompt-engineering", "Prompt engineering", ["prompt engineering", "prompting", "prompts"], ["llm"]),
    _s("embeddings", "Embeddings", ["embeddings", "embedding"], ["tf-idf", "vector", "semantic search", "nlp"]),
    _s("vector-databases", "Vector databases", ["vector database", "vector databases", "pgvector", "pinecone", "faiss", "weaviate", "qdrant"], ["embeddings", "semantic search"]),
    _s("fine-tuning", "Fine-tuning", ["fine-tuning", "fine tuning", "fine-tuned", "fine tune", "lora", "qlora"], ["pytorch", "deep learning"]),
    _s(
        "model-evaluation",
        "Model evaluation",
        ["model evaluation", "evaluating models", "offline evaluation", "holdout", "auc", "precision", "recall", "validation set"],
        ["a/b testing", "metrics", "testing"],
        query="model evaluation holdout auc validation metrics",
    ),
    _s("ab-testing", "A/B testing", ["a/b testing", "a/b test", "ab testing", "experimentation"], ["statistics", "metrics", "model evaluation"]),
    _s("statistics", "Statistics", ["statistics", "statistical"], ["a/b testing", "data analysis", "auc"]),
    _s("data-pipelines", "Data pipelines", ["data pipelines", "data pipeline", "etl", "elt"], ["kafka", "spark", "airflow", "batch jobs"]),
    _s("spark", "Spark", ["spark", "pyspark"], ["data pipelines", "etl", "big data"]),
    _s("airflow", "Airflow", ["airflow"], ["data pipelines", "etl", "scheduling"]),
    _s(
        "mlops",
        "MLOps",
        ["mlops", "model registry", "model monitoring", "drift", "model deployment", "model serving"],
        ["ci/cd", "deployment", "served predictions"],
        query="model deployment serving monitoring ci/cd pipelines",
    ),
    _s("nlp", "NLP", ["nlp", "natural language processing", "text classification", "sentiment analysis"], ["llm", "embeddings"]),
)

BY_NAME = {s.name: s for s in LEXICON}


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(term.lower()) + r"\b")


def count_mentions(text: str, terms: tuple[str, ...]) -> tuple[int, list[str]]:
    """Count word-boundary mentions of any term; return (total, terms found)."""
    lowered = text.lower()
    total = 0
    found: list[str] = []
    for term in terms:
        hits = len(_term_pattern(term).findall(lowered))
        if hits:
            total += hits
            found.append(term)
    return total, found


@dataclass(frozen=True)
class Requirement:
    skill: Skill
    jd_mentions: int  # importance weight: how often the JD asks for it


def extract_requirements(jd_text: str) -> list[Requirement]:
    """Every lexicon skill the JD mentions, sorted by mention count desc."""
    reqs = []
    for skill in LEXICON:
        count, _ = count_mentions(jd_text, skill.aliases)
        if count > 0:
            reqs.append(Requirement(skill=skill, jd_mentions=count))
    reqs.sort(key=lambda r: (-r.jd_mentions, r.skill.name))
    return reqs
