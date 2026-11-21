from agent_interview_prep.documents import chunk_text
from agent_interview_prep.retrieval import TfidfRetriever

TEXT = (
    "Built payment APIs in Python with FastAPI serving two million requests per day. "
    "Led the migration of forty services to Kubernetes on AWS. "
    "Trained a churn prediction model with scikit-learn and gradient boosting. "
    "Designed PostgreSQL schemas and optimized slow queries with indexes."
)


def _retriever():
    return TfidfRetriever(chunk_text(TEXT, source="resume"))


def test_retrieves_relevant_chunk_first():
    r = _retriever()
    top = r.retrieve("python fastapi api", top_k=1)
    assert top and "FastAPI" in top[0].chunk.text


def test_retrieval_is_deterministic():
    r = _retriever()
    first = [(s.chunk.index, s.score) for s in r.retrieve("kubernetes aws", top_k=3)]
    second = [(s.chunk.index, s.score) for s in r.retrieve("kubernetes aws", top_k=3)]
    assert first == second


def test_scores_are_bounded():
    r = _retriever()
    for scored in r.retrieve("model training", top_k=4):
        assert 0.0 <= scored.score <= 1.0


def test_empty_query_returns_nothing():
    r = _retriever()
    assert r.retrieve("", top_k=3) == []
    assert r.retrieve("the and of", top_k=3) == []  # stopwords only


def test_chunking_preserves_source_and_order():
    chunks = chunk_text(TEXT, source="resume")
    assert chunks
    assert all(c.source == "resume" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
