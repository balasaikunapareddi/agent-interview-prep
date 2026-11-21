from agent_interview_prep.documents import chunk_text, load_text
from agent_interview_prep.gaps import MATCHED, MISSING, PARTIAL, analyze_gaps, summarize
from agent_interview_prep.retrieval import TfidfRetriever
from agent_interview_prep.skills import extract_requirements


def _gaps():
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    retriever = TfidfRetriever(chunk_text(resume, source="resume"))
    return analyze_gaps(extract_requirements(jd), resume, retriever)


def _by_name(gaps):
    return {g.skill.name: g for g in gaps}


def test_python_is_matched_with_evidence():
    gap = _by_name(_gaps())["python"]
    assert gap.status == MATCHED
    assert gap.evidence is not None
    assert "python" in gap.evidence.chunk.text.lower()


def test_llm_is_missing():
    gap = _by_name(_gaps())["llm"]
    assert gap.status == MISSING
    assert gap.evidence is None
    assert gap.matched_terms == ()


def test_rag_is_partial_via_tfidf_project():
    gap = _by_name(_gaps())["rag"]
    assert gap.status == PARTIAL
    assert "tf-idf" in gap.matched_terms
    assert gap.evidence is not None
    assert "tf-idf" in gap.evidence.chunk.text.lower()


def test_machine_learning_matched_via_churn_model():
    gap = _by_name(_gaps())["machine-learning"]
    assert gap.status == MATCHED
    assert gap.evidence is not None
    assert "scikit-learn" in gap.evidence.chunk.text.lower()


def test_summary_partitions_all_gaps():
    gaps = _gaps()
    summary = summarize(gaps)
    assert sum(len(v) for v in summary.values()) == len(gaps)
    assert summary[MATCHED] and summary[PARTIAL] and summary[MISSING]


def test_every_requirement_gets_a_gap():
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    reqs = extract_requirements(jd)
    gaps = _gaps()
    assert len(gaps) == len(reqs)
