from agent_interview_prep.documents import chunk_text, load_text
from agent_interview_prep.gaps import analyze_gaps
from agent_interview_prep.questions import BRIDGE, generate_questions
from agent_interview_prep.retrieval import TfidfRetriever
from agent_interview_prep.skills import extract_requirements


def _questions(max_questions=None):
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    retriever = TfidfRetriever(chunk_text(resume, source="resume"))
    gaps = analyze_gaps(extract_requirements(jd), resume, retriever)
    return generate_questions(gaps, max_questions=max_questions), gaps


def test_one_question_per_gap():
    questions, gaps = _questions()
    assert len(questions) == len(gaps)


def test_missing_skills_get_bridge_questions_with_framing():
    questions, gaps = _questions()
    missing_names = {g.skill.name for g in gaps if g.status == "missing"}
    for q in questions:
        if q.skill_name in missing_names:
            assert q.kind == BRIDGE
            assert q.framing  # honest-framing suggestion present
            assert "don't bluff" in q.framing.lower()


def test_matched_questions_map_to_experience():
    questions, gaps = _questions()
    matched_names = {g.skill.name for g in gaps if g.status == "matched"}
    for q in questions:
        if q.skill_name in matched_names:
            assert q.experience is not None
            assert q.experience.chunk.text  # non-empty resume snippet


def test_partial_questions_reference_adjacent_experience():
    questions, gaps = _questions()
    rag_q = next(q for q in questions if q.skill_name == "rag")
    assert rag_q.kind == BRIDGE
    assert "tf-idf" in rag_q.text.lower()
    assert rag_q.experience is not None
    assert "tf-idf" in rag_q.experience.chunk.text.lower()


def test_max_questions_caps_output():
    questions, _ = _questions(max_questions=5)
    assert len(questions) == 5


def test_generation_is_deterministic():
    q1, _ = _questions()
    q2, _ = _questions()
    assert [q.text for q in q1] == [q.text for q in q2]
