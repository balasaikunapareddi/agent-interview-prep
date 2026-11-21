from agent_interview_prep.documents import chunk_text, load_text
from agent_interview_prep.gaps import analyze_gaps
from agent_interview_prep.mock_interview import MockCandidate, run_mock_interview
from agent_interview_prep.questions import generate_questions
from agent_interview_prep.retrieval import TfidfRetriever
from agent_interview_prep.scoring import score_answer
from agent_interview_prep.skills import extract_requirements

STRONG = (
    "When I joined Ledgerly, the payments API timed out under load. I needed to rebuild the hot "
    "paths, so I designed FastAPI services in Python with async I/O. I built payment APIs serving "
    "2M requests/day at p99 latency under 180ms, and the result was a 43% reduction in p95 latency."
)
WEAK = "I've done some of that. It went okay overall, I think. We didn't really measure it."


def _python_question():
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    retriever = TfidfRetriever(chunk_text(resume, source="resume"))
    gaps = analyze_gaps(extract_requirements(jd), resume, retriever)
    return next(q for q in generate_questions(gaps) if q.skill_name == "python")


def test_strong_answer_beats_weak_answer():
    q = _python_question()
    assert score_answer(q, STRONG).total > score_answer(q, WEAK).total


def test_dimensions_bounded_and_total_capped():
    q = _python_question()
    s = score_answer(q, STRONG)
    for dim in (s.relevance, s.specificity, s.structure, s.evidence):
        assert 0.0 <= dim <= 5.0
    assert s.total <= 20.0


def test_empty_answer_scores_zero():
    q = _python_question()
    s = score_answer(q, "")
    assert s.total == 0.0
    assert s.feedback  # still explains what to fix


def test_scoring_is_deterministic():
    q = _python_question()
    assert score_answer(q, STRONG) == score_answer(q, STRONG)


def test_mock_interview_scores_every_question():
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    retriever = TfidfRetriever(chunk_text(resume, source="resume"))
    gaps = analyze_gaps(extract_requirements(jd), resume, retriever)
    questions = generate_questions(gaps, max_questions=6)
    turns = run_mock_interview(questions, MockCandidate().answer)
    assert len(turns) == 6
    assert all(t.answer for t in turns)
    assert all(0.0 <= t.score.total <= 20.0 for t in turns)
