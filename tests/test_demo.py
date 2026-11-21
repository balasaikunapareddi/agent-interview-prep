from agent_interview_prep.agent import InterviewPrepAgent
from agent_interview_prep.demo import main
from agent_interview_prep.documents import load_text


def test_agent_end_to_end_on_samples():
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    report = InterviewPrepAgent().run(resume, jd, max_questions=8)
    assert report.gaps and report.questions and report.turns and report.plan
    assert len(report.turns) == 8
    by_name = {g.skill.name: g.status for g in report.gaps}
    assert by_name["python"] == "matched"
    assert by_name["llm"] == "missing"
    assert by_name["rag"] == "partial"


def test_demo_main_prints_all_sections(capsys):
    main([])
    out = capsys.readouterr().out
    for section in ("GAP ANALYSIS", "LIKELY QUESTIONS", "MOCK INTERVIEW", "STUDY PLAN"):
        assert section in out
    assert "LLMs" in out  # headline missing skill shows up


def test_demo_accepts_custom_files(tmp_path, capsys):
    resume = tmp_path / "r.txt"
    jd = tmp_path / "j.txt"
    resume.write_text("Python developer. Built REST APIs with FastAPI and PostgreSQL for 3 years.")
    jd.write_text("We need Python, REST APIs, and Kubernetes experience.")
    main(["--resume", str(resume), "--jd", str(jd), "--max-questions", "3"])
    out = capsys.readouterr().out
    assert "GAP ANALYSIS" in out
    assert "Kubernetes" in out
