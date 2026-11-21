from agent_interview_prep.agent import InterviewPrepAgent
from agent_interview_prep.documents import load_text
from agent_interview_prep.gaps import MISSING, PARTIAL


def _report():
    resume = load_text("samples/resume_backend_engineer.txt")
    jd = load_text("samples/jd_ai_ml_engineer.txt")
    return InterviewPrepAgent().run(resume, jd)


def test_missing_skills_come_before_partial():
    report = _report()
    status_by_skill = {g.skill.name: g.status for g in report.gaps}
    statuses = [status_by_skill[item.skill_name] for item in report.plan]
    first_partial = statuses.index(PARTIAL) if PARTIAL in statuses else len(statuses)
    assert all(s == MISSING for s in statuses[:first_partial])


def test_plan_ranks_are_sequential():
    report = _report()
    assert [item.rank for item in report.plan] == list(range(1, len(report.plan) + 1))


def test_every_item_has_actions_and_practice_question():
    report = _report()
    assert report.plan
    for item in report.plan:
        assert item.actions
        assert item.practice_question
        assert item.why


def test_top_missing_skill_is_llm_or_vector_databases():
    report = _report()
    assert report.plan[0].skill_name in ("llm", "vector-databases")


def test_weak_matched_answers_enter_plan():
    report = _report()
    # The scripted candidate gives weak answers for sql/aws; those matched
    # skills should surface in the plan as story-rehearsal items.
    plan_skills = {item.skill_name for item in report.plan}
    assert "sql" in plan_skills or "aws" in plan_skills
    weak_items = [i for i in report.plan if "mock answer scored" in i.why]
    assert weak_items
