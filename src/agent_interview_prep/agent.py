"""The interview-prep agent loop.

Mock mode (default): a deterministic loop over the typed tools —
analyze gaps → generate questions → mock interview → study plan. Same
inputs, same outputs, no key, no network. The system prompt (prompts.py)
defines the agent's behavior contract; every decision rule is in code.

Live mode (live.py) hands the same tools to a LlamaIndex FunctionAgent
backed by Gemini, which drives the loop conversationally instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gaps import SkillGap
from .mock_interview import InterviewTurn
from .prompts import SYSTEM_PROMPT
from .questions import Question
from .study_plan import StudyItem
from .tools import (
    AnalyzeGapsInput,
    BuildStudyPlanInput,
    GenerateQuestionsInput,
    PrepContext,
    RunMockInterviewInput,
    analyze_gaps_tool,
    build_study_plan_tool,
    generate_questions_tool,
    run_mock_interview_tool,
)


@dataclass
class PrepReport:
    gaps: list[SkillGap]
    questions: list[Question]
    turns: list[InterviewTurn]
    plan: list[StudyItem]
    system_prompt: str = SYSTEM_PROMPT


class InterviewPrepAgent:
    """Interview-prep agent: resume + job description → prep report.

    This class is the agent's front door. ``run()`` executes the agent
    loop — four typed tools (``tools.py``) over a shared ``PrepContext``:

    1. **analyze_gaps** — extract JD requirements and match them against
       resume evidence (``skills.py``, ``gaps.py``).
    2. **generate_questions** — likely interview questions mapped to the
       candidate's experience (``questions.py``).
    3. **run_mock_interview** — ask each question and score each answer
       (``mock_interview.py``, ``scoring.py``).
    4. **build_study_plan** — prioritized study actions from gaps and
       weak answers (``study_plan.py``).

    The system prompt (``prompts.SYSTEM_PROMPT``) defines the agent's
    behavior contract; every decision rule (skill matching, question
    generation, scoring, plan priority) lives in code. Mock mode
    (default) runs this deterministic loop offline — same inputs, same
    outputs, no key, no network. Live mode (``live.py``) hands the same
    tools and the same system prompt to a LlamaIndex ``FunctionAgent``
    backed by Gemini, which drives the loop conversationally instead.

    Usage::

        agent = InterviewPrepAgent()
        report = agent.run(resume_text, jd_text)
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt
        # The agent's tools, in loop order (see tools.py for schemas).
        self.tools = (
            analyze_gaps_tool,
            generate_questions_tool,
            run_mock_interview_tool,
            build_study_plan_tool,
        )

    def run(
        self,
        resume_text: str,
        jd_text: str,
        *,
        max_questions: int | None = None,
        interactive: bool = False,
    ) -> PrepReport:
        """Run the full prep loop and return the report.

        Steps: analyze gaps → generate questions → mock interview →
        study plan, all over one shared ``PrepContext``.
        """
        ctx = PrepContext.from_texts(resume_text, jd_text)
        analyze_gaps_tool(ctx, AnalyzeGapsInput(resume_text=resume_text, jd_text=jd_text))
        generate_questions_tool(ctx, GenerateQuestionsInput(max_questions=max_questions))
        run_mock_interview_tool(ctx, RunMockInterviewInput(interactive=interactive))
        build_study_plan_tool(ctx, BuildStudyPlanInput())
        assert ctx.gaps is not None and ctx.questions is not None
        assert ctx.turns is not None and ctx.plan is not None
        return PrepReport(
            gaps=ctx.gaps,
            questions=ctx.questions,
            turns=ctx.turns,
            plan=ctx.plan,
            system_prompt=self.system_prompt,
        )
