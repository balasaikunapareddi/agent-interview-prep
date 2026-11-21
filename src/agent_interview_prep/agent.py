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
    """Deterministic interview-prep agent (mock mode)."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def run(
        self,
        resume_text: str,
        jd_text: str,
        *,
        max_questions: int | None = None,
        interactive: bool = False,
    ) -> PrepReport:
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
