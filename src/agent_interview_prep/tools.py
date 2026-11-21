"""Typed tools for the interview-prep agent.

Each tool has a pydantic input schema and a plain function behind it. The
deterministic agent loop (agent.py) calls these directly; live mode hands
the same tools to a LlamaIndex FunctionAgent. Tool logic never lives in a
prompt — the schemas describe inputs, the functions do the work.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from .documents import chunk_text
from .gaps import SkillGap, analyze_gaps, summarize
from .mock_interview import InterviewTurn, MockCandidate, run_mock_interview
from .questions import Question, generate_questions
from .retrieval import TfidfRetriever
from .skills import extract_requirements
from .study_plan import StudyItem, build_study_plan


class AnalyzeGapsInput(BaseModel):
    resume_text: str = Field(description="Full resume text")
    jd_text: str = Field(description="Full job description text")


class GenerateQuestionsInput(BaseModel):
    max_questions: int | None = Field(default=None, description="Cap on questions returned")


class RunMockInterviewInput(BaseModel):
    interactive: bool = Field(default=False, description="Ask a human via stdin instead of the scripted candidate")


class BuildStudyPlanInput(BaseModel):
    pass


@dataclass
class PrepContext:
    """Shared state the tools operate on (one candidate, one role)."""

    resume_text: str
    jd_text: str
    retriever: TfidfRetriever
    gaps: list[SkillGap] | None = None
    questions: list[Question] | None = None
    turns: list[InterviewTurn] | None = None
    plan: list[StudyItem] | None = None

    @classmethod
    def from_texts(cls, resume_text: str, jd_text: str) -> "PrepContext":
        chunks = chunk_text(resume_text, source="resume")
        return cls(resume_text=resume_text, jd_text=jd_text, retriever=TfidfRetriever(chunks))


def analyze_gaps_tool(ctx: PrepContext, args: AnalyzeGapsInput) -> dict:
    requirements = extract_requirements(args.jd_text)
    ctx.gaps = analyze_gaps(requirements, args.resume_text, ctx.retriever)
    summary = summarize(ctx.gaps)
    return {status: [g.skill.label for g in items] for status, items in summary.items()}


def generate_questions_tool(ctx: PrepContext, args: GenerateQuestionsInput) -> list[Question]:
    if ctx.gaps is None:
        raise ValueError("analyze_gaps must run before generate_questions")
    ctx.questions = generate_questions(ctx.gaps, max_questions=args.max_questions)
    return ctx.questions


def run_mock_interview_tool(ctx: PrepContext, args: RunMockInterviewInput) -> list[InterviewTurn]:
    if ctx.questions is None:
        raise ValueError("generate_questions must run before run_mock_interview")
    if args.interactive:
        def ask(question: Question) -> str:
            print(f"\nQ: {question.text}")
            return input("Your answer: ").strip()
        answer_fn = ask
    else:
        answer_fn = MockCandidate().answer
    ctx.turns = run_mock_interview(ctx.questions, answer_fn)
    return ctx.turns


def build_study_plan_tool(ctx: PrepContext, args: BuildStudyPlanInput) -> list[StudyItem]:
    if ctx.gaps is None or ctx.turns is None:
        raise ValueError("analyze_gaps and run_mock_interview must run before build_study_plan")
    ctx.plan = build_study_plan(ctx.gaps, ctx.turns)
    return ctx.plan
