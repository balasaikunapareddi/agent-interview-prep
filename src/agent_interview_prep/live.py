"""Live mode: LlamaIndex FunctionAgent + Gemini, same tools.

Everything here is lazy-imported and only runs when GEMINI_API_KEY is set
(and the `live` extras are installed: llama-index-llms-gemini,
llama-index-embeddings-gemini). Mock mode never touches this module.

In live mode the retrieval swap is the point: the local TF-IDF index is
replaced by a LlamaIndex VectorStoreIndex over the resume chunks with
Gemini embeddings, and the deterministic loop is replaced by a
FunctionAgent that calls the same typed tools conversationally.

NOT exercised by the test suite (no key in CI) — stated in the README.
"""

from __future__ import annotations

import os

from .documents import chunk_text
from .prompts import SYSTEM_PROMPT
from .tools import PrepContext


def require_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; live mode needs it. Mock mode needs no key.")
    return key


def build_live_retriever(resume_text: str):
    """VectorStoreIndex over resume chunks with Gemini embeddings."""
    require_gemini_key()
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.gemini import GeminiEmbedding

    chunks = chunk_text(resume_text, source="resume")
    nodes = [TextNode(text=c.text, metadata={"source": c.source, "index": c.index}) for c in chunks]
    index = VectorStoreIndex(nodes, embed_model=GeminiEmbedding(model_name="models/text-embedding-004"))
    return index.as_retriever(similarity_top_k=3)


def build_live_agent(resume_text: str, jd_text: str):
    """FunctionAgent with the same tools as mock mode, driven by Gemini."""
    require_gemini_key()
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.gemini import Gemini

    from .tools import (
        AnalyzeGapsInput,
        BuildStudyPlanInput,
        GenerateQuestionsInput,
        RunMockInterviewInput,
        analyze_gaps_tool,
        build_study_plan_tool,
        generate_questions_tool,
        run_mock_interview_tool,
    )

    ctx = PrepContext.from_texts(resume_text, jd_text)
    tools = [
        FunctionTool.from_defaults(
            fn=lambda resume_text, jd_text: analyze_gaps_tool(
                ctx, AnalyzeGapsInput(resume_text=resume_text, jd_text=jd_text)
            ),
            name="analyze_gaps",
            description="Extract JD requirements and match them against resume evidence.",
        ),
        FunctionTool.from_defaults(
            fn=lambda max_questions=None: generate_questions_tool(
                ctx, GenerateQuestionsInput(max_questions=max_questions)
            ),
            name="generate_questions",
            description="Generate likely interview questions mapped to resume experience.",
        ),
        FunctionTool.from_defaults(
            fn=lambda interactive=False: run_mock_interview_tool(
                ctx, RunMockInterviewInput(interactive=interactive)
            ),
            name="run_mock_interview",
            description="Run the scored mock interview over the generated questions.",
        ),
        FunctionTool.from_defaults(
            fn=lambda: build_study_plan_tool(ctx, BuildStudyPlanInput()),
            name="build_study_plan",
            description="Build the prioritized study plan from gaps and weak answers.",
        ),
    ]
    return FunctionAgent(
        tools=tools,
        llm=Gemini(model="models/gemini-2.0-flash"),
        system_prompt=SYSTEM_PROMPT,
    )
