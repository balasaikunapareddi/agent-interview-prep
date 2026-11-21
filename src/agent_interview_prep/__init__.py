"""agent-interview-prep: resume + JD -> gaps, questions, mock interview, study plan."""

from .agent import InterviewPrepAgent, PrepReport
from .prompts import SYSTEM_PROMPT

__all__ = ["InterviewPrepAgent", "PrepReport", "SYSTEM_PROMPT"]
__version__ = "0.1.0"
