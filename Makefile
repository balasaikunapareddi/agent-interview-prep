.PHONY: install test demo

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

demo:
	.venv/bin/python -m agent_interview_prep.demo
