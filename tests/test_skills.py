from agent_interview_prep.documents import load_text
from agent_interview_prep.skills import count_mentions, extract_requirements

SAMPLES = "samples"


def test_extracts_known_requirements_from_sample_jd():
    jd = load_text(f"{SAMPLES}/jd_ai_ml_engineer.txt")
    names = {r.skill.name for r in extract_requirements(jd)}
    for expected in ("llm", "rag", "python", "pytorch", "kubernetes", "vector-databases", "fine-tuning"):
        assert expected in names


def test_requirements_sorted_by_mention_count():
    jd = load_text(f"{SAMPLES}/jd_ai_ml_engineer.txt")
    reqs = extract_requirements(jd)
    counts = [r.jd_mentions for r in reqs]
    assert counts == sorted(counts, reverse=True)
    assert reqs[0].jd_mentions >= 3  # LLM / vector databases dominate this JD


def test_word_boundaries_avoid_false_positives():
    # "rag" must not match inside "storage"; "sql" must not match "postgresql"
    count, found = count_mentions("We use storage buckets and PostgreSQL.", ("rag", "sql"))
    assert count == 0 and found == []


def test_count_mentions_finds_aliases():
    count, found = count_mentions("We deploy with Docker and Kubernetes on AWS.", ("docker", "kubernetes", "aws"))
    assert count == 3
    assert set(found) == {"docker", "kubernetes", "aws"}
