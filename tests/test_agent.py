import pytest
from agent import SupportAgent

agent = SupportAgent()


@pytest.mark.parametrize(
    "query,expected_fragment",
    [
        ("What does EVA do?", "eligibility"),
        ("Tell me about the eligibility verification agent", "eligibility"),
        ("How does CAM work?", "claims"),
        ("What does claims processing do?", "claims"),
        ("How does PHIL work?", "payment"),
        ("payment posting agent", "payment"),
        ("Tell me about Thoughtful AI's agents", "suite"),
        ("What agents does Thoughtful AI have?", "suite"),
        ("What are the benefits?", "reduce"),
        ("What are the benefits of using Thoughtful AI?", "reduce"),
    ],
)
def test_known_questions(query, expected_fragment):
    answer, score = agent.find_best_match(query)
    assert answer is not None, f"No match found for: {query} (score={score:.3f})"
    assert expected_fragment in answer.lower(), f"Expected '{expected_fragment}' in answer for: {query}"


@pytest.mark.parametrize(
    "query",
    [
        "What is the weather today?",
        "How do I make pasta?",
        "What is the capital of France?",
    ],
)
def test_unrelated_questions_return_no_match(query):
    answer, score = agent.find_best_match(query)
    assert answer is None, f"Should not match for: {query} (score={score:.3f})"


def test_empty_input():
    response = agent.respond("", [])
    assert "please" in response.lower()


def test_whitespace_input():
    response = agent.respond("   ", [])
    assert "please" in response.lower()
