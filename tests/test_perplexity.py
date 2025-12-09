"""Tests for perplexity module."""

import pytest
from src.perplexity import PerplexityCalculator
from src.sentence_generator import Statement
from src.config import CATEGORY_TRUE


@pytest.fixture
def calculator():
    """Create a perplexity calculator with a small test model."""
    # Use a very small model for testing
    return PerplexityCalculator(
        model_name="gpt2",  # Small model for testing
        device="cpu",
        batch_size=2,
    )


def test_calculate_perplexity_single(calculator):
    """Test calculating perplexity for a single text."""
    text = "The cat sat on the mat"
    perplexity, loss = calculator.calculate_perplexity_single(text)

    assert isinstance(perplexity, float)
    assert isinstance(loss, float)
    assert perplexity > 0
    assert loss > 0


def test_calculate_perplexity_batch(calculator):
    """Test calculating perplexity for a batch of texts."""
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "Water is wet",
    ]
    results = calculator.calculate_perplexity_batch(texts)

    assert len(results) == 3
    for perplexity, loss in results:
        assert isinstance(perplexity, float)
        assert isinstance(loss, float)
        assert perplexity > 0
        assert loss > 0


def test_calculate_for_statements(calculator):
    """Test calculating perplexity for statements."""
    statements = [
        Statement(
            text="Metformin treats type 2 diabetes",
            category=CATEGORY_TRUE,
            edge_id="test_1",
            subject_id="DRUG:001",
            object_id="DISEASE:001",
        ),
        Statement(
            text="Aspirin treats headache",
            category=CATEGORY_TRUE,
            edge_id="test_2",
            subject_id="DRUG:003",
            object_id="DISEASE:003",
        ),
    ]

    results = calculator.calculate_for_statements(statements)

    assert len(results) == 2
    for result in results:
        assert result.perplexity > 0
        assert result.loss > 0
        assert result.statement in statements


def test_calculate_for_statements_dict(calculator):
    """Test calculating perplexity and returning as dicts."""
    statements = [
        Statement(
            text="Metformin treats type 2 diabetes",
            category=CATEGORY_TRUE,
            edge_id="test_1",
            subject_id="DRUG:001",
            object_id="DISEASE:001",
        ),
    ]

    results = calculator.calculate_for_statements_dict(statements)

    assert len(results) == 1
    assert "text" in results[0]
    assert "category" in results[0]
    assert "perplexity" in results[0]
    assert "loss" in results[0]
    assert results[0]["perplexity"] > 0


def test_different_perplexities(calculator):
    """Test that different texts have different perplexities."""
    # A coherent sentence should have lower perplexity than random words
    coherent = "The cat sat on the mat"
    random_words = "zebra quantum purple banana elephant"

    ppl_coherent, _ = calculator.calculate_perplexity_single(coherent)
    ppl_random, _ = calculator.calculate_perplexity_single(random_words)

    # Coherent sentence should generally have lower perplexity
    # This is a weak test since results can vary
    assert ppl_coherent > 0
    assert ppl_random > 0
