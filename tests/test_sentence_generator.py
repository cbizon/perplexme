"""Tests for sentence_generator module."""

from pathlib import Path
import pytest
from src.data_loader import load_kg
from src.sentence_generator import SentenceGenerator
from src.config import (
    CATEGORY_TRUE,
    CATEGORY_FALSE_PERMUTED,
    CATEGORY_FALSE_RANDOM,
    CATEGORY_NONSENSE,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_data"
NODES_FILE = FIXTURES_DIR / "nodes.jsonl"
EDGES_FILE = FIXTURES_DIR / "edges.jsonl"


@pytest.fixture
def kg():
    """Load test knowledge graph."""
    return load_kg(NODES_FILE, EDGES_FILE, predicate="biolink:treats")


@pytest.fixture
def generator(kg):
    """Create sentence generator."""
    return SentenceGenerator(kg, "biolink:treats", random_seed=42)


def test_generate_true_statements(generator):
    """Test generating true statements."""
    statements = generator.generate_true_statements()

    assert len(statements) == 3
    assert all(s.category == CATEGORY_TRUE for s in statements)

    # Check one specific statement
    metformin_stmt = [s for s in statements if s.subject_id == "DRUG:001"][0]
    assert metformin_stmt.text == "Metformin treats type 2 diabetes"
    assert metformin_stmt.object_id == "DISEASE:001"


def test_generate_false_permuted_statements(generator):
    """Test generating false permuted statements."""
    statements = generator.generate_false_permuted_statements()

    assert len(statements) > 0
    assert all(s.category == CATEGORY_FALSE_PERMUTED for s in statements)

    # Check that no false statement is actually true
    for stmt in statements:
        assert not generator.kg.is_valid_pair(
            generator.predicate, stmt.subject_id, stmt.object_id
        )


def test_generate_false_random_statements(generator):
    """Test generating false random statements."""
    statements = generator.generate_false_random_statements()

    assert len(statements) > 0
    assert all(s.category == CATEGORY_FALSE_RANDOM for s in statements)

    # Check that subjects have correct category
    for stmt in statements:
        subject_node = generator.kg.get_node(stmt.subject_id)
        assert subject_node.category in generator.subject_categories

        object_node = generator.kg.get_node(stmt.object_id)
        assert object_node.category in generator.object_categories

        # Check that pair is not a valid edge
        assert not generator.kg.is_valid_pair(
            generator.predicate, stmt.subject_id, stmt.object_id
        )


def test_generate_nonsense_statements(generator):
    """Test generating nonsense statements."""
    statements = generator.generate_nonsense_statements()

    assert len(statements) > 0
    assert all(s.category == CATEGORY_NONSENSE for s in statements)

    # Check that types are swapped
    for stmt in statements:
        subject_node = generator.kg.get_node(stmt.subject_id)
        object_node = generator.kg.get_node(stmt.object_id)

        # Subject should have object category, object should have subject category
        assert subject_node.category in generator.object_categories
        assert object_node.category in generator.subject_categories


def test_generate_all_statements(generator):
    """Test generating all statement categories."""
    statements = generator.generate_all_statements()

    # Should have 4 statements per edge (true, false-perm, false-rand, nonsense)
    assert len(statements) >= 12  # 3 edges * 4 categories

    # Count by category
    true_count = sum(1 for s in statements if s.category == CATEGORY_TRUE)
    false_perm_count = sum(1 for s in statements if s.category == CATEGORY_FALSE_PERMUTED)
    false_rand_count = sum(1 for s in statements if s.category == CATEGORY_FALSE_RANDOM)
    nonsense_count = sum(1 for s in statements if s.category == CATEGORY_NONSENSE)

    assert true_count == 3
    assert false_perm_count > 0
    assert false_rand_count > 0
    assert nonsense_count > 0


def test_sentence_format(generator):
    """Test that sentences are properly formatted."""
    statements = generator.generate_true_statements()

    for stmt in statements:
        # Should contain the predicate "treats"
        assert "treats" in stmt.text
        # Should not contain "biolink:"
        assert "biolink:" not in stmt.text
        # Should not contain underscores
        assert "_" not in stmt.text


def test_reproducibility(kg):
    """Test that same seed produces same results."""
    gen1 = SentenceGenerator(kg, "biolink:treats", random_seed=42)
    gen2 = SentenceGenerator(kg, "biolink:treats", random_seed=42)

    stmts1 = gen1.generate_false_permuted_statements()
    stmts2 = gen2.generate_false_permuted_statements()

    assert len(stmts1) == len(stmts2)
    for s1, s2 in zip(stmts1, stmts2):
        assert s1.text == s2.text
        assert s1.subject_id == s2.subject_id
        assert s1.object_id == s2.object_id
