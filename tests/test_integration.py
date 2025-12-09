"""Integration test for the complete pipeline."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.data_loader import load_kg
from src.sentence_generator import SentenceGenerator
from src.perplexity import PerplexityCalculator
from src.analysis import analyze_results


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_data"
NODES_FILE = FIXTURES_DIR / "nodes.jsonl"
EDGES_FILE = FIXTURES_DIR / "edges.jsonl"


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_full_pipeline(temp_output_dir):
    """Test the complete pipeline from loading to analysis."""
    print("\n[1/5] Loading knowledge graph...")
    kg = load_kg(NODES_FILE, EDGES_FILE, predicate="biolink:treats")
    assert len(kg.get_edges("biolink:treats")) == 3

    print("[2/5] Generating sentences...")
    generator = SentenceGenerator(kg, "biolink:treats", random_seed=42)
    statements = generator.generate_all_statements()
    assert len(statements) >= 12  # At least 3 edges * 4 categories

    print("[3/5] Calculating perplexity...")
    calculator = PerplexityCalculator(
        model_name="gpt2",  # Small model for testing
        device="cpu",
        batch_size=4,
    )
    results = calculator.calculate_for_statements_dict(statements)
    assert len(results) == len(statements)

    # Check that all results have required fields
    for result in results:
        assert "text" in result
        assert "category" in result
        assert "perplexity" in result
        assert result["perplexity"] > 0

    print("[4/5] Analyzing results...")
    analyzer = analyze_results(results, temp_output_dir)

    # Check that output files were created
    assert (temp_output_dir / "results.csv").exists()
    assert (temp_output_dir / "descriptive_stats.json").exists()
    assert (temp_output_dir / "statistical_tests.json").exists()
    assert (temp_output_dir / "boxplot.png").exists()
    assert (temp_output_dir / "violinplot.png").exists()
    assert (temp_output_dir / "histogram.png").exists()

    print("[5/5] Integration test complete!")


def test_pipeline_with_limited_samples(temp_output_dir):
    """Test pipeline with limited number of edges."""
    kg = load_kg(NODES_FILE, EDGES_FILE, predicate="biolink:treats")

    # Limit to 2 edges
    kg.edges_by_predicate["biolink:treats"] = kg.edges_by_predicate["biolink:treats"][:2]

    # Update indices
    kg.subjects_by_predicate["biolink:treats"] = set()
    kg.objects_by_predicate["biolink:treats"] = set()
    kg.valid_pairs_by_predicate["biolink:treats"] = set()

    for edge in kg.edges_by_predicate["biolink:treats"]:
        kg.subjects_by_predicate["biolink:treats"].add(edge.subject)
        kg.objects_by_predicate["biolink:treats"].add(edge.object)
        kg.valid_pairs_by_predicate["biolink:treats"].add((edge.subject, edge.object))

    generator = SentenceGenerator(kg, "biolink:treats", random_seed=42)
    statements = generator.generate_all_statements()

    # Should have 2 true statements (1 per edge) + false/nonsense variants
    true_statements = [s for s in statements if s.category == "true"]
    assert len(true_statements) == 2

    calculator = PerplexityCalculator("gpt2", "cpu", 4)
    results = calculator.calculate_for_statements_dict(statements)

    analyzer = analyze_results(results, temp_output_dir)
    assert (temp_output_dir / "results.csv").exists()
