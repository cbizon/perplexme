"""Tests for analysis module."""

import pytest
from pathlib import Path
import tempfile
import shutil
import numpy as np

from src.analysis import PerplexityAnalyzer
from src.config import CATEGORY_TRUE, CATEGORY_FALSE_PERMUTED


@pytest.fixture
def sample_results():
    """Create sample results for testing."""
    np.random.seed(42)

    results = []

    # True statements - lower perplexity
    for i in range(20):
        results.append({
            'text': f"Statement {i}",
            'category': CATEGORY_TRUE,
            'edge_id': f"edge_{i}",
            'subject_id': f"subj_{i}",
            'object_id': f"obj_{i}",
            'perplexity': np.random.normal(10, 2),
            'loss': np.random.normal(2, 0.5),
        })

    # False statements - higher perplexity
    for i in range(20):
        results.append({
            'text': f"Statement false {i}",
            'category': CATEGORY_FALSE_PERMUTED,
            'edge_id': f"edge_false_{i}",
            'subject_id': f"subj_{i}",
            'object_id': f"obj_{i}",
            'perplexity': np.random.normal(15, 2),
            'loss': np.random.normal(3, 0.5),
        })

    return results


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_analyzer_initialization(sample_results, temp_output_dir):
    """Test analyzer initialization."""
    analyzer = PerplexityAnalyzer(sample_results, temp_output_dir)

    assert len(analyzer.df) == 40
    assert 'perplexity' in analyzer.df.columns
    assert 'category' in analyzer.df.columns


def test_compute_descriptive_stats(sample_results, temp_output_dir):
    """Test computing descriptive statistics."""
    analyzer = PerplexityAnalyzer(sample_results, temp_output_dir)
    stats = analyzer.compute_descriptive_stats()

    assert CATEGORY_TRUE in stats
    assert CATEGORY_FALSE_PERMUTED in stats

    true_stats = stats[CATEGORY_TRUE]
    assert 'mean' in true_stats
    assert 'median' in true_stats
    assert 'std' in true_stats
    assert 'min' in true_stats
    assert 'max' in true_stats
    assert true_stats['count'] == 20


def test_compute_statistical_tests(sample_results, temp_output_dir):
    """Test computing statistical tests."""
    analyzer = PerplexityAnalyzer(sample_results, temp_output_dir)
    tests = analyzer.compute_statistical_tests()

    assert 'anova' in tests
    assert 'kruskal_wallis' in tests
    assert 'pairwise' in tests

    # Check ANOVA results
    assert 'f_statistic' in tests['anova']
    assert 'p_value' in tests['anova']

    # Check pairwise tests
    pairwise = tests['pairwise']
    assert len(pairwise) > 0
    for comparison, results in pairwise.items():
        assert 't_test' in results
        assert 'cohens_d' in results


def test_create_plots(sample_results, temp_output_dir):
    """Test creating plots."""
    analyzer = PerplexityAnalyzer(sample_results, temp_output_dir)

    # Create box plot
    boxplot_path = analyzer.create_box_plot()
    assert boxplot_path.exists()
    assert boxplot_path.suffix == '.png'

    # Create violin plot
    violin_path = analyzer.create_violin_plot()
    assert violin_path.exists()
    assert violin_path.suffix == '.png'

    # Create histogram
    hist_path = analyzer.create_histogram()
    assert hist_path.exists()
    assert hist_path.suffix == '.png'


def test_save_results(sample_results, temp_output_dir):
    """Test saving all results."""
    analyzer = PerplexityAnalyzer(sample_results, temp_output_dir)
    analyzer.save_results()

    # Check that files were created
    assert (temp_output_dir / "results.csv").exists()
    assert (temp_output_dir / "descriptive_stats.json").exists()
    assert (temp_output_dir / "statistical_tests.json").exists()
    assert (temp_output_dir / "boxplot.png").exists()
    assert (temp_output_dir / "violinplot.png").exists()
    assert (temp_output_dir / "histogram.png").exists()


def test_print_summary(sample_results, temp_output_dir, capsys):
    """Test printing summary."""
    analyzer = PerplexityAnalyzer(sample_results, temp_output_dir)
    analyzer.print_summary()

    captured = capsys.readouterr()
    assert "PERPLEXITY ANALYSIS SUMMARY" in captured.out
    assert "Descriptive Statistics" in captured.out
    assert "Statistical Tests" in captured.out
    assert CATEGORY_TRUE in captured.out
