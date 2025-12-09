"""Common CLI utilities shared between main programs."""

import argparse
from pathlib import Path
from datetime import datetime

from src.config import (
    NODES_FILE,
    EDGES_FILE,
    DEFAULT_MODEL,
    DEFAULT_PREDICATE,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
)
from src.data_loader import load_kg, KnowledgeGraph
from src.perplexity import PerplexityCalculator


def create_base_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common arguments.

    Args:
        description: Description for the argument parser

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--predicate",
        type=str,
        default=DEFAULT_PREDICATE,
        help=f"Predicate to analyze (default: {DEFAULT_PREDICATE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to use (cuda or cpu, default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for perplexity calculation (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--nodes-file",
        type=str,
        default=str(NODES_FILE),
        help=f"Path to nodes file (default: {NODES_FILE})",
    )
    parser.add_argument(
        "--edges-file",
        type=str,
        default=str(EDGES_FILE),
        help=f"Path to edges file (default: {EDGES_FILE})",
    )

    return parser


def setup_output_directory(
    output_dir: str | None,
    predicate: str,
    prefix: str = ""
) -> Path:
    """Set up output directory with timestamp.

    Args:
        output_dir: User-specified output directory, or None for auto-generated
        predicate: Predicate being analyzed
        prefix: Prefix for auto-generated directory name

    Returns:
        Path to output directory
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predicate_name = predicate.replace("biolink:", "").replace(":", "_")
        if prefix:
            output_dir = f"outputs/{prefix}_{predicate_name}_{timestamp}"
        else:
            output_dir = f"outputs/{predicate_name}_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_knowledge_graph(
    nodes_file: str,
    edges_file: str,
    predicate: str
) -> KnowledgeGraph:
    """Load knowledge graph and validate it has edges for the predicate.

    Args:
        nodes_file: Path to nodes file
        edges_file: Path to edges file
        predicate: Predicate to filter by

    Returns:
        Loaded KnowledgeGraph

    Raises:
        SystemExit: If no edges found for predicate
    """
    kg = load_kg(
        Path(nodes_file),
        Path(edges_file),
        predicate=predicate
    )

    edges = kg.get_edges(predicate)
    if len(edges) == 0:
        print(f"ERROR: No edges found for predicate {predicate}")
        raise SystemExit(1)

    print(f"Found {len(edges)} edges for predicate {predicate}")
    return kg


def create_perplexity_calculator(
    model_name: str,
    device: str,
    batch_size: int
) -> PerplexityCalculator:
    """Create a perplexity calculator with the given configuration.

    Args:
        model_name: HuggingFace model name
        device: Device to use (cuda, mps, cpu)
        batch_size: Batch size for processing

    Returns:
        Configured PerplexityCalculator
    """
    return PerplexityCalculator(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )


def print_banner(title: str) -> None:
    """Print a formatted banner.

    Args:
        title: Title to display in banner
    """
    print("=" * 80)
    print(title)
    print("=" * 80)
