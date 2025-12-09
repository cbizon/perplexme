"""Configuration constants for PerplexMe."""

from pathlib import Path

# Input data paths
INPUT_DATA_DIR = Path("/Users/bizon/Projects/experiments/SimplePredictions/input_graphs/rbn_6f3")
NODES_FILE = INPUT_DATA_DIR / "nodes.jsonl"
EDGES_FILE = INPUT_DATA_DIR / "edges.jsonl"

# Model configuration
DEFAULT_MODEL = "BioMistral/BioMistral-7B"
DEFAULT_DEVICE = "cuda"  # "cuda", "mps" (Apple Silicon), or "cpu"
DEFAULT_BATCH_SIZE = 16  # Larger batches significantly improve performance on MPS/CUDA

# Predicate configuration
DEFAULT_PREDICATE = "biolink:treats"

# Sentence templates
def format_predicate(predicate: str) -> str:
    """Format a predicate for use in a sentence.

    Removes 'biolink:' prefix and replaces underscores with spaces.

    Args:
        predicate: The predicate string (e.g., "biolink:treats")

    Returns:
        Formatted predicate (e.g., "treats")
    """
    if predicate.startswith("biolink:"):
        predicate = predicate[8:]
    return predicate.replace("_", " ")

def make_sentence(subject_name: str, predicate: str, object_name: str) -> str:
    """Create a sentence from subject, predicate, and object.

    Args:
        subject_name: Name of the subject entity
        predicate: The predicate (will be formatted)
        object_name: Name of the object entity

    Returns:
        Formatted sentence
    """
    predicate_label = format_predicate(predicate)
    return f"{subject_name} {predicate_label} {object_name}"

# Statement categories
CATEGORY_TRUE = "true"
CATEGORY_FALSE_PERMUTED = "false_permuted"
CATEGORY_FALSE_RANDOM = "false_random"
CATEGORY_NONSENSE = "nonsense"

CATEGORIES = [
    CATEGORY_TRUE,
    CATEGORY_FALSE_PERMUTED,
    CATEGORY_FALSE_RANDOM,
    CATEGORY_NONSENSE,
]
