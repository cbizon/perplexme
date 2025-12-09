"""Calculate perplexity for individual entities to use as normalization baseline."""

from typing import Dict, List, Set
from dataclasses import dataclass
from tqdm import tqdm

from src.data_loader import KnowledgeGraph
from src.perplexity import PerplexityCalculator


@dataclass
class EntityPerplexityResult:
    """Perplexity measurements for a single entity."""
    entity_id: str
    entity_name: str
    isolated_perplexity: float  # Just the entity name
    neutral_perplexity: float   # "This is about {entity}"
    typed_perplexity: float     # "The drug/disease {entity}"
    kg_degree: int              # Number of edges this entity participates in


def get_entity_category_label(kg: KnowledgeGraph, entity_id: str) -> str:
    """Get a natural language label for the entity category.

    Args:
        kg: Knowledge graph
        entity_id: Entity ID

    Returns:
        Label like "drug", "disease", "gene", etc.
    """
    try:
        node = kg.get_node(entity_id)
        category = node.category.lower()

        # Map biolink categories to natural language
        if 'disease' in category:
            return 'disease'
        elif 'chemical' in category or 'drug' in category or 'molecule' in category:
            return 'drug'
        elif 'gene' in category or 'protein' in category:
            return 'gene'
        elif 'phenotype' in category or 'symptom' in category:
            return 'phenotype'
        elif 'pathway' in category or 'process' in category:
            return 'biological process'
        else:
            return 'entity'  # Generic fallback
    except KeyError:
        return 'entity'


def calculate_entity_perplexities(
    kg: KnowledgeGraph,
    calculator: PerplexityCalculator,
    predicate: str,
) -> Dict[str, EntityPerplexityResult]:
    """Calculate perplexity for all entities involved in edges with a specific predicate.

    Args:
        kg: Knowledge graph
        calculator: Perplexity calculator
        predicate: Predicate to filter by (e.g., "biolink:treats")

    Returns:
        Dictionary mapping entity_id to EntityPerplexityResult
    """
    # Collect all unique entities (subjects and objects) for this predicate
    entity_ids: Set[str] = set()
    entity_ids.update(kg.get_subjects_for_predicate(predicate))
    entity_ids.update(kg.get_objects_for_predicate(predicate))

    print(f"\nCalculating entity perplexities for {len(entity_ids)} unique entities...")

    # Calculate degree (edge count) for each entity
    entity_degrees: Dict[str, int] = {}
    for entity_id in entity_ids:
        # Count edges across all predicates
        degree = 0
        for edges in kg.edges_by_predicate.values():
            degree += sum(1 for e in edges if e.subject == entity_id or e.object == entity_id)
        entity_degrees[entity_id] = degree

    results: Dict[str, EntityPerplexityResult] = {}

    # Process entities in batches for efficiency
    entity_list = list(entity_ids)

    for entity_id in tqdm(entity_list, desc="Processing entities"):
        try:
            node = kg.get_node(entity_id)
            entity_name = node.name

            # Option A: Isolated entity
            isolated_text = entity_name
            isolated_ppl, _ = calculator.calculate_perplexity_single(isolated_text)

            # Option B: Neutral context
            neutral_text = f"This is about {entity_name}"
            neutral_ppl, _ = calculator.calculate_perplexity_single(neutral_text)

            # Option C: Typed context
            category_label = get_entity_category_label(kg, entity_id)
            typed_text = f"The {category_label} {entity_name}"
            typed_ppl, _ = calculator.calculate_perplexity_single(typed_text)

            results[entity_id] = EntityPerplexityResult(
                entity_id=entity_id,
                entity_name=entity_name,
                isolated_perplexity=isolated_ppl,
                neutral_perplexity=neutral_ppl,
                typed_perplexity=typed_ppl,
                kg_degree=entity_degrees.get(entity_id, 0),
            )

        except KeyError:
            # Entity not in nodes file, skip
            continue

    print(f"Calculated perplexities for {len(results)} entities")
    return results


def add_entity_perplexities_to_results(
    results: List[Dict],
    entity_perplexities: Dict[str, EntityPerplexityResult],
) -> List[Dict]:
    """Add entity perplexity columns to statement results.

    Args:
        results: List of result dictionaries from perplexity calculation
        entity_perplexities: Dictionary of entity perplexity results

    Returns:
        Updated results with added columns
    """
    for result in results:
        subject_id = result['subject_id']
        object_id = result['object_id']

        # Add subject perplexities
        if subject_id in entity_perplexities:
            subj = entity_perplexities[subject_id]
            result['subject_isolated_perp'] = subj.isolated_perplexity
            result['subject_neutral_perp'] = subj.neutral_perplexity
            result['subject_typed_perp'] = subj.typed_perplexity
            result['subject_kg_degree'] = subj.kg_degree
        else:
            result['subject_isolated_perp'] = None
            result['subject_neutral_perp'] = None
            result['subject_typed_perp'] = None
            result['subject_kg_degree'] = 0

        # Add object perplexities
        if object_id in entity_perplexities:
            obj = entity_perplexities[object_id]
            result['object_isolated_perp'] = obj.isolated_perplexity
            result['object_neutral_perp'] = obj.neutral_perplexity
            result['object_typed_perp'] = obj.typed_perplexity
            result['object_kg_degree'] = obj.kg_degree
        else:
            result['object_isolated_perp'] = None
            result['object_neutral_perp'] = None
            result['object_typed_perp'] = None
            result['object_kg_degree'] = 0

    return results
