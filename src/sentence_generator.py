"""Generate sentences from knowledge graph edges."""

import random
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from src.data_loader import KnowledgeGraph
from src.config import (
    make_sentence,
    CATEGORY_TRUE,
    CATEGORY_FALSE_PERMUTED,
    CATEGORY_FALSE_RANDOM,
    CATEGORY_NONSENSE,
)


@dataclass
class Statement:
    """Represents a generated statement."""
    text: str
    category: str
    edge_id: str
    subject_id: str
    object_id: str


class SentenceGenerator:
    """Generate true, false, and nonsense sentences from a knowledge graph."""

    def __init__(self, kg: KnowledgeGraph, predicate: str, random_seed: int = 42):
        """Initialize the sentence generator.

        Args:
            kg: The knowledge graph
            predicate: The predicate to generate sentences for
            random_seed: Random seed for reproducibility
        """
        self.kg = kg
        self.predicate = predicate
        self.random = random.Random(random_seed)
        self.edges = kg.get_edges(predicate)
        self.subjects = list(kg.get_subjects_for_predicate(predicate))
        self.objects = list(kg.get_objects_for_predicate(predicate))

        # Get subject and object categories for this predicate
        self.subject_categories, self.object_categories = kg.get_subject_object_categories(predicate)

        # Pre-build candidate lists for false-random statements (correct types)
        self.false_random_subject_candidates = []
        for cat in self.subject_categories:
            self.false_random_subject_candidates.extend(list(kg.get_nodes_by_category(cat)))

        self.false_random_object_candidates = []
        for cat in self.object_categories:
            self.false_random_object_candidates.extend(list(kg.get_nodes_by_category(cat)))

        # Pre-build candidate lists for nonsense statements (swapped types)
        self.nonsense_subject_candidates = []
        for cat in self.object_categories:  # Note: swapped
            self.nonsense_subject_candidates.extend(list(kg.get_nodes_by_category(cat)))

        self.nonsense_object_candidates = []
        for cat in self.subject_categories:  # Note: swapped
            self.nonsense_object_candidates.extend(list(kg.get_nodes_by_category(cat)))

    def generate_true_statements(self) -> List[Statement]:
        """Generate true statements from real edges.

        Returns:
            List of true statements
        """
        statements = []
        for edge in tqdm(self.edges, desc="Generating true statements"):
            try:
                subject_node = self.kg.get_node(edge.subject)
                object_node = self.kg.get_node(edge.object)

                text = make_sentence(subject_node.name, edge.predicate, object_node.name)

                statement = Statement(
                    text=text,
                    category=CATEGORY_TRUE,
                    edge_id=edge.edge_id,
                    subject_id=edge.subject,
                    object_id=edge.object,
                )
                statements.append(statement)
            except KeyError:
                # Skip edges with missing nodes
                continue

        return statements

    def generate_false_permuted_statements(self) -> List[Statement]:
        """Generate false statements by permuting subjects and objects.

        For each true edge, randomly select a different object from the pool
        of objects that appear in edges with this predicate. Ensure the pair
        is not a valid edge. Don't check type compatibility.

        Returns:
            List of false permuted statements
        """
        statements = []
        max_attempts = 100

        for edge in tqdm(self.edges, desc="Generating false-permuted statements"):
            try:
                subject_node = self.kg.get_node(edge.subject)

                # Try to find a different object that doesn't form a valid edge
                for _ in range(max_attempts):
                    new_object_id = self.random.choice(self.objects)

                    # Skip if same object or if it's a valid edge
                    if new_object_id == edge.object:
                        continue
                    if self.kg.is_valid_pair(self.predicate, edge.subject, new_object_id):
                        continue

                    # Found a good false pair
                    object_node = self.kg.get_node(new_object_id)
                    text = make_sentence(subject_node.name, edge.predicate, object_node.name)

                    statement = Statement(
                        text=text,
                        category=CATEGORY_FALSE_PERMUTED,
                        edge_id=f"{edge.edge_id}_false_perm",
                        subject_id=edge.subject,
                        object_id=new_object_id,
                    )
                    statements.append(statement)
                    break

            except KeyError:
                # Skip edges with missing nodes
                continue

        return statements

    def generate_false_random_statements(self) -> List[Statement]:
        """Generate false statements with random entity pairs.

        For each true edge, randomly select subject and object from
        appropriate categories that are NOT in any valid edge of this predicate.

        Returns:
            List of false random statements
        """
        statements = []
        max_attempts = 1000

        if not self.false_random_subject_candidates or not self.false_random_object_candidates:
            return statements

        for edge in tqdm(self.edges, desc="Generating false-random statements"):
            # Try to find a random pair that's not a valid edge
            for _ in range(max_attempts):
                subject_id = self.random.choice(self.false_random_subject_candidates)
                object_id = self.random.choice(self.false_random_object_candidates)

                if not self.kg.is_valid_pair(self.predicate, subject_id, object_id):
                    try:
                        subject_node = self.kg.get_node(subject_id)
                        object_node = self.kg.get_node(object_id)

                        text = make_sentence(subject_node.name, self.predicate, object_node.name)

                        statement = Statement(
                            text=text,
                            category=CATEGORY_FALSE_RANDOM,
                            edge_id=f"{edge.edge_id}_false_rand",
                            subject_id=subject_id,
                            object_id=object_id,
                        )
                        statements.append(statement)
                        break
                    except KeyError:
                        continue

        return statements

    def generate_nonsense_statements(self) -> List[Statement]:
        """Generate nonsense statements with wrong entity types.

        For each true edge, swap the subject and object categories and
        randomly select entities with these wrong types.

        Returns:
            List of nonsense statements
        """
        statements = []
        max_attempts = 1000

        if not self.nonsense_subject_candidates or not self.nonsense_object_candidates:
            return statements

        for edge in tqdm(self.edges, desc="Generating nonsense statements"):
            # Pick random entities with wrong types
            for _ in range(max_attempts):
                subject_id = self.random.choice(self.nonsense_subject_candidates)
                object_id = self.random.choice(self.nonsense_object_candidates)

                try:
                    subject_node = self.kg.get_node(subject_id)
                    object_node = self.kg.get_node(object_id)

                    text = make_sentence(subject_node.name, self.predicate, object_node.name)

                    statement = Statement(
                        text=text,
                        category=CATEGORY_NONSENSE,
                        edge_id=f"{edge.edge_id}_nonsense",
                        subject_id=subject_id,
                        object_id=object_id,
                    )
                    statements.append(statement)
                    break
                except KeyError:
                    continue

        return statements

    def generate_all_statements(self) -> List[Statement]:
        """Generate all four categories of statements.

        Returns:
            List of all statements (true, false-permuted, false-random, nonsense)
        """
        print(f"Generating statements for {len(self.edges)} edges...")

        true_statements = self.generate_true_statements()
        false_permuted = self.generate_false_permuted_statements()
        false_random = self.generate_false_random_statements()
        nonsense = self.generate_nonsense_statements()

        all_statements = true_statements + false_permuted + false_random + nonsense

        print(f"Generated {len(true_statements)} true statements")
        print(f"Generated {len(false_permuted)} false-permuted statements")
        print(f"Generated {len(false_random)} false-random statements")
        print(f"Generated {len(nonsense)} nonsense statements")
        print(f"Total: {len(all_statements)} statements")

        return all_statements
