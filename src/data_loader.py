"""Load and index KGX format knowledge graph data."""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Node:
    """Represents a knowledge graph node."""
    id: str
    name: str
    category: str  # Most specific category (first in list)
    all_categories: List[str]


@dataclass
class Edge:
    """Represents a knowledge graph edge."""
    subject: str
    predicate: str
    object: str
    edge_id: str
    qualifiers: Dict[str, str]


class KnowledgeGraph:
    """Load and index knowledge graph data."""

    def __init__(self, nodes_file: Path, edges_file: Path):
        """Initialize the knowledge graph loader.

        Args:
            nodes_file: Path to nodes JSONL file
            edges_file: Path to edges JSONL file
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.nodes: Dict[str, Node] = {}
        self.edges_by_predicate: Dict[str, List[Edge]] = {}
        self.subjects_by_predicate: Dict[str, Set[str]] = {}
        self.objects_by_predicate: Dict[str, Set[str]] = {}
        self.valid_pairs_by_predicate: Dict[str, Set[Tuple[str, str]]] = {}
        self.nodes_by_category: Dict[str, Set[str]] = {}

    def load_nodes(self) -> None:
        """Load nodes from JSONL file."""
        print(f"Loading nodes from {self.nodes_file}...")
        with open(self.nodes_file, 'r') as f:
            for line in tqdm(f, desc="Loading nodes"):
                node_data = json.loads(line)
                node_id = node_data['id']
                name = node_data.get('name', node_id)
                categories = node_data.get('category', [])

                if not categories:
                    category = 'unknown'
                else:
                    category = categories[0]

                node = Node(
                    id=node_id,
                    name=name,
                    category=category,
                    all_categories=categories
                )
                self.nodes[node_id] = node

                # Index by category
                if category not in self.nodes_by_category:
                    self.nodes_by_category[category] = set()
                self.nodes_by_category[category].add(node_id)

        print(f"Loaded {len(self.nodes)} nodes")

    def load_edges(self, predicate_filter: str = None) -> None:
        """Load edges from JSONL file.

        Args:
            predicate_filter: If provided, only load edges with this predicate
        """
        print(f"Loading edges from {self.edges_file}...")
        if predicate_filter:
            print(f"Filtering for predicate: {predicate_filter}")

        with open(self.edges_file, 'r') as f:
            for line in tqdm(f, desc="Loading edges"):
                edge_data = json.loads(line)
                predicate = edge_data['predicate']

                # Apply filter if specified
                if predicate_filter and predicate != predicate_filter:
                    continue

                subject = edge_data['subject']
                obj = edge_data['object']
                edge_id = edge_data.get('id', f"{subject}_{predicate}_{obj}")

                # Extract qualifiers
                qualifiers = {}
                for key, value in edge_data.items():
                    if key.endswith('_qualifier') or key == 'qualified_predicate':
                        qualifiers[key] = value

                edge = Edge(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    edge_id=edge_id,
                    qualifiers=qualifiers
                )

                # Index by predicate
                if predicate not in self.edges_by_predicate:
                    self.edges_by_predicate[predicate] = []
                    self.subjects_by_predicate[predicate] = set()
                    self.objects_by_predicate[predicate] = set()
                    self.valid_pairs_by_predicate[predicate] = set()

                self.edges_by_predicate[predicate].append(edge)
                self.subjects_by_predicate[predicate].add(subject)
                self.objects_by_predicate[predicate].add(obj)
                self.valid_pairs_by_predicate[predicate].add((subject, obj))

        total_edges = sum(len(edges) for edges in self.edges_by_predicate.values())
        print(f"Loaded {total_edges} edges across {len(self.edges_by_predicate)} predicates")

        if predicate_filter and predicate_filter in self.edges_by_predicate:
            print(f"  {predicate_filter}: {len(self.edges_by_predicate[predicate_filter])} edges")

    def get_edges(self, predicate: str) -> List[Edge]:
        """Get all edges for a specific predicate.

        Args:
            predicate: The predicate to filter by

        Returns:
            List of edges with that predicate
        """
        return self.edges_by_predicate.get(predicate, [])

    def get_subjects_for_predicate(self, predicate: str) -> Set[str]:
        """Get all subject IDs that appear in edges with this predicate.

        Args:
            predicate: The predicate

        Returns:
            Set of subject node IDs
        """
        return self.subjects_by_predicate.get(predicate, set())

    def get_objects_for_predicate(self, predicate: str) -> Set[str]:
        """Get all object IDs that appear in edges with this predicate.

        Args:
            predicate: The predicate

        Returns:
            Set of object node IDs
        """
        return self.objects_by_predicate.get(predicate, set())

    def is_valid_pair(self, predicate: str, subject: str, obj: str) -> bool:
        """Check if a (subject, object) pair is a valid edge for this predicate.

        Args:
            predicate: The predicate
            subject: Subject node ID
            obj: Object node ID

        Returns:
            True if this is a valid edge
        """
        return (subject, obj) in self.valid_pairs_by_predicate.get(predicate, set())

    def get_node(self, node_id: str) -> Node:
        """Get a node by ID.

        Args:
            node_id: The node ID

        Returns:
            Node object

        Raises:
            KeyError: If node not found
        """
        return self.nodes[node_id]

    def get_nodes_by_category(self, category: str) -> Set[str]:
        """Get all node IDs with a specific category.

        Args:
            category: The category (e.g., "biolink:Disease")

        Returns:
            Set of node IDs
        """
        return self.nodes_by_category.get(category, set())

    def get_subject_object_categories(self, predicate: str) -> Tuple[Set[str], Set[str]]:
        """Get the categories used by subjects and objects for this predicate.

        Args:
            predicate: The predicate

        Returns:
            Tuple of (subject_categories, object_categories)
        """
        subject_categories = set()
        object_categories = set()

        for edge in self.get_edges(predicate):
            if edge.subject in self.nodes:
                subject_categories.add(self.nodes[edge.subject].category)
            if edge.object in self.nodes:
                object_categories.add(self.nodes[edge.object].category)

        return subject_categories, object_categories


def load_kg(nodes_file: Path, edges_file: Path, predicate: str = None) -> KnowledgeGraph:
    """Convenience function to load a knowledge graph.

    Args:
        nodes_file: Path to nodes JSONL file
        edges_file: Path to edges JSONL file
        predicate: Optional predicate to filter edges

    Returns:
        Loaded KnowledgeGraph
    """
    kg = KnowledgeGraph(nodes_file, edges_file)
    kg.load_nodes()
    kg.load_edges(predicate_filter=predicate)
    return kg
