"""Tests for data_loader module."""

from pathlib import Path
import pytest
from src.data_loader import KnowledgeGraph, load_kg


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_data"
NODES_FILE = FIXTURES_DIR / "nodes.jsonl"
EDGES_FILE = FIXTURES_DIR / "edges.jsonl"


def test_load_nodes():
    """Test loading nodes from JSONL."""
    kg = KnowledgeGraph(NODES_FILE, EDGES_FILE)
    kg.load_nodes()

    assert len(kg.nodes) == 8
    assert "DRUG:001" in kg.nodes
    assert kg.nodes["DRUG:001"].name == "Metformin"
    assert kg.nodes["DRUG:001"].category == "biolink:SmallMolecule"
    assert "DISEASE:001" in kg.nodes
    assert kg.nodes["DISEASE:001"].name == "type 2 diabetes"


def test_load_edges():
    """Test loading edges from JSONL."""
    kg = KnowledgeGraph(NODES_FILE, EDGES_FILE)
    kg.load_nodes()
    kg.load_edges()

    assert len(kg.edges_by_predicate) == 2
    assert "biolink:treats" in kg.edges_by_predicate
    assert "biolink:affects" in kg.edges_by_predicate
    assert len(kg.edges_by_predicate["biolink:treats"]) == 3
    assert len(kg.edges_by_predicate["biolink:affects"]) == 1


def test_load_edges_with_filter():
    """Test loading edges with predicate filter."""
    kg = KnowledgeGraph(NODES_FILE, EDGES_FILE)
    kg.load_nodes()
    kg.load_edges(predicate_filter="biolink:treats")

    assert len(kg.edges_by_predicate) == 1
    assert "biolink:treats" in kg.edges_by_predicate
    assert len(kg.edges_by_predicate["biolink:treats"]) == 3


def test_get_edges():
    """Test getting edges by predicate."""
    kg = load_kg(NODES_FILE, EDGES_FILE)
    treats_edges = kg.get_edges("biolink:treats")

    assert len(treats_edges) == 3
    assert treats_edges[0].subject == "DRUG:001"
    assert treats_edges[0].object == "DISEASE:001"
    assert treats_edges[0].predicate == "biolink:treats"


def test_get_subjects_objects_for_predicate():
    """Test getting subjects and objects for a predicate."""
    kg = load_kg(NODES_FILE, EDGES_FILE)

    subjects = kg.get_subjects_for_predicate("biolink:treats")
    assert len(subjects) == 3
    assert "DRUG:001" in subjects
    assert "DRUG:002" in subjects
    assert "DRUG:003" in subjects

    objects = kg.get_objects_for_predicate("biolink:treats")
    assert len(objects) == 3
    assert "DISEASE:001" in objects
    assert "DISEASE:002" in objects
    assert "DISEASE:003" in objects


def test_is_valid_pair():
    """Test checking if a pair is a valid edge."""
    kg = load_kg(NODES_FILE, EDGES_FILE)

    assert kg.is_valid_pair("biolink:treats", "DRUG:001", "DISEASE:001")
    assert not kg.is_valid_pair("biolink:treats", "DRUG:001", "DISEASE:002")
    assert not kg.is_valid_pair("biolink:treats", "DRUG:999", "DISEASE:001")


def test_get_node():
    """Test getting a node by ID."""
    kg = load_kg(NODES_FILE, EDGES_FILE)

    node = kg.get_node("DRUG:001")
    assert node.id == "DRUG:001"
    assert node.name == "Metformin"
    assert node.category == "biolink:SmallMolecule"

    with pytest.raises(KeyError):
        kg.get_node("NONEXISTENT")


def test_get_nodes_by_category():
    """Test getting nodes by category."""
    kg = load_kg(NODES_FILE, EDGES_FILE)

    drugs = kg.get_nodes_by_category("biolink:SmallMolecule")
    assert len(drugs) == 3
    assert "DRUG:001" in drugs
    assert "DRUG:002" in drugs
    assert "DRUG:003" in drugs

    diseases = kg.get_nodes_by_category("biolink:Disease")
    assert len(diseases) == 3


def test_get_subject_object_categories():
    """Test getting categories used by subjects and objects."""
    kg = load_kg(NODES_FILE, EDGES_FILE)

    subj_cats, obj_cats = kg.get_subject_object_categories("biolink:treats")

    assert "biolink:SmallMolecule" in subj_cats
    assert "biolink:Disease" in obj_cats
    assert len(subj_cats) == 1
    assert len(obj_cats) == 1
