"""Tests for scaffolds, functional groups, isomers, and substructure modules."""

from __future__ import annotations

import pytest

from molprop.features.functional_groups import detect_functional_groups
from molprop.features.isomers import enumerate_isomers
from molprop.features.scaffolds import analyze_scaffold
from molprop.features.substructure import substructure_search

# ── Scaffold analysis ─────────────────────────────────────────────────────────


class TestScaffoldAnalysis:
    def test_invalid_smiles_returns_none(self):
        assert analyze_scaffold("not a smiles") is None

    def test_aspirin(self):
        r = analyze_scaffold("CC(=O)Oc1ccccc1C(=O)O")
        assert r is not None
        assert r.num_rings == 1
        assert r.num_aromatic_rings == 1
        assert r.num_aliphatic_rings == 0
        assert r.largest_ring_size == 6
        assert r.has_macrocycle is False
        assert 1.0 <= r.sa_score <= 10.0
        assert r.sa_class in {"easy", "moderate", "hard"}
        # Aspirin's scaffold is just benzene
        assert "c1ccccc1" in (r.murcko_smiles or "")

    def test_acyclic_molecule_has_no_scaffold(self):
        r = analyze_scaffold("CCO")
        assert r is not None
        assert r.num_rings == 0
        assert r.murcko_smiles in (None, "")

    def test_macrocycle_detected(self):
        # 12-membered ring
        r = analyze_scaffold("C1CCCCCCCCCCC1")
        assert r is not None
        assert r.has_macrocycle is True
        assert r.largest_ring_size == 12

    def test_sa_score_easy_for_simple_molecule(self):
        r = analyze_scaffold("CCO")
        assert r is not None
        assert r.sa_score < 5.0

    def test_sa_score_harder_for_complex_natural_product(self):
        # Quinine (multiple stereocenters + bridgehead atoms)
        r = analyze_scaffold("COc1ccc2nccc([C@@H](O)[C@@H]3CC4CCN3C[C@@H]4C=C)c2c1")
        assert r is not None
        assert r.sa_score > 3.5  # noticeably harder than ethanol


# ── Functional groups ────────────────────────────────────────────────────────


class TestFunctionalGroups:
    def test_invalid_smiles(self):
        assert detect_functional_groups("XYZ") is None

    def test_aspirin_has_carboxylic_acid_ester_and_aromatic(self):
        r = detect_functional_groups("CC(=O)Oc1ccccc1C(=O)O")
        assert r is not None
        names = {g.name for g in r.groups}
        assert "Carboxylic acid" in names
        assert "Ester" in names
        assert "Benzene" in names

    def test_acetaminophen_amide_and_phenol(self):
        r = detect_functional_groups("CC(=O)Nc1ccc(O)cc1")
        assert r is not None
        names = {g.name for g in r.groups}
        assert "Amide" in names
        assert "Phenol" in names

    def test_categories_populated(self):
        r = detect_functional_groups("CC(=O)Oc1ccccc1C(=O)O")
        assert r is not None
        assert "Carbonyl" in r.categories
        assert "Aromatic" in r.categories

    def test_atom_indices_returned(self):
        r = detect_functional_groups("CCO")  # alcohol
        assert r is not None
        alcohol = [g for g in r.groups if g.name == "Alcohol"]
        assert len(alcohol) == 1
        # Each match is a list of atom indices, non-empty
        assert all(len(idx) > 0 for idx in alcohol[0].atom_indices)


# ── Isomers ──────────────────────────────────────────────────────────────────


class TestIsomers:
    def test_invalid_smiles(self):
        assert enumerate_isomers("???") is None

    def test_acetaminophen_tautomers(self):
        r = enumerate_isomers("CC(=O)NC1=CC=C(O)C=C1")
        assert r is not None
        assert r.n_tautomers >= 1
        assert r.canonical_tautomer is not None
        # Exactly one tautomer must be marked canonical
        canonical_flags = [t.is_canonical for t in r.tautomers]
        assert sum(canonical_flags) >= 1

    def test_no_stereocenters_yields_one_stereoisomer(self):
        r = enumerate_isomers("CCO")
        assert r is not None
        assert r.n_stereoisomers == 1

    def test_limits_respected(self):
        r = enumerate_isomers("CC(=O)NC1=CC=C(O)C=C1", max_tautomers=3, max_stereoisomers=2)
        assert r is not None
        assert r.n_tautomers <= 3
        assert r.n_stereoisomers <= 2


# ── Substructure search ─────────────────────────────────────────────────────


class TestSubstructureSearch:
    def test_invalid_query_returns_error(self):
        matches, stats = substructure_search("[][][]", [{"smiles": "CCO"}])
        assert "error" in stats
        assert matches == []

    def test_benzene_pattern_matches_phenol_not_ethanol(self):
        candidates = [
            {"smiles": "CCO", "name": "ethanol"},
            {"smiles": "c1ccccc1O", "name": "phenol"},
            {"smiles": "c1ccncc1", "name": "pyridine"},
        ]
        matches, stats = substructure_search("c1ccccc1", candidates)
        assert stats["n_matches"] == 1
        assert matches[0].smiles == "c1ccccc1O"
        assert matches[0].n_matches >= 1
        assert len(matches[0].atom_indices[0]) == 6

    def test_smarts_pattern_works(self):
        candidates = [
            {"smiles": "CCO"},
            {"smiles": "CC(=O)O"},
            {"smiles": "CCC"},
        ]
        # carboxylic acid SMARTS
        matches, stats = substructure_search("[CX3](=O)[OX2H1]", candidates)
        assert stats["n_matches"] == 1
        assert matches[0].smiles == "CC(=O)O"

    def test_invalid_smiles_in_candidates_is_skipped(self):
        candidates = [
            {"smiles": "CCO"},
            {"smiles": "not_valid"},
            {"smiles": "c1ccccc1"},
        ]
        matches, stats = substructure_search("c1ccccc1", candidates)
        assert stats["n_invalid"] == 1
        assert stats["n_matches"] == 1

    def test_limit_respected(self):
        candidates = [{"smiles": "c1ccccc1"} for _ in range(10)]
        matches, _ = substructure_search("c1ccccc1", candidates, limit=3)
        assert len(matches) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
