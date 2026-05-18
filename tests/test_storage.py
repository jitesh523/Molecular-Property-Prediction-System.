"""Tests for the SQLite-backed compound library (v2.5.0)."""

from __future__ import annotations

import pytest

from molprop.storage.library import CompoundLibrary


@pytest.fixture
def lib(tmp_path):
    """Fresh on-disk library per test (isolated)."""
    return CompoundLibrary(tmp_path / "test.db")


def test_add_and_get(lib):
    rec = lib.add("CCO", name="Ethanol", project="p1", tags=["solvent"])
    assert rec["id"] is not None
    assert rec["smiles"] == "CCO"
    assert rec["name"] == "Ethanol"
    assert rec["tags"] == ["solvent"]

    fetched = lib.get(rec["id"])
    assert fetched == rec


def test_upsert_idempotency(lib):
    """Adding the same (smiles, project) twice updates rather than duplicates."""
    a = lib.add("CCO", name="Ethanol", project="p1")
    b = lib.add("CCO", name="Updated", project="p1", tags=["new"])

    assert a["id"] == b["id"]
    assert b["name"] == "Updated"
    assert b["tags"] == ["new"]
    assert lib.count(project="p1") == 1


def test_same_smiles_different_projects(lib):
    """Same SMILES in different projects should be separate rows."""
    a = lib.add("CCO", project="p1")
    b = lib.add("CCO", project="p2")
    assert a["id"] != b["id"]
    assert lib.count() == 2


def test_list_filters(lib):
    lib.add("CCO", name="Ethanol", project="p1", tags=["solvent"])
    lib.add("c1ccccc1", name="Benzene", project="p1", tags=["aromatic"])
    lib.add("CCN", name="Ethylamine", project="p2", tags=["amine"])

    assert len(lib.list(project="p1")) == 2
    assert len(lib.list(project="p2")) == 1
    assert len(lib.list(tag="aromatic")) == 1
    assert lib.list(tag="aromatic")[0]["smiles"] == "c1ccccc1"

    hits = lib.list(search="thanol")
    assert len(hits) == 1
    assert hits[0]["name"] == "Ethanol"


def test_update(lib):
    rec = lib.add("CCO", name="Original", tags=["a"])
    updated = lib.update(rec["id"], name="New", tags=["a", "b"], notes="hello")
    assert updated["name"] == "New"
    assert updated["tags"] == ["a", "b"]
    assert updated["notes"] == "hello"


def test_update_missing_returns_none(lib):
    assert lib.update(9999, name="x") is None


def test_delete(lib):
    rec = lib.add("CCO")
    assert lib.delete(rec["id"]) is True
    assert lib.get(rec["id"]) is None
    assert lib.delete(rec["id"]) is False  # second delete is no-op


def test_projects_and_tags_collection(lib):
    lib.add("CCO", project="p1", tags=["t1", "t2"])
    lib.add("CCN", project="p2", tags=["t2", "t3"])
    assert set(lib.projects()) == {"p1", "p2"}
    assert set(lib.all_tags()) == {"t1", "t2", "t3"}


def test_persistence_across_reopens(tmp_path):
    """The DB must survive instance recreation."""
    db = tmp_path / "persist.db"
    lib1 = CompoundLibrary(db)
    lib1.add("CCO", name="Ethanol", project="x")

    lib2 = CompoundLibrary(db)
    assert lib2.count() == 1
    assert lib2.list()[0]["smiles"] == "CCO"
