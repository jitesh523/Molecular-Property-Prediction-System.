"""Tests for reactions, MMP, and the in-process TTL cache (v2.26 – v2.30)."""

from __future__ import annotations

import asyncio
import time

import pytest

from molprop.features.mmp import find_mmp
from molprop.features.reactions import NAMED_REACTIONS, list_named_reactions, run_reaction
from molprop.serving.cache import _LRUTTLCache, cached_json, endpoint_cache

# ── Reactions ────────────────────────────────────────────────────────────────


class TestReactions:
    def test_invalid_smarts_returns_none(self):
        assert run_reaction("not a smarts", [["CCO"]]) is None

    def test_amide_coupling(self):
        smarts = NAMED_REACTIONS["amide_coupling"]["smarts"]
        r = run_reaction(smarts, [["CC(=O)O", "NCC"]])
        assert r is not None
        assert r.n_reactants == 2
        assert r.n_input_sets == 1
        # Expected canonical SMILES for N-ethylacetamide
        assert "CCNC(C)=O" in r.unique_products

    def test_n_methylation_single_reactant(self):
        smarts = NAMED_REACTIONS["n_methylation"]["smarts"]
        r = run_reaction(smarts, [["NCC"]])
        assert r is not None
        assert r.n_reactants == 1
        # Should methylate the primary amine
        assert any("N" in p and "C" in p for p in r.unique_products)

    def test_skips_invalid_substrates(self):
        smarts = NAMED_REACTIONS["amide_coupling"]["smarts"]
        r = run_reaction(smarts, [["INVALID", "NCC"], ["CC(=O)O", "NCC"]])
        assert r is not None
        assert r.n_product_sets == 1  # only the second pair reacted

    def test_named_catalog_present(self):
        names = {r["name"] for r in list_named_reactions()}
        assert "amide_coupling" in names
        assert "suzuki_coupling" in names
        assert all("smarts" in r and "description" in r for r in list_named_reactions())


# ── Matched Molecular Pairs ─────────────────────────────────────────────────


class TestMMP:
    def test_empty_list(self):
        r = find_mmp([])
        assert r.n_pairs == 0
        assert r.n_valid == 0

    def test_phenol_series_finds_pairs(self):
        """Para-substituted benzenes should share the benzene context."""
        r = find_mmp(["c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1"])
        assert r.n_valid == 3
        assert r.n_pairs >= 1
        # Find a pair where context contains the benzene ring
        assert any("c1ccccc1" in p.context for p in r.pairs)

    def test_invalid_smiles_skipped(self):
        r = find_mmp(["XYZ", "c1ccccc1O", "c1ccccc1N"])
        assert r.n_valid == 2
        assert r.n_input == 3

    def test_max_pairs_cap_respected(self):
        # Build a small series likely to produce > 2 pairs
        series = [f"c1ccc({s})cc1" for s in ["O", "N", "C", "F", "Cl"]]
        r = find_mmp(series, max_pairs=2)
        assert r.n_pairs <= 2

    def test_names_validation(self):
        with pytest.raises(ValueError):
            find_mmp(["CCO", "CCN"], names=["only_one"])


# ── TTL cache ──────────────────────────────────────────────────────────────


class TestLRUTTLCache:
    def test_set_get(self):
        c = _LRUTTLCache(maxsize=3)
        c.set("a", 1, ttl_seconds=10)
        assert c.get("a") == 1
        assert c.stats()["hits"] == 1

    def test_miss_increments_misses(self):
        c = _LRUTTLCache()
        assert c.get("nope") is None
        assert c.stats()["misses"] == 1

    def test_ttl_expiry(self):
        c = _LRUTTLCache()
        c.set("a", 1, ttl_seconds=0.05)
        time.sleep(0.1)
        assert c.get("a") is None  # expired

    def test_lru_eviction(self):
        c = _LRUTTLCache(maxsize=2)
        c.set("a", 1, ttl_seconds=10)
        c.set("b", 2, ttl_seconds=10)
        c.set("c", 3, ttl_seconds=10)  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_clear(self):
        c = _LRUTTLCache()
        c.set("a", 1, ttl_seconds=10)
        c.clear()
        assert c.stats()["size"] == 0
        assert c.get("a") is None


class TestCachedJsonDecorator:
    def test_caches_pure_function(self):
        endpoint_cache.clear()
        calls = {"n": 0}

        @cached_json(ttl_seconds=10)
        async def _fn(x: int) -> int:
            calls["n"] += 1
            return x * 2

        async def _run():
            a = await _fn(5)
            b = await _fn(5)
            return a, b

        a, b = asyncio.run(_run())
        assert a == 10 and b == 10
        # Only one underlying call
        assert calls["n"] == 1

    def test_bypasses_unhashable_args(self):
        endpoint_cache.clear()
        calls = {"n": 0}

        @cached_json(ttl_seconds=10)
        async def _fn(obj) -> int:
            calls["n"] += 1
            return 42

        class _NotJSON:
            pass

        async def _run():
            await _fn(_NotJSON())
            await _fn(_NotJSON())

        asyncio.run(_run())
        # Both calls should hit the function (no caching of unhashable args)
        assert calls["n"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
