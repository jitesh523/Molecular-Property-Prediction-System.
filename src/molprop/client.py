"""
molprop Python SDK — typed client for the molprop REST API.

Quick start:

    from molprop.client import MolpropClient

    client = MolpropClient("http://localhost:8000")
    print(client.health())

    pred = client.predict("CC(=O)Oc1ccccc1C(=O)O")
    print(pred["prediction"])

    scaf = client.scaffold("CC(=O)NC1=CC=C(O)C=C1")
    print(scaf["sa_score"], scaf["murcko_smiles"])

    # Save to library
    compound = client.library_save("CCO", name="Ethanol", tags=["solvent"])

    # Run a substructure search across the saved library
    hits = client.substructure("c1ccccc1")
    for h in hits["matches"]:
        print(h["smiles"], h["n_matches"])

The client wraps `requests` and exposes one method per REST endpoint with
typed kwargs and a uniform error path (raises `MolpropAPIError` on non-2xx).
"""

from __future__ import annotations

from typing import Any, Optional

import requests


class MolpropAPIError(RuntimeError):
    """Raised when the molprop API returns a non-2xx status."""

    def __init__(self, status_code: int, detail: Any, url: str) -> None:
        self.status_code = status_code
        self.detail = detail
        self.url = url
        super().__init__(f"{status_code} on {url}: {detail}")


class MolpropClient:
    """Thin, typed HTTP client for the molprop REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ── internal ──────────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.request(method, url, **kwargs)
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise MolpropAPIError(resp.status_code, detail, url)
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.content

    def _get(self, path: str, **params) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=clean)

    def _post(self, path: str, json: Optional[dict] = None) -> Any:
        return self._request("POST", path, json=json)

    def _patch(self, path: str, json: dict) -> Any:
        return self._request("PATCH", path, json=json)

    def _delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # ── meta ──────────────────────────────────────────────────────────────────
    def health(self) -> dict:
        return self._get("/health")

    def version(self) -> dict:
        return self._get("/version")

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, smiles: str) -> dict:
        return self._post("/predict", {"smiles": smiles})

    def predict_batch(self, smiles_list: list[str]) -> dict:
        return self._post("/predict/batch", {"smiles_list": smiles_list})

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, n: int = 10) -> dict:
        return self._post("/generate", {"n": n})

    def generate_smart(
        self,
        n: int = 10,
        max_attempts: int = 200,
        mw_min: Optional[float] = None,
        mw_max: Optional[float] = None,
        logp_min: Optional[float] = None,
        logp_max: Optional[float] = None,
        qed_min: Optional[float] = None,
        tpsa_min: Optional[float] = None,
        tpsa_max: Optional[float] = None,
    ) -> dict:
        body = {"n": n, "max_attempts": max_attempts}
        for k, v in {
            "mw_min": mw_min,
            "mw_max": mw_max,
            "logp_min": logp_min,
            "logp_max": logp_max,
            "qed_min": qed_min,
            "tpsa_min": tpsa_min,
            "tpsa_max": tpsa_max,
        }.items():
            if v is not None:
                body[k] = v
        return self._post("/generate/smart", body)

    # ── ADMET ─────────────────────────────────────────────────────────────────
    def admet(self, smiles: str) -> dict:
        return self._post("/admet", {"smiles": smiles})

    def admet_batch(self, smiles_list: list[str]) -> dict:
        return self._post("/admet/batch", {"smiles_list": smiles_list})

    # ── cheminformatics ───────────────────────────────────────────────────────
    def scaffold(self, smiles: str) -> dict:
        return self._post("/scaffold", {"smiles": smiles})

    def scaffold_batch(self, smiles_list: list[str]) -> dict:
        return self._post("/scaffold/batch", {"smiles_list": smiles_list})

    def functional_groups(self, smiles: str) -> dict:
        return self._post("/functional_groups", {"smiles": smiles})

    def isomers(
        self,
        smiles: str,
        max_tautomers: int = 25,
        max_stereoisomers: int = 16,
    ) -> dict:
        return self._post(
            "/isomers",
            {
                "smiles": smiles,
                "max_tautomers": max_tautomers,
                "max_stereoisomers": max_stereoisomers,
            },
        )

    def substructure(
        self,
        query: str,
        candidates: Optional[list[str]] = None,
        project: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "limit": limit}
        if candidates is not None:
            body["candidates"] = candidates
        if project is not None:
            body["project"] = project
        return self._post("/substructure", body)

    def compare(self, smiles_a: str, smiles_b: str) -> dict:
        return self._post("/compare", {"smiles_a": smiles_a, "smiles_b": smiles_b})

    def mcs(
        self,
        smiles_a: str,
        smiles_b: str,
        timeout: int = 5,
        complete_rings_only: bool = True,
        ring_matches_ring_only: bool = True,
    ) -> dict:
        return self._post(
            "/mcs",
            {
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "timeout": timeout,
                "complete_rings_only": complete_rings_only,
                "ring_matches_ring_only": ring_matches_ring_only,
            },
        )

    def rgroups(self, core: str, smiles_list: list[str]) -> dict:
        return self._post("/rgroups", {"core": core, "smiles_list": smiles_list})

    def react(
        self,
        substrates: list[list[str]],
        smarts: Optional[str] = None,
        named: Optional[str] = None,
    ) -> dict:
        """Apply a reaction SMARTS (or a named reaction) to substrate tuples."""
        body: dict[str, Any] = {"substrates": substrates}
        if smarts is not None:
            body["smarts"] = smarts
        if named is not None:
            body["named"] = named
        return self._post("/react", body)

    def react_named(self) -> dict:
        """Return the catalog of built-in named reactions."""
        return self._get("/react/named")

    def mmp(
        self,
        smiles_list: list[str],
        names: Optional[list[str]] = None,
        max_substituent_atoms: int = 10,
        max_pairs: int = 500,
    ) -> dict:
        """Single-cut Matched Molecular Pairs analysis."""
        body: dict[str, Any] = {
            "smiles_list": smiles_list,
            "max_substituent_atoms": max_substituent_atoms,
            "max_pairs": max_pairs,
        }
        if names is not None:
            body["names"] = names
        return self._post("/mmp", body)

    def alerts(self, smiles: str, catalogs: Optional[list[str]] = None) -> dict:
        body: dict[str, Any] = {"smiles": smiles}
        if catalogs is not None:
            body["catalogs"] = catalogs
        return self._post("/alerts", body)

    def standardize(self, smiles: str) -> dict:
        return self._post("/standardize", {"smiles": smiles})

    def conformer(self, smiles: str) -> dict:
        return self._post("/conformer", {"smiles": smiles})

    # ── similarity search ─────────────────────────────────────────────────────
    def search_similar(
        self,
        smiles: str,
        top_k: int = 10,
        method: str = "morgan",
        threshold: float = 0.0,
    ) -> dict:
        return self._post(
            "/search/similar",
            {"smiles": smiles, "top_k": top_k, "method": method, "threshold": threshold},
        )

    # ── reports ───────────────────────────────────────────────────────────────
    def report(
        self,
        smiles: str,
        include_admet: bool = True,
        include_scaffold: bool = True,
        include_functional_groups: bool = True,
        include_descriptors: bool = True,
    ) -> dict:
        return self._post(
            "/report",
            {
                "smiles": smiles,
                "include_admet": include_admet,
                "include_scaffold": include_scaffold,
                "include_functional_groups": include_functional_groups,
                "include_descriptors": include_descriptors,
            },
        )

    # ── compound library ──────────────────────────────────────────────────────
    def library_save(
        self,
        smiles: str,
        name: Optional[str] = None,
        project: str = "default",
        tags: Optional[list[str]] = None,
        properties: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> dict:
        return self._post(
            "/library",
            {
                "smiles": smiles,
                "name": name,
                "project": project,
                "tags": tags or [],
                "properties": properties or {},
                "notes": notes,
            },
        )

    def library_list(
        self,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        return self._get(
            "/library",
            project=project,
            tag=tag,
            search=search,
            limit=limit,
            offset=offset,
        )

    def library_get(self, compound_id: int) -> dict:
        return self._get(f"/library/{compound_id}")

    def library_update(
        self,
        compound_id: int,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if tags is not None:
            body["tags"] = tags
        if notes is not None:
            body["notes"] = notes
        if properties is not None:
            body["properties"] = properties
        return self._patch(f"/library/{compound_id}", body)

    def library_delete(self, compound_id: int) -> dict:
        return self._delete(f"/library/{compound_id}")

    def library_projects(self) -> dict:
        return self._get("/library/projects")
