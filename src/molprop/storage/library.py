"""
Compound Library — persistent SQLite-backed storage for molecules.

Schema:
    compounds(id, smiles, name, project, tags, properties_json, notes,
              created_at, updated_at)

This is a lightweight stdlib-only solution; no SQLAlchemy required.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CompoundLibrary:
    """
    Thin sqlite3 wrapper offering CRUD operations on a compound table.

    Thread-safe via a process-level RLock; suitable for FastAPI single-worker
    deployments or low-concurrency multi-worker setups.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── internal ──────────────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with DB_LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS compounds (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    smiles          TEXT NOT NULL,
                    name            TEXT,
                    project         TEXT DEFAULT 'default',
                    tags            TEXT DEFAULT '[]',
                    properties_json TEXT DEFAULT '{}',
                    notes           TEXT,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    UNIQUE(smiles, project)
                );

                CREATE INDEX IF NOT EXISTS idx_project ON compounds(project);
                CREATE INDEX IF NOT EXISTS idx_smiles ON compounds(smiles);
                """
            )

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def add(
        self,
        smiles: str,
        name: str | None = None,
        project: str = "default",
        tags: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update a compound (idempotent on (smiles, project))."""
        ts = _now()
        tags_json = json.dumps(tags or [])
        props_json = json.dumps(properties or {})
        with DB_LOCK, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO compounds (smiles, name, project, tags, properties_json, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(smiles, project) DO UPDATE SET
                    name            = COALESCE(excluded.name, compounds.name),
                    tags            = excluded.tags,
                    properties_json = excluded.properties_json,
                    notes           = COALESCE(excluded.notes, compounds.notes),
                    updated_at      = excluded.updated_at
                RETURNING id;
                """,
                (smiles, name, project, tags_json, props_json, notes, ts, ts),
            )
            row = cur.fetchone()
            return self.get(row["id"])

    def get(self, compound_id: int) -> dict[str, Any] | None:
        with DB_LOCK, self._connect() as conn:
            row = conn.execute("SELECT * FROM compounds WHERE id = ?", (compound_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def list(
        self,
        project: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if search:
            clauses.append("(smiles LIKE ? OR name LIKE ? OR notes LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        # Tag filter is post-query JSON match (fine for small libraries)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        # Safe: `where` is built from constant clauses; user input is parameterised.
        sql = f"SELECT * FROM compounds {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?"  # noqa: S608
        params.extend([limit, offset])
        with DB_LOCK, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        results = [self._row_to_dict(r) for r in rows]
        if tag:
            results = [r for r in results if tag in r.get("tags", [])]
        return results

    def update(
        self,
        compound_id: int,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get(compound_id)
        if not existing:
            return None
        new_name = name if name is not None else existing["name"]
        new_tags = json.dumps(tags) if tags is not None else json.dumps(existing["tags"])
        new_notes = notes if notes is not None else existing["notes"]
        new_props = (
            json.dumps(properties) if properties is not None else json.dumps(existing["properties"])
        )
        with DB_LOCK, self._connect() as conn:
            conn.execute(
                """
                UPDATE compounds
                SET name = ?, tags = ?, notes = ?, properties_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_name, new_tags, new_notes, new_props, _now(), compound_id),
            )
        return self.get(compound_id)

    def delete(self, compound_id: int) -> bool:
        with DB_LOCK, self._connect() as conn:
            cur = conn.execute("DELETE FROM compounds WHERE id = ?", (compound_id,))
            return cur.rowcount > 0

    def projects(self) -> list[str]:
        with DB_LOCK, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT project FROM compounds ORDER BY project"
            ).fetchall()
        return [r["project"] for r in rows]

    def all_tags(self) -> list[str]:
        with DB_LOCK, self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT tags FROM compounds").fetchall()
        seen: set[str] = set()
        for r in rows:
            try:
                for t in json.loads(r["tags"] or "[]"):
                    seen.add(t)
            except json.JSONDecodeError:
                continue
        return sorted(seen)

    def count(self, project: str | None = None) -> int:
        with DB_LOCK, self._connect() as conn:
            if project:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM compounds WHERE project = ?", (project,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS c FROM compounds").fetchone()
        return int(row["c"])

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "smiles": row["smiles"],
            "name": row["name"],
            "project": row["project"],
            "tags": json.loads(row["tags"] or "[]"),
            "properties": json.loads(row["properties_json"] or "{}"),
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


# ── module-level singleton ─────────────────────────────────────────────────────
_DEFAULT_DB = Path(__file__).resolve().parents[3] / "data" / "library.db"
library = CompoundLibrary(_DEFAULT_DB)
