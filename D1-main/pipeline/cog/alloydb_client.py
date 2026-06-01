"""
alloydb_client.py
─────────────────
Drop-in replacement for ChromaDB using AlloyDB (PostgreSQL + pgvector).

Provides the same logical operations (collections, sentinel records,
chunk storage, cross-collection queries) but backed by a single
AlloyDB `embeddings` table with a `collection` column for namespacing.

Schema (created by setup_alloydb.sql or auto-created here):
    embeddings (
        unique_id   TEXT PRIMARY KEY,
        sub_id      TEXT,
        collection  TEXT NOT NULL,
        text        TEXT,
        embedding   vector(3072),
        metadata    JSONB DEFAULT '{}'
    )

Environment variables (same as a2.py):
    ALLOYDB_PASSWORD  — required
    ALLOYDB_HOST      — required
    ALLOYDB_USER      — default 'postgres'
    ALLOYDB_DB        — default 'postgres'
"""

import json
import os
import urllib.parse
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras

# ── Load .env if present ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not needed on Cloud Run

# ── Connection setup ─────────────────────────────────────────────────────────

_user = os.getenv("ALLOYDB_USER", "postgres")
_db   = os.getenv("ALLOYDB_DB", "postgres")

# The connection string and schema check are built lazily (on the first
# connection) rather than at import time, so that importing this module does
# NOT require AlloyDB credentials to be present — only actually using the
# database does. This keeps the module importable for tooling, tests, and
# partial pipeline runs (e.g. report-only) that never touch AlloyDB, and
# avoids opening a network connection just to import.
DATABASE_URL = None
_schema_ready = False


def _build_database_url() -> str:
    raw_password = os.getenv("ALLOYDB_PASSWORD")
    ip           = os.getenv("ALLOYDB_HOST")
    if not raw_password or not ip:
        raise EnvironmentError(
            "Missing ALLOYDB_PASSWORD or ALLOYDB_HOST in environment. "
            "Set them as environment variables."
        )
    encoded_password = urllib.parse.quote_plus(raw_password)
    return f"postgresql://{_user}:{encoded_password}@{ip}:5432/{_db}"


def _get_conn():
    """Get a new psycopg2 connection (autocommit off by default).

    The connection string is resolved on first use, and the schema is
    ensured exactly once (the first time a connection is opened).
    """
    global DATABASE_URL, _schema_ready
    if DATABASE_URL is None:
        DATABASE_URL = _build_database_url()
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    if not _schema_ready:
        # Set the flag before calling _ensure_schema() (which itself opens a
        # connection) to avoid re-entrant schema checks; reset on failure so
        # the next connection retries.
        _schema_ready = True
        try:
            _ensure_schema()
        except Exception as e:
            _schema_ready = False
            print(f"[ALLOYDB] Schema ensure deferred (will retry): {e}")
    return conn


# ── Ensure schema exists ─────────────────────────────────────────────────────

def _ensure_schema():
    """
    Create the embeddings table + indexes if they don't exist.
    If the table already exists (from the original a2.py setup), add
    the missing `collection` and `metadata` columns.
    """
    conn = _get_conn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Check if table already exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'embeddings'
                )
            """)
            table_exists = cur.fetchone()[0]

            if not table_exists:
                # Fresh install — create with all columns
                # gemini-embedding-001 outputs 3072-dimensional vectors
                cur.execute("""
                    CREATE TABLE embeddings (
                        unique_id   TEXT PRIMARY KEY,
                        sub_id      TEXT DEFAULT '',
                        collection  TEXT NOT NULL DEFAULT '',
                        text        TEXT DEFAULT '',
                        embedding   vector(3072),
                        metadata    JSONB DEFAULT '{}'
                    )
                """)
                print("[ALLOYDB] Created embeddings table")
            else:
                # Table exists — add missing columns if needed
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'embeddings'
                """)
                existing_cols = {row[0] for row in cur.fetchall()}

                if 'collection' not in existing_cols:
                    cur.execute("""
                        ALTER TABLE embeddings
                        ADD COLUMN collection TEXT NOT NULL DEFAULT ''
                    """)
                    print("[ALLOYDB] Added 'collection' column")

                if 'metadata' not in existing_cols:
                    cur.execute("""
                        ALTER TABLE embeddings
                        ADD COLUMN metadata JSONB DEFAULT '{}'
                    """)
                    print("[ALLOYDB] Added 'metadata' column")

            # Create indexes (IF NOT EXISTS handles idempotency)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_collection
                ON embeddings (collection)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_filename
                ON embeddings ((metadata->>'filename'))
            """)

            # Drop old HNSW index if it exists (HNSW has 2000-dim limit,
            # gemini-embedding-001 outputs 3072 dims)
            cur.execute("DROP INDEX IF EXISTS idx_embeddings_hnsw")

            # IVFFlat supports >2000 dims but requires rows to exist
            # for cluster computation. Skip gracefully if table is empty.
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_ivfflat
                    ON embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except Exception as e:
                print(f"[ALLOYDB] IVFFlat index skipped (will retry when data exists): {e}")
    finally:
        conn.close()

    print("[ALLOYDB] Schema verified / migrated")


# Schema is ensured lazily on the first connection (see _get_conn), so there
# is no module-level _ensure_schema() call here — importing this module no
# longer opens a database connection.


# ─────────────────────────────────────────────────────────────────────────────
# Collection abstraction
# ─────────────────────────────────────────────────────────────────────────────

class AlloyDBCollection:
    """
    Mimics the ChromaDB collection interface but operates on the shared
    `embeddings` table, filtered by a `collection` column.
    """

    def __init__(self, name: str):
        self.name = name

    # ── add ───────────────────────────────────────────────────────────────

    def add(
        self,
        ids:        List[str],
        documents:  List[str],
        embeddings: List[List[float]],
        metadatas:  List[dict],
    ):
        """Insert rows. Uses ON CONFLICT to upsert."""
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                for uid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
                    emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                    meta_json = json.dumps(meta)
                    cur.execute("""
                        INSERT INTO embeddings (unique_id, sub_id, collection, text, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                        ON CONFLICT (unique_id) DO UPDATE SET
                            sub_id     = EXCLUDED.sub_id,
                            collection = EXCLUDED.collection,
                            text       = EXCLUDED.text,
                            embedding  = EXCLUDED.embedding,
                            metadata   = EXCLUDED.metadata
                    """, (
                        uid,
                        meta.get("sub_id", ""),
                        self.name,
                        doc,
                        emb_str,
                        meta_json,
                    ))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── get ───────────────────────────────────────────────────────────────

    def get(
        self,
        ids:     Optional[List[str]] = None,
        where:   Optional[dict] = None,
        include: Optional[List[str]] = None,
    ) -> dict:
        """
        Retrieve rows by ids or by metadata filter.
        Returns ChromaDB-style dict: {ids, documents, metadatas, embeddings}.
        """
        include = include or []
        conn = _get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if ids is not None:
                    placeholders = ",".join(["%s"] * len(ids))
                    cur.execute(f"""
                        SELECT unique_id, text, embedding, metadata
                        FROM embeddings
                        WHERE collection = %s AND unique_id IN ({placeholders})
                    """, [self.name] + ids)
                elif where is not None:
                    sql, params = _build_where_clause(where)
                    cur.execute(f"""
                        SELECT unique_id, text, embedding, metadata
                        FROM embeddings
                        WHERE collection = %s AND {sql}
                    """, [self.name] + params)
                else:
                    cur.execute("""
                        SELECT unique_id, text, embedding, metadata
                        FROM embeddings
                        WHERE collection = %s
                    """, [self.name])

                rows = cur.fetchall()
        finally:
            conn.close()

        result = {
            "ids":        [r["unique_id"] for r in rows],
            "documents":  [r["text"] for r in rows] if "documents" in include else [],
            "metadatas":  [r["metadata"] for r in rows] if "metadatas" in include else [],
            "embeddings": [],
        }
        if "embeddings" in include:
            result["embeddings"] = [
                _parse_vector(r["embedding"]) for r in rows
            ]
        # If no include specified but ids requested, still populate metadatas
        # for compatibility with sentinel checks
        if not include:
            result["metadatas"] = [r["metadata"] for r in rows]

        return result

    # ── query (vector similarity) ─────────────────────────────────────────

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results:        int = 5,
        where:            Optional[dict] = None,
        include:          Optional[List[str]] = None,
    ) -> dict:
        """
        Vector similarity search using pgvector cosine distance (<=>).
        Returns ChromaDB-style nested lists: {ids: [[]], documents: [[]], ...}.
        """
        include = include or ["documents"]
        emb = query_embeddings[0]
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"

        conn = _get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                where_sql = "collection = %s"
                params = [self.name]

                if where:
                    extra_sql, extra_params = _build_where_clause(where)
                    where_sql += f" AND {extra_sql}"
                    params += extra_params

                cur.execute(f"""
                    SELECT unique_id, text, embedding, metadata,
                           embedding <=> %s::vector AS distance
                    FROM embeddings
                    WHERE {where_sql}
                    ORDER BY distance ASC
                    LIMIT %s
                """, [emb_str] + params + [n_results])

                rows = cur.fetchall()
        finally:
            conn.close()

        result = {
            "ids":       [[r["unique_id"] for r in rows]],
            "documents": [[r["text"] for r in rows]] if "documents" in include else [[]],
            "metadatas": [[r["metadata"] for r in rows]] if "metadatas" in include else [[]],
            "distances": [[r["distance"] for r in rows]] if "distances" in include else [[]],
        }
        return result

    # ── update ────────────────────────────────────────────────────────────

    def update(
        self,
        ids:       List[str],
        metadatas: List[dict],
    ):
        """Update metadata for existing rows."""
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                for uid, meta in zip(ids, metadatas):
                    cur.execute("""
                        UPDATE embeddings
                        SET metadata = %s::jsonb
                        WHERE unique_id = %s AND collection = %s
                    """, (json.dumps(meta), uid, self.name))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── delete ────────────────────────────────────────────────────────────

    def delete(
        self,
        ids:   Optional[List[str]] = None,
        where: Optional[dict] = None,
    ):
        """Delete rows by ids or metadata filter."""
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                if ids is not None:
                    placeholders = ",".join(["%s"] * len(ids))
                    cur.execute(f"""
                        DELETE FROM embeddings
                        WHERE collection = %s AND unique_id IN ({placeholders})
                    """, [self.name] + ids)
                elif where is not None:
                    sql, params = _build_where_clause(where)
                    cur.execute(f"""
                        DELETE FROM embeddings
                        WHERE collection = %s AND {sql}
                    """, [self.name] + params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# AlloyDB client (replaces chromadb.PersistentClient)
# ─────────────────────────────────────────────────────────────────────────────

class AlloyDBClient:
    """
    Mimics chromadb.PersistentClient interface.
    Collections are logical namespaces within the shared `embeddings` table.
    """

    def get_collection(self, name: str) -> AlloyDBCollection:
        """
        Get a collection by name.
        Raises ValueError if no rows exist for this collection name.
        """
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM embeddings WHERE collection = %s LIMIT 1",
                    (name,)
                )
                if not cur.fetchone():
                    raise ValueError(f"Collection '{name}' does not exist")
        finally:
            conn.close()
        return AlloyDBCollection(name)

    def create_collection(self, name: str, metadata: dict = None) -> AlloyDBCollection:
        """
        Create (or get) a collection.
        Since collections are just a column value, this is a no-op —
        the collection starts existing once rows are inserted.
        """
        print(f"[ALLOYDB] Collection ready: {name}")
        return AlloyDBCollection(name)

    def get_or_create_collection(self, name: str, metadata: dict = None) -> AlloyDBCollection:
        """Get or create a collection."""
        return AlloyDBCollection(name)

    def delete_collection(self, name: str):
        """Delete all rows belonging to a collection."""
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM embeddings WHERE collection = %s",
                    (name,)
                )
            conn.commit()
            print(f"[ALLOYDB] Deleted collection '{name}'")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_collections(self) -> List:
        """List all distinct collection names. Returns list of objects with .name attr."""
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT collection FROM embeddings WHERE collection LIKE 'patents_%'"
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        class _ColRef:
            def __init__(self, name):
                self.name = name
        return [_ColRef(r[0]) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_vector(val) -> List[float]:
    """Parse a pgvector string like '[0.1,0.2,...]' into a list of floats."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip("[]")
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _build_where_clause(where: dict) -> tuple:
    """
    Convert a ChromaDB-style where filter to SQL.

    Supports:
        {"filename": "foo.pdf"}                           → simple equality
        {"$and": [{"filename": {"$eq": "foo.pdf"}}, ...]} → AND compound
        {"filename": {"$eq": "foo.pdf"}}                  → operator form
        {"chunk_index": {"$gte": 0}}                      → comparison
    """
    if "$and" in where:
        clauses = []
        params = []
        for sub in where["$and"]:
            c, p = _build_where_clause(sub)
            clauses.append(c)
            params.extend(p)
        return "(" + " AND ".join(clauses) + ")", params

    if "$or" in where:
        clauses = []
        params = []
        for sub in where["$or"]:
            c, p = _build_where_clause(sub)
            clauses.append(c)
            params.extend(p)
        return "(" + " OR ".join(clauses) + ")", params

    # Single field condition
    for key, val in where.items():
        if isinstance(val, dict):
            # Operator form: {"field": {"$eq": value}}
            for op, operand in val.items():
                sql_op = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}.get(op, "=")
                # metadata is JSONB — use ->> for text, cast for numbers
                if isinstance(operand, (int, float)):
                    return f"(metadata->>'{key}')::float {sql_op} %s", [operand]
                return f"metadata->>'{key}' {sql_op} %s", [str(operand)]
        else:
            # Simple equality: {"field": "value"}
            return f"metadata->>'{key}' = %s", [str(val)]

    return "TRUE", []
