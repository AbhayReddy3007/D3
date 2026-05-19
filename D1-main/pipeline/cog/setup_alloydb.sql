-- setup_alloydb.sql
-- Creates or migrates the embeddings table for the patent analysis pipeline.
-- Safe to run multiple times (idempotent).

CREATE EXTENSION IF NOT EXISTS vector;

-- Create table if it doesn't exist at all
CREATE TABLE IF NOT EXISTS embeddings (
    unique_id   TEXT PRIMARY KEY,
    sub_id      TEXT DEFAULT '',
    text        TEXT DEFAULT '',
    embedding   vector(768)
);

-- Add columns that may be missing (from original a2.py schema)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'collection'
    ) THEN
        ALTER TABLE embeddings ADD COLUMN collection TEXT NOT NULL DEFAULT '';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'embeddings' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE embeddings ADD COLUMN metadata JSONB DEFAULT '{}';
    END IF;
END $$;

-- Indexes (all idempotent)
CREATE INDEX IF NOT EXISTS idx_embeddings_collection
    ON embeddings (collection);

CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_filename
    ON embeddings ((metadata->>'filename'));

CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_chunk_index
    ON embeddings ((metadata->>'chunk_index'));

CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops);
