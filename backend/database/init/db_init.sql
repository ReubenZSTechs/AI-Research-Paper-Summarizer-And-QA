CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() for user ids

CREATE TABLE users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT NOT NULL UNIQUE,
    email         TEXT UNIQUE,
    password_hash TEXT NOT NULL,  --Hashed Version
    created_at    TIMESTAMPTZ DEFAULT now(),
    last_login_at TIMESTAMPTZ
);


CREATE TABLE chunk_embeddings (
    chunk_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     VECTOR(1024) NOT NULL,   -- BGE-M3 dense dim
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunk_embeddings (source);


CREATE TABLE chat_history (
    id            BIGSERIAL PRIMARY KEY,
    user_id       UUID REFERENCES users(user_id) ON DELETE CASCADE,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content       TEXT NOT NULL,
    embedding     VECTOR(1024),
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON chat_history USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chat_history (session_id, created_at);
CREATE INDEX ON chat_history (user_id, created_at);