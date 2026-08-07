CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT NOT NULL UNIQUE,
    email         TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now(),
    last_login_at TIMESTAMPTZ
);

CREATE TABLE documents (
    document_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(user_id) ON DELETE CASCADE,
    session_id    TEXT,
    scope         TEXT NOT NULL CHECK (scope IN ('corpus', 'user')),
    title         TEXT,
    source_uri    TEXT,
    sha256        TEXT NOT NULL,
    num_pages     INTEGER,
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, sha256)
);

CREATE TABLE chunk_embeddings (
    chunk_id      BIGSERIAL PRIMARY KEY,
    document_id   UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    parent_id     TEXT NOT NULL,
    ordinal       INTEGER NOT NULL,
    section       TEXT,
    page_start    INTEGER,
    page_end      INTEGER,
    chunk_text    TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(1024) NOT NULL,
    lexeme        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (document_id, ordinal)
);

CREATE INDEX ON chunk_embeddings (document_id);
CREATE INDEX ON chunk_embeddings USING gin (lexeme);
CREATE INDEX ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE TABLE parent_chunks (
    parent_id     TEXT PRIMARY KEY,
    document_id   UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    parent_index  INTEGER NOT NULL,
    parent_text   TEXT NOT NULL
);

CREATE TABLE qa_embeddings (
    qa_id               BIGSERIAL PRIMARY KEY,
    document_id         UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    user_id             UUID REFERENCES users(user_id) ON DELETE CASCADE,
    session_id          TEXT NOT NULL,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    question_embedding  VECTOR(1024) NOT NULL,
    grounded_chunk_ids  BIGINT[],
    feedback            SMALLINT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON qa_embeddings (session_id, created_at);
CREATE INDEX ON qa_embeddings USING hnsw (question_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

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