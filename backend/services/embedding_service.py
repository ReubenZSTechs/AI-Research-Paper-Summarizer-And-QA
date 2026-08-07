import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

from backend.services.call_llm_service import CONFIG as BRIDGE_CONFIG
from backend.services.postgres_service import PostgresService


CONFIG = {
    "PARENT_FILE": Path("training/data/processed/chunks_parent.jsonl"),
    "CHILD_FILE": Path("training/data/processed/chunks_retrieval.jsonl"),
    "TXT_DIR": Path("training/data/processed/txt"),
    "SCOPE": "corpus",
    "EMBED_BATCH_SIZE": 96,
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "POSTGRES_PORT": int(os.getenv("POSTGRES_PORT", "5432")),
    "POSTGRES_DB": os.getenv("POSTGRES_DB"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
}


class EmbeddingClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=BRIDGE_CONFIG["MODEL_BRIDGE"]["embedding"], api_key="not-needed"
        )
        self.model = BRIDGE_CONFIG["SERVED_MODEL_NAME"]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def strip_null_bytes(text: str) -> str:
    return text.replace("\x00", "")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def hash_source_text(source: str) -> str:
    txt_path = CONFIG["TXT_DIR"] / f"{source}.txt"
    if txt_path.exists():
        return hashlib.sha256(txt_path.read_bytes()).hexdigest()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def resolve_document_id(db: PostgresService, source: str) -> str:
    sha256 = hash_source_text(source)
    existing = db.get_document_by_hash(sha256)
    if existing:
        return existing["document_id"]

    return db.insert_document(
        scope=CONFIG["SCOPE"],
        sha256=sha256,
        title=source,
        source_uri=str(CONFIG["TXT_DIR"] / f"{source}.txt"),
    )


def load_parents(db: PostgresService, parents: list[dict], document_ids: dict[str, str]) -> None:
    by_source = defaultdict(list)
    for record in parents:
        by_source[record["source"]].append(record)

    for source, records in tqdm(by_source.items(), desc="parents"):
        document_id = document_ids[source]
        payload = [
            (
                record["parent_id"],
                document_id,
                record["parent_index"],
                strip_null_bytes(record["parent_text"]),
            )
            for record in records
        ]
        db.insert_parent_chunk_batch(payload)


def load_children(db: PostgresService, embedder: EmbeddingClient, children: list[dict],
                   document_ids: dict[str, str]) -> None:
    batch_size = CONFIG["EMBED_BATCH_SIZE"]
    progress = tqdm(total=len(children), desc="children")

    for start in range(0, len(children), batch_size):
        batch = children[start:start + batch_size]
        texts = [strip_null_bytes(record["chunk_text"]) for record in batch]
        vectors = embedder.embed_batch(texts)

        payload = [
            (
                document_ids[record["source"]],
                record["parent_id"],
                record["ordinal"],
                text,
                record["token_count"],
                vector,
            )
            for record, text, vector in zip(batch, texts, vectors)
        ]

        db.insert_chunk_embedding_batch(payload)
        progress.update(len(batch))

    progress.close()


def main() -> None:
    db = PostgresService(
        host=CONFIG["POSTGRES_HOST"],
        port=CONFIG["POSTGRES_PORT"],
        dbname=CONFIG["POSTGRES_DB"],
        user=CONFIG["POSTGRES_USER"],
        password=CONFIG["POSTGRES_PASSWORD"],
    )
    embedder = EmbeddingClient()

    parents = load_jsonl(CONFIG["PARENT_FILE"])
    children = load_jsonl(CONFIG["CHILD_FILE"])

    sources = sorted({record["source"] for record in parents})
    document_ids = {}
    for source in tqdm(sources, desc="documents"):
        document_ids[source] = resolve_document_id(db, source)

    load_parents(db, parents, document_ids)
    load_children(db, embedder, children, document_ids)

    print(f"Documents: {len(sources)}")
    print(f"Parents:   {len(parents)}")
    print(f"Children:  {len(children)}")


if __name__ == "__main__":
    main()