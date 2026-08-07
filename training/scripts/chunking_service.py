import json
import re
from pathlib import Path

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from backend.services.call_llm_service import CONFIG as BRIDGE_CONFIG


CONFIG = {
    "TXT_DIR": Path("training/data/processed/txt"),
    "CHILD_OUT": Path("training/data/processed/chunks_retrieval.jsonl"),
    "PARENT_OUT": Path("training/data/processed/chunks_parent.jsonl"),
    "CHILD_TOKENS": 512,
    "CHILD_OVERLAP": 64,
    "PARENT_TOKENS": 1600,
    "MIN_CHILD_CHARS": 200,
}

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " "]
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
BACK_MATTER = re.compile(
    r"\n\s*(references|bibliography|acknowledg(e)?ments)\s*\n", re.IGNORECASE
)


class RemoteTokenizer:
    def __init__(self, bridge_url: str, served_model_name: str):
        self.endpoint = bridge_url.rstrip("/").removesuffix("/v1") + "/tokenize"
        self.served_model_name = served_model_name
        self.session = requests.Session()
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        response = self.session.post(
            self.endpoint,
            json={"model": self.served_model_name, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()

        token_count = response.json()["count"]
        self._cache[text] = token_count
        return token_count


def normalise(text: str) -> str:
    text = text.replace("\x00", "")
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_back_matter(text: str) -> str:
    matches = list(BACK_MATTER.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    if cut < len(text) * 0.4:
        return text
    return text[:cut]


def build_splitter(tokenizer: RemoteTokenizer, chunk_tokens: int, overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap,
        length_function=tokenizer.count,
        separators=SEPARATORS,
        keep_separator=True,
    )


def main() -> None:
    tokenizer = RemoteTokenizer(
        bridge_url=BRIDGE_CONFIG["MODEL_BRIDGE"]["embedding"],
        served_model_name=BRIDGE_CONFIG["SERVED_MODEL_NAME"]["embedding"],
    )
    parent_splitter = build_splitter(tokenizer, CONFIG["PARENT_TOKENS"], 0)
    child_splitter = build_splitter(tokenizer, CONFIG["CHILD_TOKENS"], CONFIG["CHILD_OVERLAP"])

    CONFIG["CHILD_OUT"].parent.mkdir(parents=True, exist_ok=True)
    files = sorted(CONFIG["TXT_DIR"].rglob("*.txt"))
    total_children = 0
    total_parents = 0

    with open(CONFIG["CHILD_OUT"], "w", encoding="utf-8") as child_handle, open(
        CONFIG["PARENT_OUT"], "w", encoding="utf-8"
    ) as parent_handle:
        for path in tqdm(files, desc="Documents"):
            document_id = path.stem
            raw = path.read_text(encoding="utf-8", errors="ignore")
            body = strip_back_matter(normalise(raw))

            if len(body) < CONFIG["MIN_CHILD_CHARS"]:
                continue

            ordinal = 0
            for parent_index, parent_text in enumerate(parent_splitter.split_text(body)):
                parent_id = f"{document_id}::p{parent_index}"
                parent_handle.write(
                    json.dumps(
                        {
                            "parent_id": parent_id,
                            "source": document_id,
                            "parent_index": parent_index,
                            "parent_text": parent_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_parents += 1

                for child_text in child_splitter.split_text(parent_text):
                    if len(child_text) < CONFIG["MIN_CHILD_CHARS"]:
                        continue
                    child_handle.write(
                        json.dumps(
                            {
                                "chunk_id": f"{document_id}::c{ordinal}",
                                "parent_id": parent_id,
                                "source": document_id,
                                "ordinal": ordinal,
                                "chunk_text": child_text.strip(),
                                "token_count": tokenizer.count(child_text),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    ordinal += 1
                    total_children += 1

    print(f"Documents: {len(files)}")
    print(f"Parents:   {total_parents}")
    print(f"Children:  {total_children}")


if __name__ == "__main__":
    main()