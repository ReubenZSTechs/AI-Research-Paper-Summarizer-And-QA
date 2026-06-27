import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from pathlib import Path
from tqdm import tqdm
import os
import re


CONFIG = {
    'TXT_FILEPATH': Path("training/datasets/processed/txt")
}

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
total_chunks = 0

for root, dirs, txt_files in tqdm(os.walk(CONFIG['TXT_FILEPATH']), desc="Total files"):
    for file in txt_files:
        full_txt_path = os.path.join(root, file)

        txt_loader = TextLoader(full_txt_path, encoding='utf-8')
        content = txt_loader.load()

        chunks = splitter.split_documents(content)

        total_chunks += len(chunks)

        with open("training/datasets/processed/chunks.jsonl", "a", encoding='utf-8') as f:
            for i, chunk in tqdm(enumerate(chunks), desc=f"Total chunks"):
                data_payload = {
                    "chunk_id": i,
                    "source": chunk.metadata.get("source", "unknown"),
                    "chunk_text": clean_text(chunk.page_content)
                }
                f.write(json.dumps(data_payload, ensure_ascii=False) + "\n")


print(f"Got {total_chunks} chunks")