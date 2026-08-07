import warnings
warnings.filterwarnings(action='ignore')

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List, Optional, Tuple
from collections import defaultdict
from math import ceil
from time import sleep
import fitz
import requests
import re

import json
import os
from tqdm import tqdm

from backend.services.call_llm_service import Agent


CONFIG = {
    'ENTITY_EXTRACTOR_CONFIG': "training/configs/agents/data_generation_agent/entity_extractor.yaml",
    'PAPER_CLASSIFIER_CONFIG': "training/configs/agents/data_generation_agent/paper_classifier.yaml",
    'CONFIDENCE_THRESHOLD': 0.65,
    'ENTITY_THRESHOLD': 0.65,
    "DATA_SAVE_FILEPATH": "training/data/processed",
    'NUM_DOCUMENTS': 500,
    'MONTH_START': (2024, 1),
    'MONTH_END': (2026, 6),
    'PER_MONTH_QUOTA': None,
    'LOG_FILEPATH': "training/logs/dataset_generation_logs.jsonl",
    'METADATA_FILEPATH': "training/data/processed/filtered_results.json",
    'CHECKPOINT_FILEPATH': "training/checkpoint/arxiv_id_check.txt",
    'KEYWORD_PREFILTER': True,
    'DOWNLOAD_DELAY': 3.0,
    'REQUEST_RETRIES': 2,
}

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_rag": {"type": "boolean"},
        "is_rl": {"type": "boolean"},
        "is_agentic_workflow": {"type": "boolean"},
        "is_kg": {"type": "boolean"},
        "confidence": {
            "type": "object",
            "properties": {
                "rag": {"type": "number"},
                "rl": {"type": "number"},
                "agentic_workflow": {"type": "number"},
                "kg": {"type": "number"},
            },
            "required": ["rag", "rl", "agentic_workflow", "kg"],
        },
    },
    "required": ["is_rag", "is_rl", "is_agentic_workflow", "is_kg", "confidence"],
}

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "confidence"],
            },
        }
    },
    "required": ["entities"],
}

EMPTY_CONFIDENCE = {"rag": 0.0, "rl": 0.0, "agentic_workflow": 0.0, "kg": 0.0}

AI_RELATED_CATEGORIES = {
    "csai", "cslg", "cscl", "cscv", "csir", "csro", "csma", "csse",
    "csdb", "csdc", "cshc", "csne", "cssy", "statml", "eessas", "eesssp",
}


class PDFState(TypedDict):
    keywords: List[str]
    keyword_match: bool
    error_message: Optional[str]

    abstract: str

    is_rag: Optional[bool]
    is_rl: Optional[bool]
    is_agentic_workflow: Optional[bool]
    is_kg: Optional[bool]

    confidence: Optional[Dict[str, float]]
    category: Optional[str]
    decision_source: Optional[str]

    entities: List[str]


class AgentManager:
    def __init__(self):
        self.classifier = Agent(yaml_config=CONFIG['PAPER_CLASSIFIER_CONFIG'])
        self.extractor = Agent(yaml_config=CONFIG['ENTITY_EXTRACTOR_CONFIG'])

    def count_matches(self, terms: list[str], keyword_arr: list) -> int:
        return sum(
            1
            for keyword in keyword_arr
            for term in terms
            if term in keyword
        )

    def call_with_schema(self, agent: Agent, prompt: str, schema: dict) -> Optional[dict]:
        for _ in range(CONFIG['REQUEST_RETRIES'] + 1):
            try:
                raw = agent.generate_response(
                    prompt,
                    extra_args={"structured_outputs": {"json": schema}},
                )
            except Exception:
                continue

            if not raw:
                continue

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue

        return None

    def call_classifier_model(self, abstract: str):
        prompt = f"""
            Analyze the given abstract and classify the paper

            ABSTRACT INPUT:
            \"\"\"
            {abstract}
            \"\"\"
        """

        result = self.call_with_schema(self.classifier, prompt, CLASSIFIER_SCHEMA)

        if result is None:
            return {
                'error_message': "Classifier returned no parseable JSON",
                'is_rag': False,
                'is_rl': False,
                'is_agentic_workflow': False,
                'is_kg': False,
                'confidence': dict(EMPTY_CONFIDENCE),
            }

        return {
            'error_message': "",
            'is_rag': bool(result.get('is_rag', False)),
            'is_rl': bool(result.get('is_rl', False)),
            'is_agentic_workflow': bool(result.get('is_agentic_workflow', False)),
            'is_kg': bool(result.get("is_kg", False)),
            'confidence': result.get('confidence', dict(EMPTY_CONFIDENCE)),
        }

    def call_extractor_model(self, abstract: str):
        prompt = f"""
            Analyze the given abstract and extract the relevant entities

            ABSTRACT INPUT:
            \"\"\"
            {abstract}
            \"\"\"
        """

        result = self.call_with_schema(self.extractor, prompt, ENTITY_SCHEMA)

        if result is None:
            return {
                'error_message': "Extractor returned no parseable JSON",
                'entities': [],
            }

        return {
            'error_message': "",
            'entities': result.get("entities", []),
        }

    def keyword_filter_node(self, state: PDFState):
        keyword_arr = [
            str(k).lower().strip()
            for k in state.get("keywords", [])
            if k
        ]

        rag_terms = [
            "retrieval augmented generation",
            "retrieval-augmented generation",
            "rag",
            "graphrag",
        ]

        retrieval_terms = [
            "retrieval", "retriever", "dense retrieval", "sparse retrieval",
            "hybrid retrieval", "vector search", "semantic search",
            "document retrieval", "knowledge retrieval", "bm25", "faiss",
        ]

        generation_terms = [
            "llm", "large language model", "language model",
            "generative model", "transformer", "text generation",
        ]

        rl_terms = [
            "reinforcement learning", "rl", "q-learning", "deep q network",
            "dqn", "policy gradient", "ppo", "trpo", "a2c", "a3c", "sac",
            "actor-critic", "reward function", "markov decision process",
            "mdp", "agent environment interaction",
        ]

        agentic_terms = [
            "langgraph", "graph-based agent", "agent workflow", "multi-agent",
            "planner executor", "tool calling", "tool-use",
            "workflow orchestration", "stateful workflow", "state machine",
            "agent orchestration", "llm orchestration", "autonomous agent",
            "reasoning pipeline",
        ]

        kg_terms = [
            "knowledge graph", "entity relation graph", "semantic graph",
            "ontology", "rdf", "triple store", "knowledge base graph",
            "graph database", "neo4j",
        ]

        rag_score = self.count_matches(rag_terms, keyword_arr)

        if (
            self.count_matches(retrieval_terms, keyword_arr) > 0
            and self.count_matches(generation_terms, keyword_arr) > 0
        ):
            rag_score += 2

        scores = {
            "rag": rag_score,
            "rl": self.count_matches(rl_terms, keyword_arr),
            "agentic_workflow": self.count_matches(agentic_terms, keyword_arr),
            "kg": self.count_matches(kg_terms, keyword_arr),
        }

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score <= 0:
            return {
                "keyword_match": False,
                "category": None,
                "decision_source": "keyword_prefilter",
                "confidence": dict(EMPTY_CONFIDENCE),
            }

        confidence = dict(EMPTY_CONFIDENCE)
        confidence[best_category] = min(1.0, 0.5 + (best_score * 0.1))

        return {
            "keyword_match": True,
            "category": best_category,
            "decision_source": "keyword",
            "confidence": confidence,
        }

    def classify_paper_llm(self, state: PDFState):
        result = self.call_classifier_model(abstract=state['abstract'])

        return {
            'error_message': result['error_message'],
            'is_rag': result['is_rag'],
            'is_rl': result['is_rl'],
            'is_agentic_workflow': result['is_agentic_workflow'],
            'is_kg': result['is_kg'],
            'confidence': result['confidence'],
            'decision_source': "LLM",
        }

    def extract_entities_llm(self, state: PDFState):
        result = self.call_extractor_model(abstract=state['abstract'])

        accepted_entities = [
            entity.get('text', "")
            for entity in result['entities']
            if entity.get('confidence', 0.0) >= CONFIG['ENTITY_THRESHOLD']
            and entity.get('text')
        ]

        return {
            'error_message': result['error_message'],
            'entities': accepted_entities,
        }

    def accept_node(self, state: PDFState):
        return {
            'is_rag': state['is_rag'],
            'is_rl': state['is_rl'],
            'is_agentic_workflow': state['is_agentic_workflow'],
            'is_kg': state['is_kg'],
            'confidence': state['confidence'],
            'entities': state['entities'],
            'decision_source': state.get('decision_source', "unknown"),
        }

    def reject_node(self, state: PDFState):
        return {
            'is_rag': False,
            'is_rl': False,
            'is_agentic_workflow': False,
            'is_kg': False,
            'confidence': state.get('confidence', dict(EMPTY_CONFIDENCE)),
            'entities': state.get("entities", []),
            'decision_source': state.get('decision_source', "below_confidence_threshold"),
        }

    def route_after_keyword(self, state: PDFState):
        if CONFIG['KEYWORD_PREFILTER'] and not state.get("keyword_match", False):
            return "reject"
        return "classify_using_llm"

    def route_after_llm(self, state: PDFState):
        confidence_ref = state.get('confidence', {})

        max_conf = max(
            confidence_ref.get("rag", 0),
            confidence_ref.get("rl", 0),
            confidence_ref.get("agentic_workflow", 0),
            confidence_ref.get("kg", 0),
        )

        if max_conf < CONFIG['CONFIDENCE_THRESHOLD']:
            return "reject"

        if state['is_agentic_workflow'] or state['is_kg'] or state['is_rag'] or state['is_rl']:
            return "accept"

        return "reject"


def build_graph(state: PDFState, agent_manager: AgentManager):
    builder = StateGraph(state_schema=state)

    builder.add_node("keyword_filter_node", agent_manager.keyword_filter_node)
    builder.add_node("llm_classify_node", agent_manager.classify_paper_llm)
    builder.add_node("llm_extract_node", agent_manager.extract_entities_llm)
    builder.add_node("accept_node", agent_manager.accept_node)
    builder.add_node("reject_node", agent_manager.reject_node)

    builder.set_entry_point("keyword_filter_node")

    builder.add_conditional_edges(
        'keyword_filter_node',
        agent_manager.route_after_keyword,
        {
            'reject': 'reject_node',
            'classify_using_llm': 'llm_classify_node',
        }
    )

    builder.add_edge('llm_classify_node', 'llm_extract_node')

    builder.add_conditional_edges(
        'llm_extract_node',
        agent_manager.route_after_llm,
        {
            'accept': 'accept_node',
            'reject': 'reject_node',
        }
    )

    builder.add_edge("accept_node", END)
    builder.add_edge('reject_node', END)

    return builder.compile()


def enumerate_months(start: Tuple[int, int], end: Tuple[int, int]) -> List[str]:
    months = []
    year, month = start

    while (year, month) <= end:
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1

    return months


def resolve_month_quota(months: List[str]) -> int:
    if CONFIG['PER_MONTH_QUOTA']:
        return CONFIG['PER_MONTH_QUOTA']
    return ceil(CONFIG['NUM_DOCUMENTS'] / max(1, len(months)))


def extract_year_month(arXiv_id: str) -> Optional[Tuple[int, int]]:
    base = arXiv_id.split("v")[0]

    if "/" in base:
        return None

    date_number_id = base.split(".")[0]

    if len(date_number_id) < 4 or not date_number_id[:4].isdigit():
        return None

    year = int(date_number_id[:2])
    month = int(date_number_id[2:4])

    if month < 1 or month > 12:
        return None

    if year >= 91:
        year += 1900
    else:
        year += 2000

    return year, month


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def prepare_directories() -> None:
    for path in (
        f"{CONFIG['DATA_SAVE_FILEPATH']}/pdf",
        f"{CONFIG['DATA_SAVE_FILEPATH']}/txt",
    ):
        os.makedirs(path, exist_ok=True)

    for path in (CONFIG['LOG_FILEPATH'], CONFIG['CHECKPOINT_FILEPATH']):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path, "w", encoding='utf-8').close()


def strip_null_bytes(text: str) -> str:
    return text.replace("\x00", "")


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(filename=pdf_path)
        text = [page.get_text() for page in doc]
        doc.close()
        return strip_null_bytes("\n".join(text))

    except Exception as e:
        tqdm.write(f"PDF extraction failed for {pdf_path}: {e}")
        return ""


def save_text_file(arxiv_id: str, text: str) -> str:
    txt_path = os.path.join(f"{CONFIG['DATA_SAVE_FILEPATH']}/txt", f"{arxiv_id}.txt")

    with open(txt_path, "w", encoding='utf-8') as f:
        f.write(text)

    return txt_path


def log_selection(file_path: str, payload: dict) -> None:
    with open(file_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(payload) + "\n")


def attach_metadata(jsonl_path: str, json_path: str) -> None:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_existing_records(file_path: str) -> list[dict]:
    records = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rebuild_month_counts(records: list[dict]) -> Dict[str, int]:
    counts = defaultdict(int)

    for record in records:
        parsed = extract_year_month(record.get("id", ""))
        if parsed:
            counts[month_key(*parsed)] += 1

    return counts


def download_arxiv_pdf(arXiv_id: str) -> Optional[str]:
    output_path = os.path.join(f"{CONFIG['DATA_SAVE_FILEPATH']}/pdf", f"{arXiv_id}.pdf")
    pdf_url = f"https://arxiv.org/pdf/{arXiv_id}.pdf"

    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

    except Exception as e:
        tqdm.write(f"Download failed for {arXiv_id}: {e}")
        return None

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def add_to_checkpoint(arXiv_id: str) -> None:
    with open(CONFIG['CHECKPOINT_FILEPATH'], "a", encoding='utf-8') as f:
        f.write(f"{arXiv_id}\n")


def load_checkpoint_ids() -> set[str]:
    with open(CONFIG['CHECKPOINT_FILEPATH'], "r", encoding='utf-8') as f:
        return set(paper_id.strip() for paper_id in f if paper_id.strip())


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_ai_related(categories: list[str]) -> bool:
    return any(category in AI_RELATED_CATEGORIES for category in categories)


def report_distribution(months: List[str], counts: Dict[str, int], quota: int) -> None:
    print()
    print(f"{'month':<10}{'accepted':>10}{'quota':>8}")
    print("-" * 28)

    for key in months:
        marker = "" if counts.get(key, 0) < quota else "  full"
        print(f"{key:<10}{counts.get(key, 0):>10}{quota:>8}{marker}")

    print("-" * 28)
    print(f"{'TOTAL':<10}{sum(counts.values()):>10}")


if __name__ == "__main__":
    prepare_directories()

    months = enumerate_months(CONFIG['MONTH_START'], CONFIG['MONTH_END'])
    month_quota = resolve_month_quota(months)
    valid_months = set(months)

    agent_manager = AgentManager()
    graph = build_graph(PDFState, agent_manager)

    results_arr = load_existing_records(CONFIG['LOG_FILEPATH'])
    month_counts = rebuild_month_counts(results_arr)
    processed_ids = load_checkpoint_ids()
    processed = 0

    print(f"Target documents:   {CONFIG['NUM_DOCUMENTS']}")
    print(f"Month window:       {months[0]} to {months[-1]} ({len(months)} months)")
    print(f"Per-month quota:    {month_quota}")
    print(f"Theoretical max:    {month_quota * len(months)}")
    print(f"Already collected:  {len(results_arr)}")
    print()

    with open(os.getenv("DATASET_RAW_FILEPATH")) as f:
        iterator = tqdm(f, desc="Processing arXiv")

        for line in iterator:
            if CONFIG['NUM_DOCUMENTS'] and len(results_arr) >= CONFIG['NUM_DOCUMENTS']:
                break

            processed += 1

            iterator.set_postfix({
                'processed': processed,
                'accepted': len(results_arr),
                'months_full': sum(1 for m in months if month_counts.get(m, 0) >= month_quota),
            })

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            arxiv_id = paper.get("id", "")

            if not arxiv_id or arxiv_id in processed_ids:
                continue

            parsed = extract_year_month(arXiv_id=arxiv_id)

            if parsed is None:
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            key = month_key(*parsed)

            if key not in valid_months:
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            if month_counts.get(key, 0) >= month_quota:
                continue

            title = clean_text(paper.get('title', "").strip())
            abstract = clean_text(paper.get('abstract', "").strip())
            categories = paper.get('categories', "")

            category_keywords = [
                category.lower().replace(".", "")
                for category in categories.split()
            ]

            if not abstract:
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            title_keywords = [word.lower() for word in title.split()]
            keywords = title_keywords + category_keywords

            if not is_ai_related(categories=keywords):
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            result = graph.invoke({
                'keywords': keywords,
                'abstract': abstract,
            })

            is_relevant = (
                result.get("is_rag")
                or result.get("is_rl")
                or result.get("is_agentic_workflow")
                or result.get("is_kg")
            )
            confidence_ref = result.get("confidence", {})
            max_conf = max(
                confidence_ref.get("rag", 0),
                confidence_ref.get("rl", 0),
                confidence_ref.get("agentic_workflow", 0),
                confidence_ref.get("kg", 0),
            )

            labels = []
            if result.get("is_rag"):
                labels.append("RAG")
            if result.get("is_rl"):
                labels.append("RL")
            if result.get("is_agentic_workflow"):
                labels.append("Agentic Workflow")
            if result.get("is_kg"):
                labels.append("Knowledge Graph")

            label_str_format = ",".join(labels) if labels else "None"

            if not is_relevant or max_conf < CONFIG['CONFIDENCE_THRESHOLD']:
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            tqdm.write(
                f"ACCEPTED {arxiv_id} [{key}] [{label_str_format} | conf={max_conf:.2f}] "
                f"({month_counts.get(key, 0) + 1}/{month_quota}) {title[:60]}"
            )

            pdf_path = download_arxiv_pdf(arXiv_id=arxiv_id)

            if not pdf_path:
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            text = extract_text_from_pdf(pdf_path=pdf_path)

            if not text.strip():
                tqdm.write(f"Empty text extracted from {arxiv_id}")
                add_to_checkpoint(arXiv_id=arxiv_id)
                continue

            txt_path = save_text_file(arxiv_id=arxiv_id, text=text)

            payload_log = {
                'id': arxiv_id,
                'title': title,
                'categories': categories,
                'year': parsed[0],
                'month': parsed[1],
                'labels': labels,
                'confidence': confidence_ref,
                'entities': result.get("entities", []),
                'txt_path': txt_path,
            }

            log_selection(file_path=CONFIG['LOG_FILEPATH'], payload=payload_log)
            results_arr.append(payload_log)
            month_counts[key] += 1
            add_to_checkpoint(arXiv_id=arxiv_id)

            sleep(CONFIG['DOWNLOAD_DELAY'])

    attach_metadata(jsonl_path=CONFIG['LOG_FILEPATH'], json_path=CONFIG['METADATA_FILEPATH'])

    report_distribution(months, month_counts, month_quota)
    print(f"\nAbstracts scanned this run: {processed}")