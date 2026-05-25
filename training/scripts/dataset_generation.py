import warnings
warnings.filterwarnings(action='ignore')

from dotenv import load_dotenv

load_dotenv(".env")

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List, Optional
from ollama import chat
from datetime import datetime
from time import sleep
import fitz
import requests

import yaml
import json
import os
from tqdm import tqdm

CONFIG = {
    'ENTITY_EXTRACTOR_CONFIG': "training/configs/models/entity_extractor.yaml",
    'PAPER_CLASSIFIER_MODEL': "training/configs/models/paper_classifier_model.yaml",
    'CONFIDENCE_THRESHOLD': 0.95,
    'ENTITY_THRESHOLD': 0.9,
    "DATA_SAVE_FILEPATH": "training/datasets/processed",
    'NUM_DOCUMENTS': 150,
    'YEAR_FILTER': 2022,
    'LOG_FILEPATH': "training/logs/dataset_generation_logs.jsonl",
    'METADATA_FILEPATH': "training/dataset/processed/filtered_results.json",
    'CHECKPOINT_FILEPATH': "training/checkpoint/arxiv_id_check.txt",
}

with open(CONFIG['PAPER_CLASSIFIER_MODEL']) as f:
    classifier_config = yaml.safe_load(f)

with open(CONFIG['ENTITY_EXTRACTOR_CONFIG']) as f:
    entity_config = yaml.safe_load(f)


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
    decision_source: Optional[str]

    entities: List[str]


class AgentManager:
    def __init__(self):
        self.classifier_model = classifier_config['model']
        self.entity_model = entity_config['model']

        self.classifier_generation = classifier_config['generation']
        self.classifier_system_prompt = classifier_config['system_prompt']
        self.classifier_response_type = classifier_config['response_type']

        self.entity_generation = entity_config['generation']
        self.entity_system_prompt = entity_config['system_prompt']
        self.entity_response_type = entity_config['response_type']

    def count_matches(self, terms: list[str], keyword_arr: list) -> int:
        return sum(
            1
            for keyword in keyword_arr
            for term in terms
            if term in keyword
        )


    def call_classifier_model(self, abstract: str):
        prompt = f"""
            Analyze the given abstract and classify the paper

            ABSTRACT INPUT:
            \"\"\"
            {abstract}
            \"\"\"
        """

        response = chat(
            model=self.classifier_model,
            messages=[
                {
                    'role': 'system',
                    'content': self.classifier_system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options=self.classifier_generation,
            format=self.classifier_response_type
        )

        result = response['message']['content']

        try:
            result = json.loads(result)

        except Exception as e:
            return {
                'error_message': f"Result is not json. Got {type(result)}. Error: {e}",
                "is_rag": False,
                "is_rl": False,
                "is_agentic_workflow": False,
                'is_kg': False,
                "confidence": {"rag": 0.0, "rl": 0.0, "langgraph": 0.0},
            }
        
        return {
            'error_message': "",
            'is_rag': bool(result.get('is_rag', False)),
            'is_rl': bool(result.get('is_rl', False)),
            'is_agentic_workflow': bool(result.get('is_agentic_workflow', False)),
            'is_kg': bool(result.get("is_kg", False)),
            'confidence': result.get('confidence', {
                "rag": 0.0,
                "rl": 0.0,
                "agentic_workflow": 0.0,
                "kg": 0.0
            })
        }
    

    def call_extractor_model(self, abstract: str):
        prompt = f"""
            Analyze the given abstract and extract the relevant entities

            ABSTRACT INPUT:
            \"\"\"
            {abstract}
            \"\"\"
        """

        response = chat(
            model=self.classifier_model,
            messages=[
                {
                    'role': 'system',
                    'content': self.classifier_system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options=self.classifier_generation,
            format=self.classifier_response_type
        )

        result = response['message']['content']

        try:
            result = json.loads(result)

        except Exception as e:
            return {
                'error_message': f"Extraction failed. Error: {e}",
                'entities': result.get("entities", [])
            }
        
        return {
            'error_message': "",
            'entities': result.get("entities", [])
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
            "graphrag"
        ]

        retrieval_terms = [
            "retrieval",
            "retriever",
            "dense retrieval",
            "sparse retrieval",
            "hybrid retrieval",
            "vector search",
            "semantic search",
            "document retrieval",
            "knowledge retrieval",
            "bm25",
            "faiss"
        ]

        generation_terms = [
            "llm",
            "large language model",
            "language model",
            "generative model",
            "transformer",
            "text generation"
        ]

        rl_terms = [
            "reinforcement learning",
            "rl",
            "q-learning",
            "deep q network",
            "dqn",
            "policy gradient",
            "ppo",
            "trpo",
            "a2c",
            "a3c",
            "sac",
            "actor-critic",
            "reward function",
            "markov decision process",
            "mdp",
            "agent environment interaction"
        ]

        agentic_terms = [
            "langgraph",
            "graph-based agent",
            "agent workflow",
            "multi-agent",
            "planner executor",
            "tool calling",
            "tool-use",
            "workflow orchestration",
            "stateful workflow",
            "state machine",
            "agent orchestration",
            "llm orchestration",
            "autonomous agent",
            "reasoning pipeline"
        ]

        kg_terms = [
            "knowledge graph",
            "entity relation graph",
            "semantic graph",
            "ontology",
            "rdf",
            "triple store",
            "knowledge base graph",
            "graph database",
            "neo4j"
        ]

        rag_score = self.count_matches(rag_terms, keyword_arr)

        if (
            self.count_matches(retrieval_terms, keyword_arr) > 0
            and self.count_matches(generation_terms, keyword_arr) > 0
        ):
            rag_score += 2

        rl_score = self.count_matches(rl_terms, keyword_arr)

        agentic_score = self.count_matches(agentic_terms, keyword_arr)

        kg_score = self.count_matches(kg_terms, keyword_arr)

        scores = {
            "rag": rag_score,
            "rl": rl_score,
            "agentic_workflow": agentic_score,
            "kg": kg_score
        }

        best_category = max(scores, key=scores.get)

        best_score = scores[best_category]

        if best_score <= 0:
            return {
                "keyword_match": False,
                "decision_source": "keyword",
                "confidence": {
                    "rag": 0.0,
                    "rl": 0.0,
                    "agentic_workflow": 0.0,
                    "kg": 0.0
                }
            }

        confidence = {
            "rag": 0.0,
            "rl": 0.0,
            "agentic_workflow": 0.0,
            "kg": 0.0
        }

        confidence[best_category] = min(
            1.0,
            0.5 + (best_score * 0.1)
        )

        return {
            "keyword_match": True,
            "category": best_category,
            "decision_source": "keyword",
            "confidence": confidence,
        }
    

    def classify_paper_llm(self, state: PDFState):
        abstract = state['abstract']

        result = self.call_classifier_model(abstract=abstract)

        return {
            'is_rag': result['is_rag'],
            'is_rl': result['is_rl'],
            'is_agentic_workflow': result['is_agentic_workflow'],
            'is_kg': result['is_kg'],
            'confidence': result['confidence'],
            'decision_source': "LLM"
        }
    

    def extract_entities_llm(self, state: PDFState):
        abstract = state['abstract']

        result = self.call_extractor_model(abstract=abstract)

        entities_ref = result['entities']

        accepted_entities = [entity['text'] for entity in entities_ref if entity['confidence'] >= CONFIG['ENTITY_THRESHOLD']]

        return {
            'error_message': result['error_message'],
            'entities': accepted_entities
        }
    

    def accept_node(self, state: PDFState):
        return {
            'is_rag': state['is_rag'],
            'is_rl': state['is_rl'],
            'is_agentic_workflow': state['is_agentic_workflow'],
            'is_kg': state['is_kg'],
            'confidence': state['confidence'],
            'entities': state['entities'],
            'decision_source': state.get('decision_source', "unknown")
        }
    

    def reject_node(self, state: PDFState):
        return {
            'is_rag': False,
            'is_rl': False,
            'is_agentic_workflow': False,
            'is_kg': False,
            'confidence': state.get('confidence', {
                "rag": 0.0,
                "rl": 0.0,
                "agentic_workflow": 0.0,
                "kg": 0.0
            }),
            'entities': state.get("entities", []),
            'decision_source': state.get('decision_source', "Not exceeded confidence threshold")
        }
    

    def route_after_keyword(self, state: PDFState):
        return "classify_using_llm"
    

    def route_after_llm(self, state: PDFState):
        confidence_ref = state.get('confidence', {})

        max_conf = max(confidence_ref.get("rag", 0), confidence_ref.get("rl", 0), confidence_ref.get("agentic_workflow", 0), confidence_ref.get("kg", 0))

        if max_conf < CONFIG['CONFIDENCE_THRESHOLD']:
            return "reject"

        if state['is_agentic_workflow'] or state['is_kg'] or state['is_rag'] or state['is_rl']:
            return "accept"
        
        return "reject"
    

def build_graph(state: PDFState, AgentManager: AgentManager):
    builder = StateGraph(state_schema=state)

    builder.add_node("keyword_filter_node", AgentManager.keyword_filter_node)
    builder.add_node("llm_classify_node", AgentManager.classify_paper_llm)
    builder.add_node("llm_extract_node", AgentManager.extract_entities_llm)
    builder.add_node("accept_node", AgentManager.accept_node)
    builder.add_node('reject_node', AgentManager.reject_node)

    builder.set_entry_point("keyword_filter_node")
    
    builder.add_conditional_edges(
        'keyword_filter_node',
        AgentManager.route_after_keyword,
        {
            'accept': 'accept_node',
            'classify_using_llm': 'llm_classify_node'
        }
    )

    builder.add_edge('llm_classify_node', 'llm_extract_node')

    builder.add_conditional_edges(
        'llm_extract_node',
        AgentManager.route_after_llm,
        {
            'accept': 'accept_node',
            'reject': 'reject_node'
        }
    )

    builder.add_edge("accept_node", END)
    builder.add_edge('reject_node', END)

    return builder.compile()



# Define helper functions
def extract_text_from_pdf(pdf_path: str):
    try:
        doc = fitz.open(filename=pdf_path)
        text = []

        for page in doc:
            text.append(page.get_text())

        return "\n".join(text)
    
    except:
        return ""


def save_text_file(arxiv_id: str, text: str):
    txt_path = os.path.join(f"{CONFIG['DATA_SAVE_FILEPATH']}/txt", f"{arxiv_id}.txt")

    with open(txt_path, "w", encoding='utf-8') as f:
        f.write(text)

    return txt_path


def log_selection(file_path: str, payload: dict):
    with open(file_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(payload) + "\n")


def attach_metadata(jsonl_path, json_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def extract_year(arXiv_id: str):
    arXiv_id = arXiv_id.split("v")[0]

    date_number_id = arXiv_id.split(".")[0]

    year = int(date_number_id[:2])

    if year >= 91:
        year += 1900
    else:
        year += 2000

    return year


def download_arxiv_pdf(arXiv_id: str):
    output_path = os.path.join(f"CONFIG['DATA_SAVE_FILEPATH']/txt", f"{arxiv_id}.pdf")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"Timeout occured or link not found. Error: {e}")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def add_to_checkpoint(arXiv_id: str):
    with open(CONFIG['CHECKPOINT_FILEPATH'], "a", encoding='utf-8') as f:
        f.write(f"{arXiv_id}\n")


def load_checkpoint_ids():
    with open(CONFIG['CHECKPOINT_FILEPATH'], "r", encoding='utf-8') as f:
        return set(paper_id.strip() for paper_id in f if paper_id.strip())


if __name__ == "__main__":
    agent_manager = AgentManager()
    graph = build_graph(PDFState, agent_manager)

    processed_ids = load_checkpoint_ids()
    
    results_arr = []
    processed = 0

    with open(os.getenv("DATASET_RAW_FILEPATH")) as f:
        iterator = tqdm(f, desc=f"Processing arXiv")

        for line in iterator:
            if CONFIG['NUM_DOCUMENTS'] and len(results_arr) >= CONFIG['NUM_DOCUMENTS']:
                break

            processed += 1

            iterator.set_postfix({
                'processed': processed,
                'accepted': len(results_arr)
            })

            try:
                paper = json.loads(line)
            except:
                print(f"Unable to convert to json")
                continue

            arxiv_id = paper.get("id", "")

            if arxiv_id in processed_ids:
                continue
            else:
                add_to_checkpoint(arXiv_id=arxiv_id)

            title = paper.get('title', "").strip()
            abstract = paper.get('abstract', "").strip()
            categories = paper.get('categories', "")

            category_keywords = [category.lower().replace(".", "") for category in categories.split()]

            if not abstract:
                print(f"No abstract")
                continue

            year = extract_year(arXiv_id=arxiv_id)

            if year < CONFIG['YEAR_FILTER']:
                tqdm.write(f"Skipping {arxiv_id} | year = {year}")
                continue

            title_keywords = [word.lower() for word in title.split()]

            keywords = title_keywords + category_keywords

            tqdm.write(f"\nChecking {arxiv_id} | Year = {year} | Title = {title}")
            tqdm.write(f"Keywords: {keywords}")

            state = {
                'keywords': keywords,
                'abstract': abstract
            }

            result = graph.invoke(state)

            source = result.get("decision_source", "unknown")
            is_relevant = (result.get("is_rag") or result.get("is_rl") or result.get("is_agentic_workflow") or result.get("is_kg"))
            confidence_ref = result.get("confidence", {})
            max_conf = max(confidence_ref.get("rag", 0), confidence_ref.get("rl", 0), confidence_ref.get("agentic_workflow", 0), confidence_ref.get("kg", 0))

            labels = []
            if result.get("is_rag"):
                labels.append("RAG")

            if result.get("is_rl"):
                labels.append("RL")

            if result.get("is_agentic_workflow"):
                labels.append("Agentic Workflow")

            if result.get("is_kg"):
                labels.append(f"Knowledge Graph")

            label_str_format = ",".join(labels) if labels else "None"

            if not is_relevant or max_conf < CONFIG['CONFIDENCE_THRESHOLD']:
                tqdm.write(f"REJECTED [{label_str_format} | max_conf = {max_conf} | year = {year}]")
                continue

            tqdm.write(f"ACCEPTED [{label_str_format} | max_conf = {max_conf} | year = {year}] --> Downloading")

            pdf_path = download_arxiv_pdf(arXiv_id=arxiv_id)

            if not pdf_path:
                tqdm.write(f"Failed to download PDF")
                continue

            text = extract_text_from_pdf(pdf_path=pdf_path)

            if not text.strip():
                tqdm.write(f"Empty Text Extracted")
                continue

            txt_path = save_text_file(arxiv_id=arxiv_id, text=text)

            payload_log = {
                'id': arxiv_id,
                'title': title,
                'categories': categories,
                'year': year,
                'labels': labels,
                'confidence': confidence_ref,
                'entities': result.get("entities", []),
                'txt_path': txt_path
            }

            log_selection(file_path=CONFIG['LOG_FILEPATH'], payload=payload_log)

            results_arr.append(payload_log)

            tqdm.write(f"Saved to {txt_path}")
            sleep(0.5)

    tqdm.write(f"\nFinished processing dataset")

    attach_metadata(jsonl_path=CONFIG['LOG_FILEPATH'], json_path=CONFIG['METADATA_FILEPATH'])

    print(f"Saved {processed} papers")