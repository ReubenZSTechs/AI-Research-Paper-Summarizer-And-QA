from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List, Optional

CONFIG = {
    'PAPER_CLASSIFIER_MODEL': 'deepseek-r1:32b'
}

class PDFState(TypedDict):
    keywords: List[str]
    abstract: str

    keyword_match: bool
    llm_result: Optional[Dict]

    is_rag: Optional[bool]
    is_rl: Optional[bool]
    is_langgraph: Optional[bool]
    is_kg: Optional[bool]

    confidence: Optional[Dict[str, float]]
    decision_source: Optional[str]

    entities: Optional[List]