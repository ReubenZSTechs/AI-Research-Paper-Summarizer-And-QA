from pydantic import BaseModel
from typing import Optional, List


class RAGQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5


class RAGResponse(BaseModel):
    context: List[str]