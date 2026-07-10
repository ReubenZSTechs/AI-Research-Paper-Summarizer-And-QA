from pydantic import BaseModel
from typing import Optional


class ChatQuery(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str