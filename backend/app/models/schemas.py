from pydantic import BaseModel, Field
from typing import List, Optional

class IngestResponse(BaseModel):
    doc_id: str
    sections: List[str]
    summary: str
    entities: List[str]

class SummaryUpdate(BaseModel):
    summary: str
    entities: Optional[List[str]] = Field(default=None)

class QARequest(BaseModel):
    doc_id: Optional[str] = None     # if None, search whole corpus
    question: str

class QAResponse(BaseModel):
    answer: str
    contexts: List[str] = []
