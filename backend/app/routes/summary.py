from fastapi import APIRouter, HTTPException
from storage.local_store import get_document
from backend.app.models.schemas import SummaryUpdate
from backend.app.services.orchestrator import apply_user_edit

router = APIRouter()

@router.get("/summary/{doc_id}")
async def get_summary(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {"doc_id": doc_id, "summary": doc["summary"], "entities": doc["entities"], "sections": doc["sections"]}

@router.put("/summary/{doc_id}")
async def update_summary(doc_id: str, payload: SummaryUpdate):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    apply_user_edit(doc_id, payload.summary, payload.entities)
    return {"ok": True}
