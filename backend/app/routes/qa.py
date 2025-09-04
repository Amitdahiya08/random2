from fastapi import APIRouter
from backend.app.models.schemas import QARequest
from backend.app.services.orchestrator import answer_question

router = APIRouter()

@router.post("/qa")
async def qa(payload: QARequest):
    answer, contexts = await answer_question(payload.question, payload.doc_id)
    return {"answer": answer, "contexts": contexts}
