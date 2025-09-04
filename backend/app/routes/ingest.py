import os, tempfile
from fastapi import APIRouter, UploadFile, HTTPException
from storage.local_store import make_doc_id
from backend.app.services.orchestrator import ingest_document

router = APIRouter()

@router.post("/ingest")
async def ingest(file: UploadFile):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in [".pdf", ".docx", ".html", ".htm", ".txt"]:
        raise HTTPException(400, "Unsupported file format. Use PDF, DOCX, HTML, or TXT.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    doc_id = make_doc_id(file.filename or "document")
    sections, summary, entities = await ingest_document(tmp_path, doc_id)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return {"doc_id": doc_id, "sections": sections, "summary": summary, "entities": entities}
