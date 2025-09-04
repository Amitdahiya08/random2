import asyncio, json
from typing import Dict, List, Tuple
from backend.app.services.langsmith_logger import traceable
from backend.app.services.agent_registry import agent_registry
from agents.tool_wrappers import mcp_kb_add
from backend.app.services.chunk_utils import split_into_chunks
from backend.app.services.exceptions import (
    ParsingError, SummarizationError, EntityExtractionError, QAError, ValidationError
)
from backend.app.services.rollback import take_summary_snapshot, rollback_summary
from agents.workflows import (
    DocumentSummarizationWorkflow, EntityExtractionWorkflow, QAWorkflow
)
from storage.local_store import put_document, update_summary, get_document
from backend.app.services.validators import SummaryValidator

doc_summarizer = DocumentSummarizationWorkflow()
entity_workflow = EntityExtractionWorkflow()
qa_workflow = QAWorkflow()

@traceable("ingest_document")
async def ingest_document(file_path: str, doc_id: str) -> Tuple[List[str], str, List[str]]:
    """Parse -> Summarize -> Extract Entities -> Index KB with validations & rollback."""
    # 1) Parse
    parse_task = "Parse the document at this path and return JSON with keys: sections, raw_text.\nPATH: " + file_path
    parse_res = await agent_registry.parser.run(task=parse_task)
    try:
        last = parse_res.messages[-1].content
        parsed_json = json.loads(last if isinstance(last, str) else last[0])
        raw_text = parsed_json.get("raw_text", "")
        sections = parsed_json.get("sections", []) or split_into_chunks(raw_text, 3000)
        if not raw_text.strip():
            raise ParsingError("Parser returned empty raw_text")
    except Exception as e:
        raise ParsingError(f"Failed to parse content: {e}") from e

    # Snapshot for rollback
    snapshot = take_summary_snapshot(doc_id)

    # 2) Summarize (validated)
    try:
        summary = await doc_summarizer.run(raw_text)
    except ValidationError as ve:
        # no prior state for new doc; just propagate
        raise SummarizationError(f"Summary validation failed: {ve.details}") from ve
    except Exception as e:
        raise SummarizationError(str(e)) from e

    # 3) Entities (validated)
    try:
        entities = await entity_workflow.run(raw_text)
    except ValidationError as ve:
        # rollback any partial writes we might do later
        if snapshot: rollback_summary(snapshot)
        raise EntityExtractionError(f"Entity validation failed: {ve.details}") from ve
    except Exception as e:
        if snapshot: rollback_summary(snapshot)
        raise EntityExtractionError(str(e)) from e

    # 4) Chunk & KB index via MCP
    chunks = split_into_chunks(raw_text, 1200)
    chunks_text = "\n\n".join(chunks)
    await mcp_kb_add(doc_id=doc_id, text=chunks_text)

    # 5) Persist (if this fails, no special rollback needed as it's first write)
    put_document(doc_id, raw_text=raw_text, sections=sections, summary=summary, entities=entities)
    return sections, summary, entities

@traceable("answer_question")
async def answer_question(question: str, doc_id: str | None) -> Tuple[str, List[str]]:
    """Retrieve via MCP -> QA agent with grounded context + validation."""
    contexts: List[str]
    if doc_id:
        doc = get_document(doc_id)
        candidates = doc["sections"] if doc else []
        mcp_hits = await agent_registry.qa.run(task=f"Search KB for: {question}\nReturn top 5 chunks by calling mcp_kb_search.")
        retrieved = str(mcp_hits.messages[-1].content)
        contexts = (candidates[:5]) + [retrieved]
    else:
        mcp_hits = await agent_registry.qa.run(task=f"Search KB for: {question}\nReturn top 8 chunks by calling mcp_kb_search.")
        contexts = [str(mcp_hits.messages[-1].content)]

    try:
        answer, ctxs = await qa_workflow.run(question, contexts)
        return answer, ctxs
    except ValidationError as ve:
        # Provide a graceful degraded response
        fallback = "I don't know from the provided documents."
        return f"{fallback}\n\n(validator: {ve.details})", contexts
    except Exception as e:
        raise QAError(str(e)) from e

def apply_user_edit(doc_id: str, summary: str, entities: List[str] | None):
    """Apply user edit with validation + rollback on failure."""
    # Take snapshot
    snapshot = take_summary_snapshot(doc_id)
    # Validate user-provided content too
    doc = get_document(doc_id)
    raw_text = (doc or {}).get("raw_text", "")
    ok_s, info_s = (True, {}) if not summary else SummaryValidator.validate(raw_text, summary)  # type: ignore # late import
    if not ok_s:
        raise ValidationError("User summary failed validation", info_s)
    if entities is not None:
        from backend.app.services.validators import EntityValidator  # local import to avoid cycle
        ok_e, info_e = EntityValidator.validate(raw_text, entities)
        if not ok_e:
            raise ValidationError("User entities failed validation", info_e)
    try:
        update_summary(doc_id, summary, entities)
    except Exception as e:
        # rollback
        if snapshot:
            rollback_summary(snapshot)
        raise e
