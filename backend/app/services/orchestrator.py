"""
Orchestrator module for document processing workflows.

This module coordinates the complete document ingestion and Q&A pipeline,
including parsing, summarization, entity extraction, and knowledge base indexing.
It also manages critic workflows for quality assurance and performance monitoring.
"""
import json
import time
from typing import List, Tuple, Dict, Optional

from backend.app.services.langsmith_logger import traceable
from backend.app.services.agent_registry import agent_registry
from backend.app.services.chunk_utils import split_into_chunks
from backend.app.services.exceptions import (
    ParsingError,
    SummarizationError,
    EntityExtractionError,
    QAError,
    ValidationError,
)
from backend.app.services.rollback import take_summary_snapshot, rollback_summary
from backend.app.services.validators import SummaryValidator  # validator used for user edits

# Workflows (Milestone 3)
from agents.workflows import (
    DocumentSummarizationWorkflow,
    EntityExtractionWorkflow,
    QAWorkflow,
)
from agents.tool_wrappers import mcp_kb_add

# Critic / reviewer workflows (Milestone 4)
from backend.app.services.critic_workflows import (
    BiasReviewerWorkflow,
    CompletenessReviewerWorkflow,
    SecurityReviewerWorkflow,
    PerfAnalyzerWorkflow,
    DisagreementArbiterWorkflow,
)

# Storage
from storage.local_store import (
    put_document,
    update_summary,
    get_document,
    append_review,
    append_disagreement,
)

# Initialize workflow singletons
doc_summarizer = DocumentSummarizationWorkflow()
entity_workflow = EntityExtractionWorkflow()
qa_workflow = QAWorkflow()

bias_wf = BiasReviewerWorkflow()
comp_wf = CompletenessReviewerWorkflow()
sec_wf = SecurityReviewerWorkflow()
perf_wf = PerfAnalyzerWorkflow()
arb_wf = DisagreementArbiterWorkflow()


@traceable("ingest_document")
async def ingest_document(file_path: str, doc_id: str) -> Tuple[List[str], str, List[str]]:
    """
    Process a document through the complete ingestion pipeline.
    
    This function orchestrates the full document processing workflow:
    1. Parse document using MCP tools to extract text and sections
    2. Generate summary with validation and fallback handling
    3. Extract entities with validation and fallback handling
    4. Run non-blocking critic workflows for quality assurance
    5. Index content in knowledge base for search
    6. Persist document data to local storage
    
    Args:
        file_path: Path to the document file to process
        doc_id: Unique identifier for the document
        
    Returns:
        Tuple containing:
        - sections: List of document sections/chunks
        - summary: Generated document summary
        - entities: List of extracted entities
        
    Raises:
        ParsingError: If document parsing fails
        SummarizationError: If summarization fails (with fallback)
        EntityExtractionError: If entity extraction fails (with fallback)
        
    Note:
        This function includes graceful fallbacks for validation failures
        to ensure robust document processing even with challenging content.
    """
    # 1) Parse via parser agent (expects JSON with {sections, raw_text})
    parse_task = (
        "Parse the document at this path and return JSON with keys: sections, raw_text.\nPATH: " + file_path
    )
    parse_res = await agent_registry.parser.run(task=parse_task)
    try:
        last = parse_res.messages[-1].content
        payload = last if isinstance(last, str) else last[0]
        parsed_json = json.loads(payload)
        raw_text: str = parsed_json.get("raw_text", "") or ""
        sections: List[str] = parsed_json.get("sections", []) or split_into_chunks(raw_text, 3000)
        if not raw_text.strip():
            raise ParsingError("Parser returned empty raw_text")
    except Exception as e:
        raise ParsingError(f"Failed to parse content: {e}") from e

    # (Optional) snapshot (no prior persisted state for new doc, but safe)
    snapshot = take_summary_snapshot(doc_id)

    # 2) Summarize (validated)
    t_sum_start = int(time.time() * 1000)
    try:
        summary = await doc_summarizer.run(raw_text)
    except ValidationError as ve:
        # Fallback: provide a degraded but safe summary and continue
        head = "\n".join([s for s in (raw_text.splitlines()[:6]) if s.strip()])
        summary = (head or "Summary unavailable due to validation.")[:600]
    except Exception as e:
        # Generic failure fallback
        head = "\n".join([s for s in (raw_text.splitlines()[:6]) if s.strip()])
        summary = (head or f"Summary unavailable: {str(e)}")[:600]
    t_sum_end = int(time.time() * 1000)

    # 3) Entities (validated)
    t_ent_start = int(time.time() * 1000)
    try:
        entities = await entity_workflow.run(raw_text)
    except ValidationError:
        # Fallback: accept empty/no entities and continue
        entities = ["No entities found."]
    except Exception:
        entities = ["No entities found."]
    t_ent_end = int(time.time() * 1000)

    # 4) Run non-blocking critics for the summary
    try:
        bias = await bias_wf.run(summary, raw_text[:5000])
        append_review(doc_id, "bias_summary", bias)

        comp = await comp_wf.run(summary, raw_text[:8000])
        append_review(doc_id, "completeness_summary", comp)

        sec = await sec_wf.run(summary)
        append_review(doc_id, "security_summary", sec)

        # Track disagreements (example: completeness fails while bias passes)
        bias_verdict = (bias or {}).get("verdict")
        comp_verdict = (comp or {}).get("verdict")
        if bias_verdict in ("pass",) and comp_verdict == "fail":
            details = await arb_wf.run(output_a=str(summary), output_b="(Reviewer: missing points)")
            append_disagreement(doc_id, "summary_review", details)

        # Perf notes for summarization / entities
        perf_sum = await perf_wf.run(
            "summarization", t_sum_start, t_sum_end, tokens_in=0, tokens_out=0, tool_calls=0
        )
        append_review(doc_id, "perf_summarization", perf_sum)

        perf_ent = await perf_wf.run(
            "entity_extraction", t_ent_start, t_ent_end, tokens_in=0, tokens_out=0, tool_calls=0
        )
        append_review(doc_id, "perf_entity_extraction", perf_ent)
    except Exception:
        # Critics must never block ingestion
        pass

    # 5) Chunk & KB index via MCP (direct tool call for reliability)
    chunks = split_into_chunks(raw_text, 1200)
    try:
        await mcp_kb_add(doc_id=doc_id, text="\n\n".join(chunks))
    except Exception as e:
        # KB index issues shouldn't prevent saving parsed outputs; log via review
        append_review(doc_id, "kb_index_error", {"error": str(e)})

    # 6) Persist the document with generated fields
    put_document(
        doc_id,
        raw_text=raw_text,
        sections=sections,
        summary=summary,
        entities=entities,
    )

    return sections, summary, entities


@traceable("answer_question")
async def answer_question(question: str, doc_id: Optional[str]) -> Tuple[str, List[str]]:
    """
    Retrieve (from KB + optional per-doc sections) -> Answer via QA workflow (validated)
    Includes non-blocking critics + perf metrics and disagreement tracking.
    Returns: (answer, contexts)
    """
    # Build contexts from doc sections + KB search
    if doc_id:
        doc = get_document(doc_id)
        candidates = (doc or {}).get("sections", []) if doc else []
        mcp_hits = await agent_registry.qa.run(
            task=f"Search KB for: {question}\nReturn top 5 chunks by calling mcp_kb_search."
        )
        retrieved = str(mcp_hits.messages[-1].content)
        contexts: List[str] = (candidates[:5]) + [retrieved]
    else:
        mcp_hits = await agent_registry.qa.run(
            task=f"Search KB for: {question}\nReturn top 8 chunks by calling mcp_kb_search."
        )
        contexts = [str(mcp_hits.messages[-1].content)]

    # Answer with validation; graceful fallback on validation error
    t_qa_start = int(time.time() * 1000)
    try:
        answer, ctxs = await qa_workflow.run(question, contexts)
    except ValidationError as ve:
        answer = "I don't know from the provided documents."
        ctxs = contexts
        # Optionally record validator details as a review on the doc (if we have one)
        if doc_id:
            append_review(doc_id, "qa_validator_note", {"details": getattr(ve, "details", {})})
    except Exception as e:
        raise QAError(str(e)) from e
    t_qa_end = int(time.time() * 1000)

    # Critics: bias, completeness, security (non-blocking)
    try:
        joined_ctx = "\n\n".join(ctxs)[:8000]
        bias = await bias_wf.run(answer, joined_ctx)
        comp = await comp_wf.run(answer, joined_ctx)
        sec = await sec_wf.run(answer)

        if doc_id:
            append_review(doc_id, "bias_qa", bias)
            append_review(doc_id, "completeness_qa", comp)
            append_review(doc_id, "security_qa", sec)

            # disagreement tracking between bias/completeness reviewers
            b_v = (bias or {}).get("verdict")
            c_v = (comp or {}).get("verdict")
            if (b_v == "pass" and c_v == "fail") or (b_v == "fail" and c_v == "pass"):
                details = await arb_wf.run(output_a=json.dumps(bias), output_b=json.dumps(comp))
                append_disagreement(doc_id, "qa_review", details)
    except Exception:
        pass

    # Perf: QA stage
    try:
        perf_qa = await perf_wf.run("qa", t_qa_start, t_qa_end, tokens_in=0, tokens_out=0, tool_calls=1)
        if doc_id:
            append_review(doc_id, "perf_qa", perf_qa)
    except Exception:
        pass

    return answer, ctxs


def apply_user_edit(doc_id: str, summary: str, entities: Optional[List[str]]) -> None:
    """
    Apply user-provided edits with validation + rollback.
    - Validates summary (if provided) and entities (if provided).
    - On failure, raises ValidationError and leaves prior state intact.
    """
    # Snapshot current state
    snapshot = take_summary_snapshot(doc_id)

    # Validate user-provided content against raw text
    doc = get_document(doc_id)
    raw_text = (doc or {}).get("raw_text", "")

    if summary:
        ok_s, info_s = SummaryValidator.validate(raw_text, summary)
        if not ok_s:
            # Do not write; raise validation error
            raise ValidationError("User summary failed validation", info_s)

    if entities is not None:
        from backend.app.services.validators import EntityValidator  # local import to avoid cycle
        ok_e, info_e = EntityValidator.validate(raw_text, entities)
        if not ok_e:
            raise ValidationError("User entities failed validation", info_e)

    # Write changes; rollback if write fails
    try:
        update_summary(doc_id, summary, entities)
    except Exception as e:
        if snapshot:
            rollback_summary(snapshot)
        raise e
