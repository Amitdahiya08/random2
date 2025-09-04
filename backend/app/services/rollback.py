"""
Very lightweight rollback helper for user-visible fields.
We snapshot the current summary/entities before writing, and revert on failure.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from storage.local_store import get_document, update_summary

@dataclass
class SummarySnapshot:
    doc_id: str
    summary: str
    entities: List[str]

def take_summary_snapshot(doc_id: str) -> Optional[SummarySnapshot]:
    doc = get_document(doc_id)
    if not doc:
        return None
    return SummarySnapshot(doc_id=doc_id, summary=doc.get("summary", ""), entities=list(doc.get("entities", [])))

def rollback_summary(snapshot: SummarySnapshot) -> None:
    update_summary(snapshot.doc_id, snapshot.summary, snapshot.entities)
