from __future__ import annotations
from typing import Dict, List, Tuple

class SummaryValidator:
    """Very lightweight, heuristic validator for summaries."""
    @staticmethod
    def validate(raw_text: str, summary: str) -> Tuple[bool, Dict]:
        if not summary or not summary.strip():
            return False, {"reason": "empty_summary"}
        # Heuristics: length bounds & minimal coverage check
        if len(summary) < 40:
            return False, {"reason": "too_short"}
        if len(summary) > 4000:
            return False, {"reason": "too_long"}
        # Simple coverage: at least 3 unique tokens from source appear
        src_terms = set(t.lower() for t in raw_text.split()[:500])
        hit = sum(1 for t in set(summary.lower().split()) if t in src_terms)
        if hit < 3:  # extremely loose
            return False, {"reason": "low_coverage"}
        return True, {}

class EntityValidator:
    """Check entities are non-empty and plausible wrt source."""
    @staticmethod
    def validate(raw_text: str, entities: List[str]) -> Tuple[bool, Dict]:
        if entities is None:
            return True, {}
        if len(entities) > 200:
            return False, {"reason": "too_many_entities"}
        # Require most entities to appear in text (loose)
        raw_lower = raw_text.lower()
        present = sum(1 for e in entities if e.strip() and e.lower() in raw_lower)
        if entities and present / max(1, len(entities)) < 0.4:
            return False, {"reason": "low_presence", "present_ratio": present / max(1, len(entities))}
        return True, {}

class QAValidator:
    """Check QA answers reference context and avoid hallucination format."""
    @staticmethod
    def validate(answer: str, contexts: List[str]) -> Tuple[bool, Dict]:
        if not answer or not answer.strip():
            return False, {"reason": "empty_answer"}
        joined = "\n".join(contexts).lower()
        # Require at least 2 tokens from answer to be in context (loose)
        tokens = [t for t in answer.lower().split() if t.isalpha()]
        hits = sum(1 for t in set(tokens) if t in joined)
        if hits < 2 and "don't know" not in answer.lower():
            return False, {"reason": "ungrounded_answer"}
        return True, {}
