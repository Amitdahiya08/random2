"""
Validation modules for document processing quality control.

This module provides validation logic for summaries, entities, and Q&A responses
to ensure high-quality outputs from the document processing pipeline.
"""
from __future__ import annotations
from typing import Dict, List, Tuple


class SummaryValidator:
    """
    Validator for document summaries.
    
    Provides heuristic validation to ensure summaries meet quality standards
    including length bounds and content coverage requirements.
    """
    
    @staticmethod
    def validate(raw_text: str, summary: str) -> Tuple[bool, Dict]:
        """
        Validate a document summary against the source text.
        
        Args:
            raw_text: The original document text
            summary: The generated summary to validate
            
        Returns:
            Tuple of (is_valid, details) where:
            - is_valid: Boolean indicating if summary passes validation
            - details: Dictionary with validation failure details if applicable
        """
        if not summary or not summary.strip():
            return False, {"reason": "empty_summary"}
        # Heuristics: length bounds & minimal coverage check
        if len(summary) < 40:
            return False, {"reason": "too_short"}
        if len(summary) > 4000:
            return False, {"reason": "too_long"}
        # Simple coverage: at least 2 unique tokens from source appear (reduced from 3)
        src_terms = set(t.lower() for t in raw_text.split()[:500])
        hit = sum(1 for t in set(summary.lower().split()) if t in src_terms)
        if hit < 2:  # extremely loose - reduced threshold for HTML content
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
        
        # Handle case where no entities were found
        if len(entities) == 1 and entities[0].lower() in ["no entities found.", "no entities found", "none"]:
            return True, {}
        
        # Require some entities to appear in text (very loose for HTML content)
        raw_lower = raw_text.lower()
        present = sum(1 for e in entities if e.strip() and e.lower() in raw_lower)
        if entities and present / max(1, len(entities)) < 0.2:  # Reduced from 0.4 to 0.2
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
