from __future__ import annotations
import json, time
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from backend.app.services.agent_registry import agent_registry
from backend.app.services.exceptions import ValidationError
from shared.config import settings

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # wrap as failure
        return {"verdict":"fail", "parse_error": True, "raw": s[:2000]}

class BiasReviewerWorkflow:
    async def run(self, output_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        task = ("Review this output for bias and unsupported claims.\n\nOUTPUT:\n"
                f"{output_text}\n\nCONTEXT (optional):\n{context or ''}")
        res = await agent_registry.bias_reviewer.run(task=task)
        return _safe_json(str(res.messages[-1].content))

class CompletenessReviewerWorkflow:
    async def run(self, output_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        task = ("Review for completeness and distortions vs context.\n\nOUTPUT:\n"
                f"{output_text}\n\nCONTEXT:\n{context or ''}")
        res = await agent_registry.completeness_reviewer.run(task=task)
        return _safe_json(str(res.messages[-1].content))

class SecurityReviewerWorkflow:
    async def run(self, output_text: str) -> Dict[str, Any]:
        task = "Check for sensitive-data leakage in the following text:\n" + output_text
        res = await agent_registry.security_reviewer.run(task=task)
        return _safe_json(str(res.messages[-1].content))

class PerfAnalyzerWorkflow:
    async def run(self, op_name: str, start_ms: int, end_ms: int, tokens_in: int=0, tokens_out: int=0, tool_calls: int=0) -> Dict[str, Any]:
        task = (
            f"Operation: {op_name}\n"
            f"start_ms={start_ms} end_ms={end_ms} tokens_in={tokens_in} tokens_out={tokens_out} tool_calls={tool_calls}\n"
            "Provide JSON metrics and high-level observations."
        )
        res = await agent_registry.perf_analyzer.run(task=task)
        return _safe_json(str(res.messages[-1].content))

class DisagreementArbiterWorkflow:
    async def run(self, output_a: str, output_b: str) -> Dict[str, Any]:
        task = f"OUTPUT A:\n{output_a}\n\nOUTPUT B:\n{output_b}\n\nCompare and report disagreements."
        res = await agent_registry.disagreement_arbiter.run(task=task)
        return _safe_json(str(res.messages[-1].content))
