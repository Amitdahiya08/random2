"""
Skeleton workflow classes orchestrating AutoGen agents with validation & rollback.

Each workflow exposes a `run(...)` coroutine that:
1) calls the appropriate agent(s),
2) validates outputs using validators,
3) raises ValidationError on failure.

These are intentionally slim so you can swap validators or add richer logic later.
"""
from __future__ import annotations
from typing import List, Tuple
from backend.app.services.agent_registry import agent_registry
from backend.app.services.validators import SummaryValidator, EntityValidator, QAValidator
from backend.app.services.exceptions import ValidationError, SummarizationError, EntityExtractionError, QAError

class SectionSummarizationWorkflow:
    async def run(self, section_text: str) -> str:
        try:
            res = await agent_registry.summarizer.run(
                task="Summarize precisely the following section, keep it brief and factual:\n" + section_text[:120000]
            )
            summary = str(res.messages[-1].content)
            ok, info = SummaryValidator.validate(section_text, summary)
            if not ok:
                raise ValidationError("Section summary did not pass validation", info)
            return summary
        except ValidationError:
            raise
        except Exception as e:
            raise SummarizationError(str(e)) from e

class DocumentSummarizationWorkflow:
    async def run(self, raw_text: str) -> str:
        try:
            res = await agent_registry.summarizer.run(
                task="Summarize the following document. Be concise, section-aware, <=300 words if possible:\n" + raw_text[:120000]
            )
            summary = str(res.messages[-1].content)
            ok, info = SummaryValidator.validate(raw_text, summary)
            if not ok:
                raise ValidationError("Document summary did not pass validation", info)
            return summary
        except ValidationError:
            raise
        except Exception as e:
            raise SummarizationError(str(e)) from e

class CorpusSummarizationWorkflow:
    async def run(self, docs_texts: List[str]) -> str:
        try:
            joined = "\n\n---\n\n".join(d[:80000] for d in docs_texts)[:120000]
            res = await agent_registry.summarizer.run(
                task=(
                    "You will generate a high-level corpus summary across multiple documents. "
                    "Surface common themes and key differences.\n\nCORPUS:\n" + joined
                )
            )
            summary = str(res.messages[-1].content)
            # Validate against first doc as a proxy (loose)
            ok, info = SummaryValidator.validate(docs_texts[0] if docs_texts else "", summary)
            if not ok:
                raise ValidationError("Corpus summary did not pass validation", info)
            return summary
        except ValidationError:
            raise
        except Exception as e:
            raise SummarizationError(str(e)) from e

class EntityExtractionWorkflow:
    async def run(self, raw_text: str) -> List[str]:
        try:
            res = await agent_registry.entity_extractor.run(
                task="Extract key entities (PERSON, ORG, DATE, MONEY, LOCATION, LAW/CLAUSE) one per line:\n" + raw_text[:120000]
            )
            entities = [e.strip() for e in str(res.messages[-1].content).splitlines() if e.strip()]
            ok, info = EntityValidator.validate(raw_text, entities)
            if not ok:
                raise ValidationError("Entities did not pass validation", info)
            return entities
        except ValidationError:
            raise
        except Exception as e:
            raise EntityExtractionError(str(e)) from e

class QAWorkflow:
    async def run(self, question: str, contexts: List[str]) -> Tuple[str, List[str]]:
        try:
            ctx_text = "\n\n".join(contexts)
            qa_task = (
                f"Question: {question}\n\nContext:\n{ctx_text}\n\n"
                "Answer strictly from the context; if unknown, say you don't know."
            )
            res = await agent_registry.qa.run(task=qa_task)
            answer = str(res.messages[-1].content)
            ok, info = QAValidator.validate(answer, contexts)
            if not ok:
                raise ValidationError("QA answer did not pass validation", info)
            return answer, contexts
        except ValidationError:
            raise
        except Exception as e:
            raise QAError(str(e)) from e
