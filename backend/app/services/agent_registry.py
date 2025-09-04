import asyncio
from typing import List
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from agents import system_prompts
from agents.tool_wrappers import (
    mcp_extract_text, mcp_kb_add, mcp_kb_search, mcp_file_read, mcp_file_write
)
from agents.critics import (
    BIAS_REVIEW_PROMPT, COMPLETENESS_REVIEW_PROMPT,
    SECURITY_REVIEW_PROMPT, PERF_ANALYZER_PROMPT, DISAGREEMENT_ARBITER_PROMPT
)
from shared.config import settings

async def _azure_client():
    return AzureOpenAIChatCompletionClient(
        azure_deployment=settings.az_deployment,
        model=settings.az_model,
        api_version=settings.az_api_version,
        azure_endpoint=settings.az_endpoint,
        api_key=settings.az_api_key,
    )

class AgentRegistry:
    def __init__(self):
        self._initialized = False

    async def init(self):
        if self._initialized: return
        # Core agents (existing)
        self.parser = AssistantAgent(
            name="parser",
            model_client=await _azure_client(),
            system_message=system_prompts.PARSER_PROMPT,
            tools=[mcp_extract_text, mcp_file_read],
            reflect_on_tool_use=True,
            model_client_stream=False,
        )
        self.summarizer = AssistantAgent(
            name="summarizer",
            model_client=await _azure_client(),
            system_message=system_prompts.SUMMARIZER_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
            model_client_stream=False,
        )
        self.entity_extractor = AssistantAgent(
            name="entity_extractor",
            model_client=await _azure_client(),
            system_message=system_prompts.ENTITY_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
            model_client_stream=False,
        )
        self.qa = AssistantAgent(
            name="qa",
            model_client=await _azure_client(),
            system_message=system_prompts.QA_PROMPT,
            tools=[mcp_kb_search],
            reflect_on_tool_use=True,
            model_client_stream=False,
        )
        self.kb = AssistantAgent(
            name="kb_agent",
            model_client=await _azure_client(),
            system_message="You maintain the KB by calling tools; acknowledge once done.",
            tools=[mcp_kb_add],
            reflect_on_tool_use=True,
        )

        # === New critic/reviewer agents ===
        self.bias_reviewer = AssistantAgent(
            name="bias_reviewer",
            model_client=await _azure_client(),
            system_message=BIAS_REVIEW_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
        )
        self.completeness_reviewer = AssistantAgent(
            name="completeness_reviewer",
            model_client=await _azure_client(),
            system_message=COMPLETENESS_REVIEW_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
        )
        self.security_reviewer = AssistantAgent(
            name="security_reviewer",
            model_client=await _azure_client(),
            system_message=SECURITY_REVIEW_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
        )
        self.perf_analyzer = AssistantAgent(
            name="perf_analyzer",
            model_client=await _azure_client(),
            system_message=PERF_ANALYZER_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
        )
        self.disagreement_arbiter = AssistantAgent(
            name="disagreement_arbiter",
            model_client=await _azure_client(),
            system_message=DISAGREEMENT_ARBITER_PROMPT,
            tools=[],
            reflect_on_tool_use=False,
        )
        self._initialized = True

    async def close(self):
        if not self._initialized: return
        for ag in [
            self.parser, self.summarizer, self.entity_extractor, self.qa, self.kb,
            self.bias_reviewer, self.completeness_reviewer, self.security_reviewer,
            self.perf_analyzer, self.disagreement_arbiter
        ]:
            if hasattr(ag, '_model_client'):
                await ag._model_client.close()
        self._initialized = False

agent_registry = AgentRegistry()
