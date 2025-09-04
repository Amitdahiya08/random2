import asyncio
from typing import List
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from agents import system_prompts
from agents.tool_wrappers import (
    mcp_extract_text, mcp_kb_add, mcp_kb_search, mcp_file_read, mcp_file_write
)
from shared.config import settings

async def _azure_client():
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=settings.az_deployment,
        model=settings.az_model,
        api_version=settings.az_api_version,
        azure_endpoint=settings.az_endpoint,
        api_key=settings.az_api_key,
    )
    return client

class AgentRegistry:
    def __init__(self):
        self._initialized = False

    async def init(self):
        if self._initialized:
            return
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
        self._initialized = True

    async def close(self):
        if not self._initialized:
            return
        # Close model client connections
        await self.parser._model_client.close()
        await self.summarizer._model_client.close()
        await self.entity_extractor._model_client.close()
        await self.qa._model_client.close()
        await self.kb._model_client.close()
        self._initialized = False

agent_registry = AgentRegistry()
