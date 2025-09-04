import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from shared.config import settings

logger = logging.getLogger(__name__)

class McpBridge:
    def __init__(self):
        self._exit: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    async def start(self):
        try:
            self._exit = AsyncExitStack()

            # Ensure the server path exists
            server_path = os.path.abspath(settings.mcp_server_path)
            if not os.path.exists(server_path):
                raise FileNotFoundError(f"MCP server not found at: {server_path}")

            logger.info(f"Starting MCP server at: {server_path}")

            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_path],
                env=None
            )

            # Add timeout for server startup
            stdio = await asyncio.wait_for(
                self._exit.enter_async_context(stdio_client(server_params)),
                timeout=30.0
            )
            self._stdio, self._write = stdio

            self._session = await self._exit.enter_async_context(
                ClientSession(self._stdio, self._write)
            )

            # Add timeout for initialization
            await asyncio.wait_for(self._session.initialize(), timeout=10.0)
            # List tools for a clear startup confirmation
            try:
                tools_resp = await asyncio.wait_for(self._session.list_tools(), timeout=10.0)
                tool_names = ",".join(sorted([t.name for t in tools_resp.tools]))
                logger.info(f"MCP server started successfully; tools: {tool_names}")
            except Exception as list_err:
                logger.warning(f"MCP started but failed listing tools: {list_err}")

        except asyncio.TimeoutError:
            logger.error("MCP server startup timed out")
            await self.stop()
            raise RuntimeError("MCP server startup timed out")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            await self.stop()
            raise

    async def stop(self):
        if self._exit:
            await self._exit.aclose()
        self._exit = None
        self._session = None

    async def list_tools(self) -> List[str]:
        assert self._session
        resp = await self._session.list_tools()
        return [t.name for t in resp.tools]

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        assert self._session
        try:
            logger.info(f"MCP call: tool={tool_name} args_keys={list(args.keys())}")
            result = await self._session.call_tool(tool_name, args)
            # result.content is a list of parts; we'll join any string parts
            parts = []
            for c in result.content:
                if isinstance(c, str):
                    parts.append(c)
                else:
                    # Handle different content types from MCP
                    if hasattr(c, 'text'):
                        parts.append(c.text)
                    elif hasattr(c, 'content'):
                        parts.append(c.content)
                    else:
                        parts.append(str(c))
            output = "\n".join(parts)
            logger.info(f"MCP call done: tool={tool_name} output_len={len(output)}")
            return output
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            raise

# Singleton instance for app
mcp_bridge = McpBridge()
