import asyncio
from backend.app.services.mcp_bridge import mcp_bridge


async def main():
    await mcp_bridge.start()
    try:
        tools = await mcp_bridge.list_tools()
        print("TOOLS:", ",".join(sorted(tools)))
    finally:
        await mcp_bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())


