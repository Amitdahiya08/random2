from fastapi import FastAPI
from backend.app.routes import ingest, summary, qa
from backend.app.services.mcp_bridge import mcp_bridge
from backend.app.services.agent_registry import agent_registry

app = FastAPI(title="MCP-AutoGen DocQA")

@app.on_event("startup")
async def _startup():
    # Start MCP server & client
    await mcp_bridge.start()
    # Init agents
    await agent_registry.init()

@app.on_event("shutdown")
async def _shutdown():
    await agent_registry.close()
    await mcp_bridge.stop()

app.include_router(ingest.router)
app.include_router(summary.router)
app.include_router(qa.router)
