from backend.app.services.mcp_bridge import mcp_bridge

# AutoGen tools must be async functions that return strings.

async def mcp_extract_text(path: str) -> str:
    """Extract text from PDF/DOCX/HTML/TXT using MCP server."""
    return await mcp_bridge.call("extract_text", {"path": path})

async def mcp_kb_add(doc_id: str, text: str) -> str:
    """Add text (with chunks separated by blank lines) into KB under doc_id."""
    return await mcp_bridge.call("kb_add", {"doc_id": doc_id, "text": text})

async def mcp_kb_search(query: str, top_k: int = 5) -> str:
    """Search KB; returns top_k chunks (joined)."""
    return await mcp_bridge.call("kb_search", {"query": query, "top_k": top_k})

async def mcp_file_read(path: str) -> str:
    return await mcp_bridge.call("file_read", {"path": path})

async def mcp_file_write(path: str, content: str) -> str:
    return await mcp_bridge.call("file_write", {"path": path, "content": content})
