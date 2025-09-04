# Do not print to stdout; MCP uses stdio. Use logging to stderr.
import os, json, logging, asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import nltk

# Configure logging to stderr before any usage
logger = logging.getLogger("mcp-server")
logging.basicConfig(level=logging.INFO)

# NLTK token setup for BM25 with safe fallback (no stdout)
def _tokenize_with_mode(text: str) -> tuple[list[str], str]:
    """Tokenize text, returning tokens and the tokenizer mode used ('punkt' or 'fallback')."""
    try:
        nltk.data.find("tokenizers/punkt")
        return nltk.word_tokenize(text), "punkt"
    except Exception:
        tokens = text.split()
        return tokens, "fallback"

mcp = FastMCP("docqa_tools")

KB_PATH = os.path.join(os.path.dirname(__file__), "kb_store.json")

def _load_kb():
    if not os.path.exists(KB_PATH):
        return {"docs": {}}
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_kb(kb):
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

@mcp.tool()
async def file_read(path: str) -> str:
    """Read a UTF-8 text file and return content."""
    logger.info(f"MCP tool 'file_read' called: path={path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

@mcp.tool()
async def file_write(path: str, content: str) -> str:
    """Write UTF-8 text to file and return 'ok'."""
    logger.info(f"MCP tool 'file_write' called: path={path}, content_len={len(content)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "ok"

@mcp.tool()
async def extract_text(path: str) -> str:
    """Extract text from PDF/DOCX/HTML/TXT located at 'path'."""
    logger.info(f"MCP tool 'extract_text' called: path={path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove script/style
            for s in soup(["script", "style"]): s.decompose()
            # Get text and clean up whitespace
            text = soup.get_text("\n")
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return "\n".join(lines)
    # fallback: txt
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

@mcp.tool()
async def kb_add(doc_id: str, text: str) -> str:
    """Add or update KB entries for a doc_id from a 'text' that contains multiple chunks separated by blank lines."""
    logger.info(f"MCP tool 'kb_add' called: doc_id={doc_id}, text_len={len(text)}")
    kb = _load_kb()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    kb["docs"][doc_id] = {"chunks": chunks}
    _save_kb(kb)
    logger.info(f"MCP tool 'kb_add' completed: doc_id={doc_id}, chunks_indexed={len(chunks)}")
    return f"Indexed {len(chunks)} chunks for {doc_id}"

@mcp.tool()
async def kb_search(query: str, top_k: int = 5) -> str:
    """Simple BM25 search over all KB chunks. Returns top_k chunks concatenated."""
    logger.info(f"MCP tool 'kb_search' called: query_len={len(query)}, top_k={top_k}")
    kb = _load_kb()
    corpus = []
    owners = []
    for did, entry in kb["docs"].items():
        for ch in entry.get("chunks", []):
            corpus.append(ch)
            owners.append(did)
    if not corpus:
        logger.info("MCP tool 'kb_search': KB is empty")
        return "KB is empty."
    # Tokenize
    tokenized = []
    mode_used = None
    for c in corpus:
        toks, mode = _tokenize_with_mode(c)
        tokenized.append(toks)
        mode_used = mode if mode_used is None else mode_used
    q_tokens, q_mode = _tokenize_with_mode(query)
    # Prefer to report the query mode if it differs
    effective_mode = q_mode if q_mode != (mode_used or q_mode) else (mode_used or q_mode)
    logger.info(f"MCP tool 'kb_search': tokenizer_mode={effective_mode}, corpus_size={len(corpus)}")
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(list(enumerate(scores)), key=lambda t: t[1], reverse=True)[:max(1, int(top_k))]
    out = []
    for rank, (idx, sc) in enumerate(ranked, start=1):
        out.append(f"[Chunk {idx} | score {sc:.2f}]\n{corpus[idx]}")
    logger.info(f"MCP tool 'kb_search' completed: returned={len(out)}")
    return "\n\n".join(out)

if __name__ == "__main__":
    # Run over stdio
    try:
        logger.info("Starting MCP server 'docqa_tools' over stdio. Waiting for a client/host connection...")
        mcp.run(transport="stdio")
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("MCP server shutdown requested")
    except Exception as e:
        # Log error to stderr without printing to stdout
        logger.error(f"MCP server terminated with error: {e}")
