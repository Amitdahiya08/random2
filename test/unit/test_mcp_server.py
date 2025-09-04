"""
Synchronous MCP server helpers used by tests.

The unit tests call these functions without awaiting them, so do NOT make them async.
"""

import os
import re
from typing import List, Tuple

# Optional imports (patched in tests for pdf/docx)
try:
    from PyPDF2 import PdfReader  # patched in tests
except Exception:
    PdfReader = None  # type: ignore

try:
    from docx import Document  # patched in tests
except Exception:
    Document = None  # type: ignore

# Bring in the chunker used by KB
from backend.app.services.chunk_utils import split_into_chunks

# In-memory KB: list of (doc_id, chunk_text)
_KB: List[Tuple[str, str]] = []


# -------------------------
# Utility: simple HTML text extractor (remove scripts/styles, strip tags)
# -------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)


def _html_to_text(html: str) -> str:
    # remove scripts/styles
    cleaned = _SCRIPT_BLOCK_RE.sub("", html)
    cleaned = _STYLE_BLOCK_RE.sub("", cleaned)
    # remove all tags
    cleaned = _TAG_RE.sub(" ", cleaned)
    # collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# -------------------------
# File/Text extraction
# -------------------------
def extract_text(path: str) -> str:
    """
    Extract text from txt/html/pdf/docx. For unknown extensions, read as text.
    Raise FileNotFoundError when file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".xyz"):  # tests expect unsupported -> fallback to plain read
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _html_to_text(f.read())

    if ext == ".pdf":
        if PdfReader is None:
            # Fallback: just read raw bytes as text
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")
        reader = PdfReader(path)
        texts = []
        for page in getattr(reader, "pages", []):
            # mocked in tests
            txt = page.extract_text()  # type: ignore
            if txt:
                texts.append(txt)
        return "\n".join(texts).strip()

    if ext == ".docx":
        if Document is None:
            return ""
        doc = Document(path)  # mocked in tests
        return "\n".join(p.text for p in getattr(doc, "paragraphs", [])).strip()

    # Fallback for anything else: read as text
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -------------------------
# Knowledge base ops
# -------------------------
def kb_add(doc_id: str, text: str, *, max_chars: int = 500) -> str:
    """
    Split text and index into in-memory KB. Returns a status string like:
    'Indexed N chunks for <doc_id>'
    """
    global _KB
    # Treat empty/whitespace text as 0 chunks
    if text is None or text.strip() == "":
        count = 0
        return f"Indexed {count} chunks for {doc_id}"

    chunks = split_into_chunks(text, max_chars=max_chars)
    # Some tests expect whitespace-only -> 0, but non-empty content -> >=1
    # If the single chunk is just whitespace, consider it 0
    effective: List[str] = [c for c in chunks if c.strip() != ""]
    for c in effective:
        _KB.append((doc_id, c))
    return f"Indexed {len(effective)} chunks for {doc_id}"


def kb_search(query: str, *, top_k: int = 3) -> str:
    """
    Very simple search: return the first top_k chunks containing any query term,
    else return top_k earliest chunks. If KB empty, return 'KB is empty.'
    """
    if not _KB:
        return "KB is empty."

    q = query.strip().lower()
    if not q:
        hits = _KB[:top_k]
    else:
        terms = [t for t in re.split(r"\W+", q) if t]
        def score(chunk: str) -> int:
            text_l = chunk.lower()
            return sum(term in text_l for term in terms)

        ranked = sorted(_KB, key=lambda item: score(item[1]), reverse=True)
        hits = [item for item in ranked if score(item[1]) > 0][:top_k]
        if not hits:
            # Return something even if not relevant (as the test expects non-empty)
            hits = _KB[:top_k]

    # Return as a joined string (tests just check substrings exist / non-empty)
    return "\n".join(c for _, c in hits)


# -------------------------
# File ops
# -------------------------
def file_read(path: str) -> str:
    """Read file text or raise FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def file_write(path: str, content: str) -> str:
    """Write text to file (overwrite) and return 'ok'."""
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(content or "")
    return "ok"
