import json, os, hashlib, time
from typing import Dict, Any, Optional
from shared.config import settings

DOCS_FILE = os.path.join(settings.data_dir, "docs_index.json")

def _load_index() -> Dict[str, Any]:
    if not os.path.exists(DOCS_FILE):
        return {}
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_index(idx: Dict[str, Any]) -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def make_doc_id(filename: str) -> str:
    base = f"{filename}-{time.time()}"
    return hashlib.sha1(base.encode()).hexdigest()[:16]

def put_document(doc_id: str, raw_text: str, sections=None, summary=None, entities=None) -> None:
    idx = _load_index()
    idx[doc_id] = {
        "doc_id": doc_id,
        "raw_text": raw_text,
        "sections": sections or [],
        "summary": summary or "",
        "entities": entities or [],
        "reviews": [],         # list of {type, payload, timestamp}
        "disagreements": []    # list of {phase, details, timestamp}
    }
    _save_index(idx)

def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    return _load_index().get(doc_id)

def update_summary(doc_id: str, summary: str, entities: Optional[list]=None):
    idx = _load_index()
    if doc_id in idx:
        idx[doc_id]["summary"] = summary
        if entities is not None:
            idx[doc_id]["entities"] = entities
        _save_index(idx)

def all_docs() -> Dict[str, Any]:
    return _load_index()

# ... existing imports ...


def append_review(doc_id: str, review_type: str, payload: dict):
    idx = _load_index()
    if doc_id in idx:
        idx[doc_id].setdefault("reviews", []).append({
            "type": review_type, "payload": payload, "ts": int(time.time()*1000)
        })
        _save_index(idx)

def append_disagreement(doc_id: str, phase: str, details: dict):
    idx = _load_index()
    if doc_id in idx:
        idx[doc_id].setdefault("disagreements", []).append({
            "phase": phase, "details": details, "ts": int(time.time()*1000)
        })
        _save_index(idx)
