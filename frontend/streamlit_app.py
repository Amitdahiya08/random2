import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Intelligent Doc Summarization & Q&A", layout="wide")
st.title("📄 Intelligent Document Summarization & Q&A")

with st.sidebar:
    st.header("Upload Document")
    file = st.file_uploader("PDF / DOCX / HTML / TXT", type=["pdf","docx","html","htm","txt"])
    if st.button("Ingest", type="primary") and file:
        with st.spinner("Ingesting..."):
            resp = requests.post(f"{API_BASE}/ingest", files={"file": (file.name, file.getvalue())})
        if resp.ok:
            st.session_state["current_doc"] = resp.json()
            st.success(f"Document ingested. doc_id={resp.json()['doc_id']}")
        else:
            st.error(resp.text)

# Summary + entity review
doc = st.session_state.get("current_doc")
if doc:
    st.subheader("📝 Summary (editable)")
    summary = st.text_area("Summary", value=doc["summary"], height=220)
    st.subheader("🏷️ Entities (one per line)")
    ent_text = "\n".join(doc["entities"])
    ents = st.text_area("Entities", value=ent_text, height=150)
    if st.button("Save Edits"):
        payload = {"summary": summary, "entities": [e.strip() for e in ents.splitlines() if e.strip()]}
        r = requests.put(f"{API_BASE}/summary/{doc['doc_id']}", json=payload)
        if r.ok:
            st.success("Saved!")
            doc["summary"] = summary; doc["entities"] = payload["entities"]
        else:
            st.error(r.text)

# Q&A
st.subheader("🔎 Ask Questions")
q = st.text_input("Your question")
doc_scope = st.radio("Scope", ["Current document" if doc else "—", "All documents"])
if st.button("Ask") and q:
    payload = {"doc_id": (doc["doc_id"] if doc and doc_scope.startswith("Current") else None), "question": q}
    with st.spinner("Thinking..."):
        r = requests.post(f"{API_BASE}/qa", json=payload)
    if r.ok:
        data = r.json()
        st.markdown("**Answer:**")
        st.write(data["answer"])
        with st.expander("Show retrieved context"):
            for c in data.get("contexts", []):
                st.code(c[:4000])
    else:
        st.error(r.text)
