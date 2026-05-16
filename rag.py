"""
rag.py — retrieval-augmented generation pipeline.

Kept separate from app.py so the core logic is testable without
needing to spin up a Streamlit server.
"""

import os
import re
import hashlib
import io
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4
LLM_MODEL     = "llama-3.3-70b-versatile"   # updated — llama3-8b-8192 was decommissioned

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided document excerpts.

Rules:
- Only use information from the excerpts below. Do not use outside knowledge.
- If the answer isn't in the excerpts, say so clearly — don't make things up.
- Keep answers concise and direct.
- When helpful, quote short phrases from the source.
"""


# ── PDF parsing ───────────────────────────────────────────────────────────────

def extract_text(pdf_file) -> str:
    """Pull all text from a PDF file-like object."""
    reader = PdfReader(pdf_file)
    pages  = [page.extract_text() or "" for page in reader.pages]
    text   = "\n\n".join(pages)
    text   = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]


# ── vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client   = chromadb.Client(Settings(anonymized_telemetry=False))

    def _file_hash(self, raw_bytes: bytes) -> str:
        return hashlib.md5(raw_bytes).hexdigest()[:12]

    def index_document(self, pdf_file, filename: str) -> tuple[str, int]:
        """
        Extract → chunk → embed → store.

        Fix applied: read raw bytes for hashing first, then seek(0)
        so PdfReader can still read from the beginning of the stream.
        """
        raw = pdf_file.read()       # get raw bytes for hashing
        doc_id = self._file_hash(raw)

        # get_or_create avoids the "collection already exists" crash
        # when Streamlit reruns the script on any interaction
        col = self.client.get_or_create_collection(doc_id)

        if col.count() > 0:
            # already indexed this exact file, nothing to do
            return doc_id, col.count()

        pdf_file.seek(0)            # rewind so PdfReader starts from the beginning
        text   = extract_text(pdf_file)
        chunks = chunk_text(text)

        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()

        col.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
        )

        return doc_id, len(chunks)

    def retrieve(self, doc_id: str, query: str, k: int = TOP_K) -> list[str]:
        col = self.client.get_or_create_collection(doc_id)
        if col.count() == 0:
            raise ValueError(f"No index found for doc '{doc_id}'. Upload it first.")

        query_vec = self.embedder.encode([query]).tolist()
        results   = col.query(query_embeddings=query_vec, n_results=min(k, col.count()))
        return results["documents"][0]


# ── chatbot ───────────────────────────────────────────────────────────────────

class RAGChatbot:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Make sure your .env file exists and contains GROQ_API_KEY=gsk_..."
            )
        self.llm    = Groq(api_key=api_key)
        self.store  = VectorStore()
        self.doc_id: str | None = None
        self.history: list[dict] = []

    def load_document(self, pdf_file, filename: str) -> int:
        self.doc_id, n_chunks = self.store.index_document(pdf_file, filename)
        self.history = []
        return n_chunks

    def ask(self, question: str) -> tuple[str, list[str]]:
        if not self.doc_id:
            return "Please upload a PDF first.", []

        chunks  = self.store.retrieve(self.doc_id, question)
        context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks))

        user_message = f"Document excerpts:\n{context}\n\nQuestion: {question}"
        self.history.append({"role": "user", "content": user_message})

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.history,
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})

        if len(self.history) > 20:
            self.history = self.history[-20:]

        return answer, chunks
