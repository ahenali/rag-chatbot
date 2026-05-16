# Doc Chatbot — RAG with Groq + ChromaDB

Upload any PDF and ask questions about it. Built with Retrieval-Augmented Generation (RAG) — the model only answers from your document, so it can't hallucinate facts that aren't there.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Groq](https://img.shields.io/badge/LLM-Groq_Llama3.3-orange) ![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)

---

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| LLM | Groq (Llama 3.3 70B) | Free API, fastest inference available |
| Embeddings | all-MiniLM-L6-v2 | Runs locally, no API cost |
| Vector DB | ChromaDB | Simple, in-process, no server needed |
| PDF parsing | pypdf | Lightweight, handles most PDFs |
| UI | Streamlit | Fast to build, easy to deploy |

---

## Setup

**1. Get a free Groq API key**
Go to [console.groq.com](https://console.groq.com) → API Keys → Create. No credit card needed. Key starts with `gsk_`.

**2. Clone and install**
```bash
git clone https://github.com/YOUR-USERNAME/rag-chatbot.git
cd rag-chatbot
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API key**
```bash
cp .env.example .env
```
Open `.env` and replace `gsk_your_key_here` with your actual key.

**4. Run**
```bash
streamlit run app.py
```

Upload a PDF in the sidebar and start asking questions.

---

## How it works

### Indexing (once per document)
1. `pypdf` extracts raw text from the PDF
2. Text is split into 500-character chunks with 50-char overlap
3. `sentence-transformers` converts each chunk into a 384-dimensional vector
4. Vectors are stored in ChromaDB with the original text attached

### Answering (every question)
1. The question is embedded with the same model
2. ChromaDB finds the top-4 most similar chunks via cosine similarity
3. Those chunks are injected into the prompt alongside the question
4. Groq (Llama 3.3 70B) generates an answer grounded in those chunks
5. Source excerpts are shown so you can verify the answer yourself

---

## Project structure

```
rag-chatbot/
├── app.py          # Streamlit UI and session state
├── rag.py          # PDF parsing, chunking, embedding, retrieval, generation
├── requirements.txt
├── .env            # your API key — never commit this
├── .env.example    # safe to commit — shows what .env should look like
├── .gitignore
└── README.md
```

---

## What I'd add next

- Persistent ChromaDB storage so the index survives app restarts
- Multi-document support — ask across several PDFs at once
- Streaming responses so answers appear word-by-word
- Hybrid search (keyword + semantic) for better retrieval on technical docs

---

*Part of my AI & Data Science portfolio.*
