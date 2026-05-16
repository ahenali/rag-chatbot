import io
import os
import streamlit as st
from dotenv import load_dotenv
from rag import RAGChatbot

load_dotenv()

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Doc Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat with your documents")
st.markdown(
    "Upload a PDF, then ask questions about it. "
    "Answers come only from your document — the model won't make things up."
)
st.divider()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Setup")

    # API key comes from .env — no more typing it into the sidebar
    api_key_loaded = bool(os.getenv("GROQ_API_KEY", "").strip())
    if api_key_loaded:
        st.success("✅ Groq API key loaded from .env")
    else:
        st.error(
            "❌ GROQ_API_KEY not found.\n\n"
            "Create a `.env` file in the project folder with:\n"
            "```\nGROQ_API_KEY=gsk_your_key_here\n```\n\n"
            "Get a free key at [console.groq.com](https://console.groq.com)"
        )

    st.divider()

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded and api_key_loaded:
        # only re-index if it's a different file from last time
        if st.session_state.get("loaded_file") != uploaded.name:
            with st.spinner("Indexing document — this takes a few seconds..."):
                try:
                    bot = RAGChatbot()
                    pdf_bytes = io.BytesIO(uploaded.read())
                    n_chunks  = bot.load_document(pdf_bytes, uploaded.name)
                    st.session_state.chatbot     = bot
                    st.session_state.loaded_file = uploaded.name
                    st.session_state.messages    = []
                    st.success(f"✅ Indexed **{n_chunks}** chunks from **{uploaded.name}**")
                except Exception as e:
                    st.error(f"Failed to index document: {e}")

    st.divider()
    st.markdown("**Model:** Llama 3.3 70B via Groq")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2 (local)")
    st.markdown("**Vector DB:** ChromaDB (in-memory)")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        if "chatbot" in st.session_state:
            st.session_state.chatbot.history = []
        st.rerun()

# ── main chat area ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Source excerpts used"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**Excerpt {i}**")
                    st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))

# gate the input behind having both key and doc ready
if not api_key_loaded:
    st.info("👈 Add your Groq API key to a `.env` file to get started.")
elif not uploaded:
    st.info("👈 Upload a PDF to start chatting with it.")
elif "chatbot" not in st.session_state:
    st.info("Something went wrong during indexing. Try re-uploading the PDF.")
else:
    question = st.chat_input("Ask anything about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    bot: RAGChatbot = st.session_state.chatbot
                    answer, sources = bot.ask(question)
                except Exception as e:
                    answer  = f"Error: {e}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("📎 Source excerpts used"):
                    for i, chunk in enumerate(sources, 1):
                        st.markdown(f"**Excerpt {i}**")
                        st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # show prompt ideas if it's a fresh conversation
    if not st.session_state.messages:
        st.markdown("""
        **Try asking things like:**
        - *"What is the main topic of this document?"*
        - *"Summarise the key points"*
        - *"What does it say about X?"*
        - *"List all the recommendations mentioned"*
        """)
