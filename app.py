import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "collection" not in st.session_state:
    st.session_state.collection = None

# ── Functions ──
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def embed_text(text):
    response = client.embeddings.create(
        model="llama3-8b-8192",
        input=text
    )
    return response.data[0].embedding

def load_pdf(pdf_file, filename):
    with st.spinner("📖 Reading PDF..."):
        text = extract_text(pdf_file)

    with st.spinner("✂️ Splitting into chunks..."):
        chunks = chunk_text(text)

    with st.spinner("🧠 Building knowledge base..."):
        collection = chroma_client.get_or_create_collection(
            name=f"pdf_{uuid.uuid4().hex[:8]}"
        )
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )

    st.session_state.collection = collection
    st.session_state.pdf_loaded = True
    st.session_state.pdf_name = filename
    st.session_state.messages = []
    return len(chunks)

def search_chunks(query, n=4):
    results = st.session_state.collection.query(
        query_texts=[query],
        n_results=n
    )
    return results["documents"][0]

def answer_question(question):
    chunks = search_chunks(question)
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful assistant that answers questions about a document.
Use ONLY the context below to answer. If the answer isn't in the context, say so clearly.
Always mention which part of the document your answer comes from.

Context:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a precise document assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content, chunks

# ── UI ──
col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.markdown("## 📄 PDF Chatbot")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        if st.button("Load PDF"):
            chunks = load_pdf(uploaded_file, uploaded_file.name)
            st.success(f"✅ Loaded! {chunks} chunks created.")

    if st.session_state.pdf_loaded:
        st.markdown("---")
        st.markdown(f"**Loaded:** {st.session_state.pdf_name}")
        st.markdown("**Ready to answer questions!**")

        if st.button("Clear & Upload New"):
            st.session_state.pdf_loaded = False
            st.session_state.collection = None
            st.session_state.messages = []
            st.rerun()

with col_main:
    if not st.session_state.pdf_loaded:
        st.markdown("""
        ## Welcome to PDF Chatbot 📄

        **How it works:**
        1. Upload any PDF on the left
        2. Click "Load PDF" — it reads and indexes the document
        3. Ask any question — it finds the relevant section and answers

        **Try it with:**
        - A research paper
        - A textbook chapter
        - A legal document
        - Your resume
        - Any PDF you have!
        """)
    else:
        st.markdown(f"### Chatting with: {st.session_state.pdf_name}")
        st.markdown("---")

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")
                if msg.get("sources"):
                    with st.expander("📚 Source chunks used"):
                        for i, chunk in enumerate(msg["sources"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(f"> {chunk[:300]}...")
            st.markdown("---")

        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input("",
                placeholder="Ask anything about your PDF...",
                label_visibility="collapsed",
                key="question")
        with col2:
            ask = st.button("Ask →")

        if ask and question.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.spinner("🔍 Searching document..."):
                answer, sources = answer_question(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            st.rerun()