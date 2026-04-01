import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Smart RAG Chatbot")
st.write("📄 Upload documents and ask questions intelligently")

# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------- FUNCTIONS ----------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

def search(query, index, chunks, k=3):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Section")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

# ---------------- MAIN APP ----------------
if uploaded_files:
    with st.spinner("📊 Processing documents..."):
        all_text = ""
        for file in uploaded_files:
            all_text += extract_text(file)

        chunks = split_text(all_text)
        index = build_index(chunks)

    st.success("✅ Documents processed successfully!")

    query = st.text_input("💬 Ask your question:")

    if query:
        with st.spinner("🔍 Searching..."):
            results = search(query, index, chunks)

        st.markdown("### 🧠 Answer")
        st.success(results[0])  # Top relevant chunk

        with st.expander("📚 View Sources"):
            for r in results:
                st.write("🔹", r[:200], "...")

else:
    st.info("⬅️ Upload PDF files from sidebar to begin")
