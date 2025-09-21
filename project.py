# ---------- Import necessary libraries ----------
import streamlit as st  # Streamlit for web app
from PyPDF2 import PdfReader  # PDF reader
from langchain.vectorstores import FAISS  # Vector DB for similarity search
from langchain.embeddings import HuggingFaceEmbeddings  # Text embeddings using Hugging Face
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting large text into chunks
from transformers import pipeline  # Hugging Face pipeline (summarization)

# ---------- Model Configuration ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and small embedding model
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # Lightweight, fast summarizer

# ---------- Load PDF text ----------
def load_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ---------- Split long text into smaller overlapping chunks ----------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# ---------- Create FAISS Vector Store ----------
@st.cache_resource
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts(chunks, embeddings)

# ---------- Load Hugging Face Summarizer ----------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model=SUMMARIZER_MODEL)

# ---------- Generate detailed summary ----------
def generate_summary(summarizer, selected_chunks):
    summaries = []
    for chunk in selected_chunks:
        if len(chunk.strip()) > 0:
            summary = summarizer(chunk, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
            summaries.append(summary)
    return "\n\n".join(summaries)

# ---------- Streamlit App Configuration ----------
st.set_page_config(page_title="Legal Case Summary Generator", layout="centered")

# ---------- Modern & Colorful Styling ----------
st.markdown("""
    <style>
        html, body, .main {
            background-color: #3b0a57;
            color: #ffffff;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #f5a623;
            text-align: center;
            margin-bottom: 30px;
        }
        .subtitle {
            font-size: 18px;
            color: #e0d7f9;
            text-align: center;
        }
        .summary-box {
            background-color: #5e239d;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #f5a623;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #f5a623;
            color: #000;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ffcf66;
            color: #000;
        }
        .stSelectbox label {
            color: #ffffff;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="title">📚 Legal Case Summary Generator</div>', unsafe_allow_html=True)

# ---------- File Uploader ----------
uploaded_file = st.file_uploader("📄 Upload a Legal Case PDF", type="pdf")

# ---------- When a file is uploaded ----------
if uploaded_file:
    text = load_pdf(uploaded_file)  # Load the PDF text
    chunks = split_text(text)  # Split it into chunks

    # ---------- Generate Summary Button ----------
    if st.button("🔍 Generate Summary"):
        with st.spinner("Analyzing and summarizing..."):
            db = create_vector_store(chunks)  # Build vector DB
            docs = db.similarity_search(text, k=3)  # Get top 3 matching chunks
            selected_chunks = [doc.page_content for doc in docs]  # Extract text from results

            summarizer = load_summarizer()  # Load summarizer
            summary = generate_summary(summarizer, selected_chunks)  # Get summary

        # ---------- Display the Summary ----------
        st.success("✅ Summary Generated!")
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.subheader("📝 Summary:")
        st.write(summary)
        st.markdown('</div>', unsafe_allow_html=True)
