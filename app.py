"""
Text_summerizer – Intelligent Document Summarizer & Q&A System
----------------------------------------------------
Streamlit app that:
 - Uploads a PDF
 - Extracts text using PyMuPDF (fitz)
 - Displays extracted text preview
 - Summarizes (facebook/bart-large-cnn) with chunking (max 1000 words)
 - Answers user questions by retrieving top-2 relevant chunks using
   embeddings (sentence-transformers/all-MiniLM-L6-v2) and answering with
   deepset/roberta-base-squad2
 - Shows answer, confidence score, similarity, and source paragraph
"""

import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
import math
import numpy as np
from typing import List
import textwrap
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    text_chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                text_chunks.append(text)
    full_text = "\n\n".join(text_chunks)
    return full_text


def chunk_text_by_words(text: str, max_words: int = 1000) -> List[str]:
    """Chunk text into pieces each with ≤ max_words words."""
    words = text.split()
    if len(words) == 0:
        return []
    if len(words) <= max_words:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start = end
    return chunks


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------
# Cached model loaders
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource(show_spinner=False)
def load_qa_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("question-answering",
                    model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2",
                    device=device)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Text_summerizer – Intelligent Document Summarizer & Q&A System", layout="wide")

st.title("🤖 Text_summerizer")
st.subheader("Intelligent Document Summarizer & Q&A System")

# Sidebar info/help
with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
1. Upload a PDF.
2. Wait for extraction and summarization.
3. Review the Extracted Text Preview.
4. Read the Summary.
5. Ask a question — Text_summerizer retrieves the most relevant paragraph(s) and answers it.

**Models used (all free):**
- `facebook/bart-large-cnn` (Summarization)  
- `sentence-transformers/all-MiniLM-L6-v2` (Embeddings)  
- `deepset/roberta-base-squad2` (Q&A)
        """
    )

# Sections
st.header("Upload PDF")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

preview_container = st.container()
summary_container = st.container()
qa_container = st.container()

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    # Extract text
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(pdf_bytes)
    if not extracted_text.strip():
        st.error("No text could be extracted from the uploaded PDF.")
    else:
        st.success("✅ Text extracted successfully.")

        # Text preview
        with preview_container:
            st.subheader("Extracted Text Preview")
            st.text_area("Extracted Text (Preview)", value=extracted_text, height=300)

        # Chunking
        with st.spinner("Chunking document..."):
            chunks = chunk_text_by_words(extracted_text, max_words=1000)
        st.info(f"📄 Document split into {len(chunks)} chunk(s) (≤1000 words each).")

        # Load models
        with st.spinner("Loading models (first time may take a moment)..."):
            summarizer = load_summarizer_model()
            embedding_model = load_embedding_model()
            qa_pipeline = load_qa_model()

        # Summarization
        with summary_container:
            st.subheader("Summary")
            st.markdown("Summarizing document. Large documents are summarized per chunk and then refined.")

            final_summary = ""
            if len(chunks) == 0:
                st.warning("No chunks generated from document.")
            else:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    progress_text.text(f"Summarizing chunk {i+1} of {len(chunks)}...")
                    try:
                        summary_out = summarizer(chunk, max_length=300, min_length=80, do_sample=False)
                        text_summary = summary_out[0]["summary_text"].strip()
                    except Exception:
                        text_summary = " ".join(chunk.split()[:200]) + "..."
                    chunk_summaries.append(text_summary)
                    progress_bar.progress(int(((i + 1) / len(chunks)) * 100))

                progress_text.text("Combining chunk summaries...")
                progress_bar.empty()
                combined = "\n\n".join(chunk_summaries)
                if len(chunk_summaries) > 1:
                    with st.spinner("Creating final combined summary..."):
                        try:
                            combined_summary_out = summarizer(combined, max_length=400, min_length=100, do_sample=False)
                            final_summary = combined_summary_out[0]["summary_text"].strip()
                        except Exception:
                            final_summary = combined
                else:
                    final_summary = combined
                st.success("✅ Summary generated.")

            st.markdown("**Final Summary:**")
            st.text_area("Document Summary", value=final_summary, height=250)

        # Precompute embeddings
        with st.spinner("Computing embeddings for retrieval..."):
            chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        # Q&A
        with qa_container:
            st.subheader("Ask a Question")
            st.markdown("Text_summerizer will retrieve the most relevant text and answer using extractive QA.")
            question = st.text_input("Your question", placeholder="E.g., What is Artificial Intelligence?")

            if st.button("Get Answer"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Retrieving and answering..."):
                        q_emb = embedding_model.encode([question], convert_to_numpy=True)[0]
                        sims = [compute_cosine_similarity(q_emb, emb) for emb in chunk_embeddings]

                        # Get top 2 chunks instead of just one
                        top_k = 2
                        top_indices = np.argsort(sims)[-top_k:][::-1]
                        best_score = float(max(sims))
                        best_chunk = "\n\n".join([chunks[i] for i in top_indices])

                        # Refine to best paragraph
                        paras = [p.strip() for p in best_chunk.split("\n\n") if p.strip()]
                        if len(paras) <= 1:
                            source_para = best_chunk
                        else:
                            para_embs = embedding_model.encode(paras, convert_to_numpy=True, show_progress_bar=False)
                            para_sims = [compute_cosine_similarity(q_emb, pe) for pe in para_embs]
                            para_best_idx = int(np.argmax(para_sims))
                            source_para = paras[para_best_idx]

                        # QA model
                        try:
                            qa_input = {"question": question, "context": source_para}
                            qa_result = qa_pipeline(qa_input)
                            answer_text = qa_result.get("answer", "").strip()
                            score = qa_result.get("score", 0.0)
                        except Exception:
                            answer_text, score = "", 0.0

                    if not answer_text:
                        st.info("No confident extractive answer found. Showing most relevant paragraph.")
                        st.markdown("**Most Relevant Paragraph (Source):**")
                        st.write(textwrap.shorten(source_para, width=2000, placeholder="..."))
                    else:
                        st.success("✅ Answer generated.")
                        st.markdown("**Answer:**")
                        st.write(answer_text)
                        st.progress(min(score, 1.0))
                        st.markdown(f"**Confidence: {score:.3f} | Retrieval similarity: {best_score:.3f}**")
                        st.markdown("**Source Paragraph:**")
                        st.write(source_para)

else:
    st.info("Upload a PDF file to get started.")

# Footer
st.write("---")
st.caption("Built by Mehul with free Hugging Face models – facebook/bart-large-cnn, sentence-transformers/all-MiniLM-L6-v2, deepset/roberta-base-squad2.")
