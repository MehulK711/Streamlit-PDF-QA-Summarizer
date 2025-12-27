# 🤖 Text Summerier – Intelligent Document Summarizer & Q&A System

A Streamlit-based NLP application that intelligently **summarizes PDF documents** and allows users to **ask questions** about the content using **free Hugging Face transformer models**.

---

## 🚀 Features

- 📄 **PDF Upload:** Upload and extract text from any PDF using **PyMuPDF (fitz)**.  
- 🧾 **Text Preview:** View extracted text in a scrollable box before processing.  
- ✂️ **Chunking:** Automatically splits long documents into 1000-word chunks for efficient summarization.  
- 🧠 **Summarization:** Generates concise document summaries using `facebook/bart-large-cnn`.  
- ❓ **Q&A System:** Ask questions about the document — the system retrieves the most relevant text using semantic similarity (`sentence-transformers/all-MiniLM-L6-v2`) and answers with `deepset/roberta-base-squad2`.  
- ⚡ **Real-time Feedback:** Displays progress spinners, success messages, and organized output sections for a smooth user experience.  
- 🎨 **Clean UI:** Built with Streamlit — minimal, modern, and responsive.  

---

## 🧩 Tech Stack

- **Frontend:** Streamlit  
- **Backend / NLP:** Hugging Face Transformers  
- **Models Used:**  
  - Summarization → `facebook/bart-large-cnn`  
  - Sentence Embeddings → `sentence-transformers/all-MiniLM-L6-v2`  
  - Question Answering → `deepset/roberta-base-squad2`  
- **Text Extraction:** PyMuPDF (fitz)  
- **Language:** Python  # Streamlit-PDF-QA-Summarizer
