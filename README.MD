# AI-Powered Web Assistant by Himanshu Kabra

## Overview

This project is an end-to-end AI-powered web assistant that allows users to ask questions strictly based on the content of a given website. The system crawls the website, cleans and processes the text, generates embeddings, stores them in a vector database (FAISS), and uses an LLM to answer user queries grounded only in the indexed content.

The key focus is **accuracy, reliability, and hallucination prevention**, making it suitable for real-world company use cases.

---

## Key Features

* Website crawling and text extraction
* Intelligent text cleaning
* Chunking with overlap for better context retention
* Vector embeddings stored using FAISS
* Retrieval-Augmented Generation (RAG)
* Hallucination-safe responses
* Interactive Streamlit-based UI

---

## Architecture Flow

1. **User provides website URL**
2. **Crawler** fetches website content
3. **Text Cleaner** removes noise and irrelevant data
4. **Chunker** splits text into overlapping chunks
5. **Embedding Generator** converts chunks into vectors
6. **FAISS Vector Store** saves embeddings locally
7. **Retriever** finds relevant chunks for a query
8. **LLM** generates an answer strictly from retrieved content

---

## Folder Structure

```
project_root/
│
├── app.py
├── .env
│
├── crawler/
│   ├── web_loader.py
│   └── text_cleaner.py
│
├── processing/
│   ├── chunker.py
│   └── embeddings.py
│
├── qa/
│   └── qa_pipeline.py
│
├── vectorstore/
│   ├── faiss.index
│   └── metadata.pkl
│
└── requirements.txt
```

---

## Tech Stack

* Python 3.10+
* Streamlit (UI)
* OpenAI API (LLM + Embeddings)
* FAISS (Vector Database)
* dotenv (Environment Management)

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-url>
cd project_root
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## How It Prevents Hallucination

* The LLM only receives content retrieved from FAISS
* If no relevant chunks are found, the system responds:

  > "The answer is not available on the provided website."
* No external or prior knowledge is used

---

## Design Decisions

* **FAISS** chosen for fast similarity search and offline persistence
* **Chunk overlap** ensures context continuity
* **Session-based indexing** improves performance
* **Modular architecture** makes system scalable and maintainable

---

## Future Improvements

* Multi-page crawling with depth control
* Multi-website indexing
* Source citation per answer
* Authentication and role-based access

---

## Ideal Use Cases

* Company documentation assistant
* Product website Q&A bot
* Compliance-safe information retrieval
* Internal knowledge base assistant

---

## Author Notes

This project is built following industry-standard RAG architecture and is suitable for technical assessments, internships, and real-world deployments.
