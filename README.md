# 📄 PDF Question-Answering Chatbot (RAG)

![App screenshot](App%20screenshot.png)

Upload any PDF and ask questions about it. The app retrieves the most relevant
passages with embeddings and uses an LLM to write an answer grounded in the document.

## How it works (RAG pipeline)
1. **Load**: extract text from each PDF page (`pypdf`)
2. **Chunk**: split text into overlapping 800-character pieces
3. **Embed**: convert chunks to vectors (`sentence-transformers`, all-MiniLM-L6-v2)
4. **Retrieve**: embed the question and find the top-4 chunks by cosine similarity
5. **Generate**: send the chunks plus the question to an LLM, which answers only from them
6. **Cite**: the UI shows the source pages used

## Tech stack
Python, Streamlit, sentence-transformers, NumPy, pypdf, Anthropic API

## Setup
```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-key"   # Windows: set ANTHROPIC_API_KEY=your-key
streamlit run app.py
```
Without an API key the app still runs and shows the retrieved passages.

## Project structure
```
pdf-rag-chatbot/
├── app.py            # Streamlit chat UI
├── rag.py            # chunking, embeddings, retrieval, LLM call
├── requirements.txt
└── README.md
```

## Possible improvements
- Use FAISS or ChromaDB for large documents
- Support multiple PDFs and conversation memory
- Add evaluation of answer quality
- Deploy on Streamlit Community Cloud

## Author
Lakshmi Prasanna Sarvepalli
