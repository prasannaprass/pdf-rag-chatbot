"""Core RAG (Retrieval-Augmented Generation) logic for the PDF chatbot.

Pipeline: PDF -> text -> chunks -> embeddings -> similarity search -> LLM answer.
"""
import os

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"          # small, free, runs on CPU
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")


def load_pdf_text(file) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page of the PDF."""
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Split page text into overlapping chunks, cutting at word boundaries."""
    chunks = []
    for page_no, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
            nxt = text.find(" ", start)
            if nxt != -1 and nxt < end:
                start = nxt + 1
    return chunks


class VectorStore:
    """Tiny in-memory vector store using cosine similarity (numpy only)."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[dict] = []
        self.embeddings = None

    def add(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        # normalised vectors -> dot product equals cosine similarity
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Ask the LLM to answer ONLY from the retrieved chunks."""
    context_text = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in contexts)
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:  # fallback so the app still demos without an API key
        return ("(No ANTHROPIC_API_KEY set - showing the most relevant passages)\n\n"
                + context_text)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=600,
        system=("You answer questions using ONLY the provided document excerpts. "
                "If the answer is not in the excerpts, say you could not find it. "
                "Mention page numbers when useful."),
        messages=[{
            "role": "user",
            "content": f"Document excerpts:\n{context_text}\n\nQuestion: {question}",
        }],
    )
    return response.content[0].text
