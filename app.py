"""Streamlit UI for the PDF Question-Answering chatbot (RAG).

Run:  streamlit run app.py
"""
import streamlit as st

from rag import VectorStore, chunk_pages, generate_answer, load_pdf_text

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF (RAG)")

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.messages = []

uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded and st.sidebar.button("Process PDF"):
    with st.spinner("Reading, chunking and embedding the PDF..."):
        pages = load_pdf_text(uploaded)
        chunks = chunk_pages(pages)
        store = VectorStore()
        store.add(chunks)
        st.session_state.store = store
        st.session_state.messages = []
    st.sidebar.success(f"Ready! {len(chunks)} chunks indexed.")

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your PDF")
if question:
    if st.session_state.store is None:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                hits = st.session_state.store.search(question, k=4)
                answer = generate_answer(question, hits)
            st.markdown(answer)
            with st.expander("Sources"):
                for h in hits:
                    st.caption(f"Page {h['page']} (similarity {h['score']:.2f})")
                    st.write(h["text"])
        st.session_state.messages.append({"role": "assistant", "content": answer})
