
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from functools import lru_cache

from app.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    VECTOR_STORE_DIR,
    get_api_key,
    GROQ_MODEL
)


def extract_pdf(pdf_path):
    """Extract text from each PDF page and retain its page number."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )

    if not documents:
        raise ValueError(
            "No extractable text found. The PDF may be scanned or empty."
        )

    return documents



@lru_cache(maxsize=1)
def get_embeddings():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )



def build_vector_store(pdf_path):
    """Extract, chunk, embed, and index a PDF."""
    pages = extract_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))

    return vector_store, len(chunks)


def load_vector_store():
    """Load an index previously created by this application."""
    index_file = VECTOR_STORE_DIR / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            "No saved index found. Build the vector store first."
        )

    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def get_llm():
    """Create and cache the LLM client."""
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=get_api_key(),
        temperature=0,
    )

def answer_question(question, vector_store):
    """Retrieve relevant chunks and generate a grounded answer."""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "I couldn't find relevant information in this document.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Source: {doc.metadata['source']}, "
        f"Page: {doc.metadata['page']}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a research paper assistant.
Answer the user's question using only the supplied document context.
Do not invent facts or use outside knowledge.
If the context does not contain enough information, clearly say so.
Cite supporting pages using [Page N] in your answer.
Distinguish the authors' claims from your own interpretation.""",
        ),
        (
            "human",
            "Document context:\n{context}\n\nQuestion: {question}",
        ),
    ])

    response = (prompt | get_llm()).invoke(
        {"context": context, "question": question}
    )

    # Keep source references unique and ordered by page.
    sources = sorted(
        {
            (doc.metadata["source"], doc.metadata["page"])
            for doc in retrieved_docs
        },
        key=lambda item: (item[0], item[1]),
    )

    return {
        "answer": response.content,
        "sources": [
            {"file": source, "page": page}
            for source, page in sources
        ],
    }
