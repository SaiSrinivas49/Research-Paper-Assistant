
import argparse

from app.config import DATA_DIR, VECTOR_STORE_DIR
from app.rag_pipeline import build_vector_store, load_vector_store, answer_question


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about a research paper."
    )
    parser.add_argument(
        "--pdf",
        default=str(DATA_DIR / "sample.pdf"),
        help="Path to the PDF to index",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector index from the supplied PDF",
    )
    parser.add_argument(
        "--question",
        help="Question to ask about the document",
    )
    args = parser.parse_args()

    index_exists = (VECTOR_STORE_DIR / "index.faiss").exists()

    if args.rebuild or not index_exists:
        vector_store, chunk_count = build_vector_store(args.pdf)
        print(f"Indexed {chunk_count} text chunks.")
    else:
        vector_store = load_vector_store()
        print("Loaded the saved vector index.")

    question = args.question or input("\nAsk a question about the paper: ")
    result = answer_question(question, vector_store)

    print("\nAnswer:\n")
    print(result["answer"])

    if result["sources"]:
        print("\nRetrieved sources:")
        for source in result["sources"]:
            print(f"- {source['file']}, page {source['page']}")


if __name__ == "__main__":
    main()
