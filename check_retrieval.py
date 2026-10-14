
from app.rag_pipeline import load_vector_store

vector_store = load_vector_store()

question = "What is the main objective of this paper?"
results = vector_store.similarity_search(question, k=4)

print(f"Retrieved {len(results)} chunks.\n")

for i, doc in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:400])
    print()
