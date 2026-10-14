
from app.rag_pipeline import extract_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "data/sample.pdf"

pages = extract_pdf(pdf_path)

print("Pages with extracted text:", len(pages))
print("First page metadata:", pages[0].metadata)
print("\nExtracted text preview:\n")
print(pages[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)

print("\nTotal chunks:", len(chunks))
print("First chunk metadata:", chunks[0].metadata)
