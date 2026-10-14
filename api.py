import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag_pipeline import build_vector_store, answer_question

app = FastAPI(title="Research Paper Assistant API")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vector_store = None
current_filename = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, current_filename

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        safe_filename = Path(file.filename).name
        pdf_path = UPLOAD_DIR / safe_filename
        pdf_path.write_bytes(content)

        new_store, chunk_count = build_vector_store(pdf_path)

        vector_store = new_store
        current_filename = safe_filename

        return {
            "message": "PDF indexed successfully.",
            "filename": current_filename,
            "chunks": chunk_count,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {exc}",
        ) from exc


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking questions.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(question, vector_store)
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "filename": current_filename,
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {repr(exc)}",
        ) from exc
