import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag_pipeline import build_vector_store, answer_question
import hashlib

app = FastAPI(title="Research Paper Assistant API")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

current_file_hash = None
current_chunk_count = 0

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
    global current_file_hash, current_chunk_count

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        safe_filename = Path(file.filename).name
        file_hash = hashlib.sha256(content).hexdigest()

        # Reuse the active index for an identical PDF.
        if (
            vector_store is not None
            and file_hash == current_file_hash
            and safe_filename == current_filename
        ):
            return {
                "message": "PDF already indexed. Reusing existing index.",
                "filename": current_filename,
                "chunks": current_chunk_count,
                "reused_index": True,
            }

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = UPLOAD_DIR / safe_filename
        pdf_path.write_bytes(content)

        # Build first; update active state only on success.
        new_store, chunk_count = build_vector_store(pdf_path)

        vector_store = new_store
        current_filename = safe_filename
        current_file_hash = file_hash
        current_chunk_count = chunk_count

        return {
            "message": "PDF indexed successfully.",
            "filename": current_filename,
            "chunks": chunk_count,
            "reused_index": False,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {repr(exc)}",
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
