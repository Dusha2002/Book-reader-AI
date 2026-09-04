from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .llm import OpenAICompatibleProvider
from .pipeline import translate_book

app = FastAPI(title="Book Reader AI", version="0.1.0")
WORK = Path(".bookai-work")
WORK.mkdir(exist_ok=True)
JOBS: dict[str, dict[str, str]] = {}


def _run(job_id: str, source: Path, output: Path, mode: str) -> None:
    try:
        JOBS[job_id] = {"status": "running"}
        translate_book(source, output, OpenAICompatibleProvider(), mode=mode, cache_dir=WORK / "cache")
        JOBS[job_id] = {"status": "done", "output": str(output)}
    except Exception as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/books")
def upload_book(background: BackgroundTasks, file: UploadFile = File(...), mode: str = Form("high")):
    suffix = Path(file.filename or "book").suffix.lower()
    if suffix not in {".fb2", ".epub"}:
        raise HTTPException(415, "MVP supports .fb2 and .epub")
    if mode not in {"fast", "standard", "high"}:
        raise HTTPException(400, "mode must be fast, standard, or high")
    job_id = uuid.uuid4().hex
    source = WORK / f"{job_id}{suffix}"
    output = WORK / f"{job_id}.ru{suffix}"
    with source.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    JOBS[job_id] = {"status": "queued"}
    background.add_task(_run, job_id, source, output, mode)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    state = JOBS.get(job_id)
    if not state or state.get("status") != "done":
        raise HTTPException(409, "Book is not ready")
    return FileResponse(state["output"], filename=Path(state["output"]).name)
