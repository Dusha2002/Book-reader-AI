from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .harness import TranslationHarness
from .parsers.base import SUPPORTED_SUFFIXES
from .pipeline import MODE_ALIASES, translate_book
from .web import PAGE

load_dotenv()

app = FastAPI(title="Book Reader AI", version="0.3.0")
WORK = Path(".bookai-work")
WORK.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _set_job(job_id: str, **values) -> None:
    with LOCK:
        JOBS.setdefault(job_id, {}).update(values)


def _run(job_id: str, source: Path, output: Path, mode: str) -> None:
    try:
        harness = TranslationHarness.from_env()
        _set_job(job_id, status="running", phase="parsing", progress=1, translator=harness.translator.name)

        def progress(event: dict) -> None:
            _set_job(job_id, **event)

        translate_book(source, output, harness, mode=mode, cache_dir=WORK / "cache", progress=progress)
        _set_job(job_id, status="done", phase="done", progress=100, output=str(output), usage=harness.usage)
    except Exception as exc:
        _set_job(job_id, status="error", phase="error", error=str(exc))


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/health")
def health():
    return {"ok": True, "version": "0.3.0"}


@app.post("/books")
def upload_book(file: UploadFile = File(...), mode: str = Form("optimal")):
    suffix = Path(file.filename or "book").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    if mode not in {"fast", "optimal", "literary", "standard", "high"}:
        raise HTTPException(400, "mode must be fast, optimal, or literary")
    mode = MODE_ALIASES.get(mode, mode)
    job_id = uuid.uuid4().hex
    source = WORK / f"{job_id}{suffix}"
    output = WORK / f"{job_id}.ru{suffix}"
    with source.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    _set_job(job_id, status="queued", phase="queued", progress=0, filename=file.filename or source.name, mode=mode)
    threading.Thread(target=_run, args=(job_id, source, output, mode), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "mode": mode}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    with LOCK:
        state = JOBS.get(job_id)
        if state is None:
            raise HTTPException(404, "Unknown job")
        return dict(state)


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    with LOCK:
        state = dict(JOBS.get(job_id) or {})
    if state.get("status") != "done":
        raise HTTPException(409, "Book is not ready")
    source_name = state.get("filename") or Path(state["output"]).name
    suffix = Path(state["output"]).suffix
    filename = f"{Path(source_name).stem}.ru{suffix}"
    return FileResponse(state["output"], filename=filename)
