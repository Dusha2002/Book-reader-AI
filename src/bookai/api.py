from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .harness import TranslationHarness
from .parsers.base import SUPPORTED_SUFFIXES, load_book, save_book
from .pipeline import MODE_ALIASES, _cache_path, translate_book
from .web import PAGE

load_dotenv()

app = FastAPI(title="Book Reader AI", version="0.4.0")
WORK = Path(".bookai-work")
WORK.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _set_job(job_id: str, **values) -> None:
    with LOCK:
        JOBS.setdefault(job_id, {}).update(values)


def _publish_segment_delta(job_id: str, delta: dict[str, str], output: Path) -> None:
    if not delta:
        return
    with LOCK:
        state = JOBS.setdefault(job_id, {})
        revision = int(state.get("revision") or 0) + 1
        events = state.setdefault("segment_events", [])
        events.append({"revision": revision, "segments": dict(delta)})
        # A long job can revise a segment during literary refinement. ready_segments
        # counts unique translated ids, not event entries.
        latest = state.setdefault("translated_segments", {})
        latest.update(delta)
        state.update(
            revision=revision,
            ready_segments=len(latest),
            partial_ready=True,
            output=str(output),
        )


def _run(job_id: str, source: Path, output: Path, mode: str) -> None:
    try:
        harness = TranslationHarness.from_env()
        progressive_document = load_book(source)
        cache_dir = WORK / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        state_path = _cache_path(source, cache_dir, mode)
        seen: dict[str, str] = {}
        _set_job(
            job_id,
            status="running",
            phase="parsing",
            progress=1,
            translator=harness.translator.name,
            revision=0,
            ready_segments=0,
            partial_ready=False,
            segment_events=[],
            translated_segments={},
        )

        def sync_progressive_output() -> None:
            if not state_path.exists():
                return
            try:
                state = json.loads(state_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            translations = {
                str(sid): str(text)
                for sid, text in dict(state.get("translations") or {}).items()
                if isinstance(text, str) and text.strip()
            }
            delta = {sid: text for sid, text in translations.items() if seen.get(sid) != text}
            if not delta:
                return
            seen.update(delta)
            # FB2/EPUB/TXT/DOCX serializers already replace only ids present in
            # translations, so untranslated tail remains readable in the source language.
            save_book(progressive_document, translations, output)
            _publish_segment_delta(job_id, delta, output)

        def progress(event: dict) -> None:
            # translate_book persists accepted translations before its progress
            # callback. Reading that cache here makes every accepted batch visible
            # to the reader immediately instead of waiting for the whole book.
            sync_progressive_output()
            _set_job(job_id, **event)

        translate_book(source, output, harness, mode=mode, cache_dir=cache_dir, progress=progress)
        sync_progressive_output()
        _set_job(
            job_id,
            status="done",
            phase="done",
            progress=100,
            output=str(output),
            usage=harness.usage,
            partial_ready=True,
        )
    except Exception as exc:
        # Keep already translated/serialized fragments readable after a later error.
        _set_job(job_id, status="error", phase="error", error=str(exc))


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/health")
def health():
    return {"ok": True, "version": "0.4.0", "progressive_translation": True}


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
    _set_job(
        job_id,
        status="queued",
        phase="queued",
        progress=0,
        filename=file.filename or source.name,
        mode=mode,
        revision=0,
        ready_segments=0,
        partial_ready=False,
        segment_events=[],
        translated_segments={},
    )
    threading.Thread(target=_run, args=(job_id, source, output, mode), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "mode": mode, "progressive": True}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    with LOCK:
        state = JOBS.get(job_id)
        if state is None:
            raise HTTPException(404, "Unknown job")
        # Do not send the growing translation/event dictionaries with every status poll.
        public = {k: v for k, v in state.items() if k not in {"translated_segments", "segment_events"}}
        return public


@app.get("/jobs/{job_id}/segments")
def translated_segments(job_id: str, after: int = Query(0, ge=0)):
    """Return translation deltas newer than `after` for a live reader.

    The client keeps the returned revision and asks again with ?after=<revision>.
    Revisions can contain replacements of already translated ids after refinement;
    the reader should simply replace its current text for those ids.
    """
    with LOCK:
        state = JOBS.get(job_id)
        if state is None:
            raise HTTPException(404, "Unknown job")
        revision = int(state.get("revision") or 0)
        events = list(state.get("segment_events") or [])
        status = str(state.get("status") or "queued")
        ready = int(state.get("ready_segments") or 0)
    merged: dict[str, str] = {}
    for event in events:
        if int(event.get("revision") or 0) > after:
            merged.update(dict(event.get("segments") or {}))
    return {
        "revision": revision,
        "segments": merged,
        "ready_segments": ready,
        "status": status,
        "done": status == "done",
    }


@app.get("/jobs/{job_id}/partial-download")
def partial_download(job_id: str):
    """Download the latest mixed-language snapshot while translation is running."""
    with LOCK:
        state = dict(JOBS.get(job_id) or {})
    if not state:
        raise HTTPException(404, "Unknown job")
    output = Path(str(state.get("output") or ""))
    if not state.get("partial_ready") or not output.is_file():
        raise HTTPException(409, "No translated fragments are ready yet")
    source_name = state.get("filename") or output.name
    filename = f"{Path(source_name).stem}.partial{output.suffix}"
    return FileResponse(output, filename=filename)


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    with LOCK:
        state = dict(JOBS.get(job_id) or {})
    if state.get("status") != "done":
        raise HTTPException(409, "Book is not ready; use /partial-download while translation is running")
    source_name = state.get("filename") or Path(state["output"]).name
    suffix = Path(state["output"]).suffix
    filename = f"{Path(source_name).stem}.ru{suffix}"
    return FileResponse(state["output"], filename=filename)
