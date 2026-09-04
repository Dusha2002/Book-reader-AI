# Book Reader AI

MVP engine for context-aware literary translation of ebooks. It accepts **FB2** and **EPUB**, builds a book-level translation bible, translates in batches, runs literary editing and optional bilingual QA, then rebuilds the ebook.

## Why this is different from plain machine translation

The pipeline keeps a shared `BookMemory` for the whole book: narrative voice, rhythm, dialogue rules, characters, glossary and a rolling plot summary. In `high` mode each batch passes through three roles:

1. **Translator** — faithful EN → RU translation.
2. **Literary editor** — removes calques and adapts idioms/wordplay without rewriting the author.
3. **Bilingual QA** — checks the Russian candidate against the English original for omissions, additions and consistency errors.

Progress is checkpointed in `.bookai-cache`, so an interrupted book can resume without paying for completed batches again.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Set these environment variables:

```bash
export BOOKAI_API_KEY='...'
export BOOKAI_BASE_URL='https://api.deepseek.com'
export BOOKAI_MODEL='deepseek-chat'
```

## CLI

```bash
bookai translate my-book.fb2 --mode high
bookai translate my-book.epub -o my-book.ru.epub --mode standard
```

Modes:
- `fast` — translation only; cheapest.
- `standard` — translation + literary edit.
- `high` — translation + literary edit + bilingual QA.

## HTTP API

```bash
uvicorn bookai.api:app --reload
```

Upload a book with `POST /books` (`multipart/form-data`: `file`, optional `mode`), poll `GET /jobs/{job_id}`, then download via `GET /jobs/{job_id}/download`.

## Current scope

Implemented: FB2, EPUB, restartable translation cache, book memory, literary editing, bilingual QA, CLI, HTTP API and tests.

Next useful steps: DOCX/TXT import, PDF text/OCR pipeline, persistent job queue (Redis/Postgres), web UI, cost/token accounting, smarter glossary updates after every chapter, and optional side-by-side original/translation reader.

## Copyright

Use the tool for books you are legally allowed to translate/process. A translation can itself be a protected derivative work; this project does not grant redistribution rights for copyrighted books.
