# Book Reader AI

A selective multi-model **EN → RU literary translation harness** for **FB2, EPUB, DOCX and TXT**. Instead of sending the whole book through several expensive LLM passes, the engine routes work by difficulty.

## Default cascade

```text
Book
  ↓
Style / terminology analysis        DeepSeek V4 Flash
  ↓
Base translation                    DeepSeek V4 Flash (or local MADLAD-400-3B)
  ↓
Quality Gate                        DeepSeek V4 Flash
  ├─ clean ───────────────────────→ keep translation
  ├─ medium issue → selective edit  Qwen3.8 Flash
  └─ hard issue   → selective edit  Qwen3.8 Flash
                       ↓ (Literary mode only)
                    senior pass     DeepSeek V4 Pro
  ↓
Continuity memory                    DeepSeek V4 Flash
  ↓
Rebuilt FB2 / EPUB / DOCX / TXT
```

The important optimization is that **Qwen3.8 Flash and V4 Pro do not read/rewrite the whole book**. They only receive passages flagged by the gate. The quality gate itself returns tiny JSON issue lists instead of another full translation.

## Why these models

Default OpenRouter role slugs:

- `deepseek/deepseek-v4-flash-0731` — high-volume translation, style analysis, gate and memory.
- `qwen/qwen3.8-flash` — selective literary editor for awkward/ambiguous passages.
- `deepseek/deepseek-v4-pro-0813` — rare hard cases in `literary` mode only.

You can replace any role with another OpenAI-compatible model through environment variables without changing the pipeline.

## Modes

- **fast** — base translation only. Maximum speed/minimum API use.
- **optimal** — base translation → gate → selective Qwen3.8 editing → continuity memory. Recommended.
- **literary** — Optimal plus a senior model pass for passages the gate marks `hard`.

Legacy `standard` and `high` names are accepted as aliases for `optimal` and `literary`.

## OpenRouter setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Set:

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
export BOOKAI_BASE_URL='https://openrouter.ai/api/v1'
export BOOKAI_TRANSLATOR_MODEL='deepseek/deepseek-v4-flash-0731'
export BOOKAI_GATE_MODEL='deepseek/deepseek-v4-flash-0731'
export BOOKAI_EDITOR_MODEL='qwen/qwen3.8-flash'
export BOOKAI_HARD_MODEL='deepseek/deepseek-v4-pro-0813'
export BOOKAI_REASONING='none'
```

`OPENROUTER_API_KEY` is enough for every model role.

## Optional local MADLAD base translator

The API cascade remains the default because it is simple and already extremely cheap. If you have a suitable GPU and want to remove API cost from the bulk translation pass:

```bash
pip install -e '.[local]'
export BOOKAI_TRANSLATOR_BACKEND='madlad'
export BOOKAI_MADLAD_MODEL='google/madlad400-3b-mt'
```

MADLAD only performs the first draft. The quality gate and selective literary editor still protect quality. The weights are loaded lazily, so normal API installs do not pull PyTorch/Transformers.

## Context and book memory

Every API translation batch receives:

- stable style bible;
- glossary and character speech notes;
- rolling plot summary;
- previous 2 and next 2 passages as **context-only** text;
- chapter-aware batching.

After each batch, continuity memory is refreshed in `optimal` / `literary` modes. Progress and translations are checkpointed in `.bookai-cache`, including gate findings.

## Web UI

```bash
uvicorn bookai.api:app --reload
```

Open `http://127.0.0.1:8000`, drag in a book, choose Fast / Optimal / Literary, watch progress and download the rebuilt Russian file.

## CLI

```bash
bookai translate novel.fb2
bookai translate novel.epub --mode literary
bookai translate manuscript.docx --mode optimal
bookai translate story.txt --mode fast
```

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Current limitations

- PDF/OCR remains a separate future pipeline because layout and scanned pages require different handling.
- DOCX preserves paragraph styles but translated paragraphs can flatten multiple inline runs.
- The in-process job registry is for local MVP use; production should use a persistent queue/store.
- Automatic literary translation can still miss exceptional poetry, experimental typography or very dense wordplay. The cascade is designed to make those cases rare and visible, not pretend they do not exist.

## Copyright

Use the tool for books you are legally allowed to translate/process. Translation can itself be a protected derivative work; this project does not grant redistribution rights for copyrighted books.
