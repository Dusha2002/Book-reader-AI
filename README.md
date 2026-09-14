# Book Reader AI

A selective **EN → RU literary book translation system** for FB2, EPUB, DOCX and TXT. The production goal is not to run every paragraph through more models; it is to keep the fast primary translator and spend specialist work only where source-grounded risk signals justify it.

## Production architecture

The supported production entrypoint is:

```bash
python scripts/production_literary_translation.py
```

Current production strategy: `literary-production-v1`.

```text
Book / parsed source segments
        ↓
Source-only book intelligence / terminology / continuity context
        ↓
GigaChat-3-Lightning
primary progressive literary translation
        ↓
Deterministic + source-grounded risk routing
  • terminology / names / numbers / chronology / register
  • polarity-scope risks
  • possible cross-segment contamination
        ↓ only for selected risky segments
GigaChat-3-Ultra
sparse semantic specialist / repair / final verification
        ↓
DeepSeek direct API
legacy emergency fallback only where the proven kernel still requires it
        ↓
Formatting + release guards + regression gate
        ↓
Russian book + evaluation artifacts
```

There is intentionally **no additional independent judge model** after Ultra. New deterministic risk signals route suspicious passages into the existing specialist instead of creating another translation layer.

## Production-path policy

The historical `scripts/chapter_reference_translation_v9*.py` files remain in the repository so experiments are reproducible. They are **legacy kernel/history, not new production entrypoints**.

Production changes must go through `LiteraryTranslationStrategy` in `scripts/production_literary_translation.py` rather than creating `v9ai`, `v9aj`, `v9ak`, and so on. `chapter_reference_translation_v9ah_ultra.py` is now only a compatibility shim that delegates to the stable production entrypoint.

The current facade still reuses the proven v9ah kernel internally while that kernel is gradually migrated into normal modules. This keeps translation behavior stable while stopping further version-chain growth.

## Quality and release state

A fully translated chapter can have one of two materially different states:

- `qa_passed` — final checks actually passed;
- `needs_review` — usable translated text exists, but a late polish/QA failure was waived only so a long resumable book run can continue.

A `needs_review` chapter is **never** promoted into `qa_passed_chapters`. A whole-book artifact containing such a chapter is labelled `needs_review`, and `hard_issues` is not falsely reported as zero.

## Source provenance and cache safety

Accepted draft translations now keep source provenance:

- source segment id;
- stable source hash;
- context segment ids;
- producing stage.

If the source text changes underneath the same segment id, the stale cached translation is invalidated on resume rather than silently reused.

## Contamination and polarity guards

`src/bookai/release_guards.py` contains conservative routing signals. They do not translate or judge text themselves; they only force suspicious passages into the existing Ultra specialist.

Current guards include:

- polarity constructions such as `I don't see why not`, which are easy to invert in Russian;
- suspicious target/source expansion;
- unusually high overlap with neighboring Russian paragraphs, which can indicate cross-segment carry-over.

The specialist prompt then verifies the candidate strictly against the current source segment. Neighboring passages may resolve context but are not allowed to contribute new events or propositions.

## Regression release gate

The production CI uses:

```bash
python scripts/evaluate_translation_regression.py \
  --dataset eval/literary_regression_v2.json \
  --fail-on-hard \
  --fail-on-objective \
  chapter-v9-*-translation-map.json
```

`literary_regression_v2.json` targets **150 real high-risk source cases** through curated failures plus risk-based sampling. Known historical failures are retained as permanent regressions, including:

- inverted `I don't see why not` polarity;
- cross-segment text contamination in Chapter Nine;
- prosecuting-role terminology;
- chronology/time markers;
- number relations, referents and formal register.

The release workflow now fails on curated hard regressions and conservative objective invariant failures instead of producing a report that CI ignores.

## Resilience

Long book runs are checkpointed and resumable. GigaChat PERS throttling is handled by retrying the **same request** with bounded exponential backoff instead of recursively splitting a throttled batch into more requests. Ultra specialist calls are serialized and use the same bounded 429 retry policy.

## Supported formats

- FB2
- EPUB
- DOCX
- TXT

PDF/OCR remains a separate pipeline because scanned/layout-heavy documents require different extraction and reconstruction logic.

## General CLI / API

The reusable package pipeline is still available through the normal CLI and API:

```bash
bookai translate novel.fb2
bookai translate novel.epub --mode literary
bookai translate manuscript.docx --mode optimal
bookai translate story.txt --mode fast
```

```bash
uvicorn bookai.api:app --reload
```

The dedicated production literary benchmark path is `scripts/production_literary_translation.py`.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

GitHub Actions also compiles `src`/`scripts`, runs the complete unit suite, executes the production chapter sample, then enforces the literary regression gate.

## Copyright

Use the tool for books you are legally allowed to translate/process. Translation can itself be a protected derivative work; this project does not grant redistribution rights for copyrighted books.
