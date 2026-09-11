from __future__ import annotations

import gc
import json
import os
import resource
import time
from pathlib import Path

import ctranslate2
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bookai.literary_context import locked_glossary_violations
from bookai.models import BookMemory, Segment
from bookai.parsers.base import load_book
from bookai.pipeline import _should_translate
from bookai.quality import batch_issues, hard_ids
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

SOURCE = Path(os.getenv('BOOKAI_SOURCE') or 'Devices_and_Desires.fb2')
MODEL_ID = os.getenv('BOOKAI_T5_MODEL') or 'utrobinmv/t5_translate_en_ru_zh_large_1024_v2'
MODEL_DIR = Path(os.getenv('BOOKAI_T5_CT2_DIR') or '.bookai-t5/t5-en-ru-zh-v2-ct2-int8')
OUT = Path('t5-mt-probe.json')
SAMPLES = Path('t5-mt-probe-samples.json')


def choose_sample(targets: list[Segment], count: int = 32) -> list[Segment]:
    buckets = [
        [s for s in targets if 80 <= len(s.text) < 220],
        [s for s in targets if 220 <= len(s.text) < 500],
        [s for s in targets if 500 <= len(s.text) < 900],
        [s for s in targets if 900 <= len(s.text) <= 1800],
    ]
    per = max(1, count // 4)
    picked: list[Segment] = []
    for rows in buckets:
        if not rows:
            continue
        if len(rows) <= per:
            picked.extend(rows)
        else:
            step = (len(rows) - 1) / max(1, per - 1)
            picked.extend(rows[round(i * step)] for i in range(per))
    order = {s.id: i for i, s in enumerate(targets)}
    unique = {s.id: s for s in picked}
    return sorted(unique.values(), key=lambda s: order[s.id])[:count]


def qa(sample: list[Segment], translated: dict[str, str], mem: BookMemory) -> dict:
    hard = set(hard_ids(batch_issues(sample, translated, mem)))
    glossary = locked_glossary_violations(sample, translated, REFERENCE_GLOSSARY_SEED)
    missing = {s.id for s in sample if not str(translated.get(s.id) or '').strip()}
    abnormal: set[str] = set()
    english_tail: set[str] = set()
    for s in sample:
        text = str(translated.get(s.id) or '')
        if not text:
            continue
        ratio = len(text) / max(1, len(s.text))
        if ratio < 0.42 or ratio > 1.75:
            abnormal.add(s.id)
        latin = sum(ch.isascii() and ch.isalpha() for ch in text)
        if latin / max(1, len(text)) > 0.18:
            english_tail.add(s.id)
    bad = hard | set(glossary) | missing | abnormal | english_tail
    return {
        'hard_qa': sorted(hard),
        'glossary_violations': glossary,
        'missing': sorted(missing),
        'abnormal_ratio': sorted(abnormal),
        'english_tail': sorted(english_tail),
        'bad_ids': sorted(bad),
        'qa_pass': len(sample) - len(bad),
        'qa_pass_rate': round((len(sample) - len(bad)) / max(1, len(sample)), 4),
    }


def main() -> None:
    doc = load_book(SOURCE)
    targets = [s for s in doc.segments if _should_translate(s.text)]
    sample = choose_sample(targets, max(16, int(os.getenv('BOOKAI_T5_PROBE_SEGMENTS') or '32')))
    total_chars = sum(len(s.text) for s in targets)
    mem = apply_reference_profile(BookMemory(glossary=dict(REFERENCE_GLOSSARY_SEED)))

    if not (MODEL_DIR / 'model.bin').exists():
        raise FileNotFoundError(f'CTranslate2 model not prepared: {MODEL_DIR}')

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    translator = ctranslate2.Translator(
        str(MODEL_DIR), device='cpu', compute_type='int8', inter_threads=1,
        intra_threads=max(1, os.cpu_count() or 2),
    )
    source = [tok.convert_ids_to_tokens(tok.encode('translate to ru: ' + s.text)) for s in sample]
    gc.collect()
    started = time.perf_counter()
    rows = translator.translate_batch(source, beam_size=1, max_batch_size=64, batch_type='examples')
    elapsed = max(0.001, time.perf_counter() - started)
    translated: dict[str, str] = {}
    errors: dict[str, str] = {}
    for s, row in zip(sample, rows):
        hyp = list(row.hypotheses[0]) if row.hypotheses else []
        text = tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).strip()
        if text:
            translated[s.id] = text
        else:
            errors[s.id] = 'empty translation'

    chars = sum(len(s.text) for s in sample if s.id in translated)
    cps = chars / elapsed
    report = {
        'model': MODEL_ID,
        'runtime': 'CTranslate2 INT8 CPU',
        'segments': len(sample),
        'sample_chars': chars,
        'elapsed_seconds': round(elapsed, 3),
        'source_chars_per_second': round(cps, 2),
        'estimated_full_minutes': None if cps <= 0 else round(total_chars / cps * 1.15 / 60.0, 2),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        'errors': errors,
        **qa(sample, translated, mem),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    SAMPLES.write_text(json.dumps({'source': {s.id: s.text for s in sample}, 'translation': translated}, ensure_ascii=False, indent=2), 'utf-8')
    print('[t5-mt-probe] ' + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
