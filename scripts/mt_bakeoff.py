from __future__ import annotations

import gc
import json
import math
import os
import resource
import time
from pathlib import Path

from bookai.bulk_mt import OPUSMTBackend
from bookai.literary_context import locked_glossary_violations
from bookai.models import BookMemory, Segment
from bookai.parsers.base import load_book
from bookai.pipeline import _should_translate
from bookai.quality import batch_issues, hard_ids
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

SOURCE = Path(os.getenv("BOOKAI_SOURCE") or "Devices_and_Desires.fb2")
REPORT = Path("mt-bakeoff.json")
SAMPLES = Path("mt-bakeoff-samples.json")


def memory() -> BookMemory:
    return apply_reference_profile(BookMemory(glossary=dict(REFERENCE_GLOSSARY_SEED)))


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
    missing = {s.id for s in sample if not str(translated.get(s.id) or "").strip()}
    abnormal: set[str] = set()
    english_tail: set[str] = set()
    for s in sample:
        text = str(translated.get(s.id) or "")
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
        "hard_qa": sorted(hard),
        "glossary_violations": glossary,
        "missing": sorted(missing),
        "abnormal_ratio": sorted(abnormal),
        "english_tail": sorted(english_tail),
        "bad_ids": sorted(bad),
        "qa_pass": len(sample) - len(bad),
        "qa_pass_rate": round((len(sample) - len(bad)) / max(1, len(sample)), 4),
    }


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def run_model(name: str, fn, sample: list[Segment], mem: BookMemory, total_chars: int):
    gc.collect()
    started = time.perf_counter()
    translated, errors = fn(sample, mem)
    elapsed = max(0.001, time.perf_counter() - started)
    chars = sum(len(s.text) for s in sample if s.id in translated)
    cps = chars / elapsed
    result = {
        "model": name,
        "segments": len(sample),
        "translated": len(translated),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 3),
        "source_chars": chars,
        "source_chars_per_second": round(cps, 2),
        "estimated_full_minutes": None if cps <= 0 else round(total_chars / cps * 1.15 / 60.0, 2),
        "peak_rss_mb": rss_mb(),
        **qa(sample, translated, mem),
    }
    print("[mt-bakeoff] " + json.dumps({k: result[k] for k in ("model", "elapsed_seconds", "source_chars_per_second", "estimated_full_minutes", "peak_rss_mb", "qa_pass", "qa_pass_rate")}, ensure_ascii=False), flush=True)
    return result, translated


def opus(sample: list[Segment], mem: BookMemory):
    return OPUSMTBackend().translate_many(sample, mem)


def m2m100(sample: list[Segment], mem: BookMemory):
    del mem
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import M2M100Tokenizer

    model_dir = snapshot_download(repo_id="gn64/M2M100_418M_CTranslate2")
    tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ru")
    tr = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8", inter_threads=1, intra_threads=max(1, os.cpu_count() or 2))
    source = [tok.convert_ids_to_tokens(tok.encode(s.text)) for s in sample]
    rows = tr.translate_batch(source, target_prefix=[["__ru__"] for _ in source], beam_size=1, max_batch_size=64, batch_type="examples")
    out, errors = {}, {}
    for s, row in zip(sample, rows):
        hyp = list(row.hypotheses[0]) if row.hypotheses else []
        if hyp and hyp[0] == "__ru__":
            hyp = hyp[1:]
        text = tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).strip()
        if text:
            out[s.id] = text
        else:
            errors[s.id] = "empty translation"
    del tr
    return out, errors


def nllb(sample: list[Segment], mem: BookMemory):
    del mem
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_dir = snapshot_download(repo_id="osa911/nllb-200-distilled-600M-ct2-int8")
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="rus_Cyrl")
    tr = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8", inter_threads=1, intra_threads=max(1, os.cpu_count() or 2))
    source = [tok.convert_ids_to_tokens(tok.encode(s.text)) for s in sample]
    rows = tr.translate_batch(source, target_prefix=[["rus_Cyrl"] for _ in source], beam_size=1, max_batch_size=64, batch_type="examples")
    out, errors = {}, {}
    for s, row in zip(sample, rows):
        hyp = list(row.hypotheses[0]) if row.hypotheses else []
        if hyp and hyp[0] == "rus_Cyrl":
            hyp = hyp[1:]
        text = tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).strip()
        if text:
            out[s.id] = text
        else:
            errors[s.id] = "empty translation"
    del tr
    return out, errors


def wmt19(sample: list[Segment], mem: BookMemory):
    del mem
    import torch
    from transformers import FSMTForConditionalGeneration, FSMTTokenizer

    tok = FSMTTokenizer.from_pretrained("facebook/wmt19-en-ru")
    model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-ru")
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 2))
    out, errors = {}, {}
    with torch.inference_mode():
        for start in range(0, len(sample), 4):
            chunk = sample[start:start + 4]
            try:
                inputs = tok([s.text for s in chunk], return_tensors="pt", padding=True, truncation=True, max_length=768)
                generated = model.generate(**inputs, num_beams=1, max_new_tokens=900)
                for s, text in zip(chunk, tok.batch_decode(generated, skip_special_tokens=True)):
                    text = text.strip()
                    if text:
                        out[s.id] = text
                    else:
                        errors[s.id] = "empty translation"
            except Exception as exc:
                for s in chunk:
                    errors[s.id] = f"{type(exc).__name__}: {exc}"
    del model
    return out, errors


def main() -> None:
    doc = load_book(SOURCE)
    targets = [s for s in doc.segments if _should_translate(s.text)]
    sample = choose_sample(targets, max(16, int(os.getenv("BOOKAI_BAKEOFF_SEGMENTS") or "32")))
    total_chars = sum(len(s.text) for s in targets)
    mem = memory()
    runners = [
        ("opus-mt-en-ru-int8", opus),
        ("m2m100-418m-int8", m2m100),
        ("nllb-200-distilled-600m-int8-noncommercial", nllb),
        ("wmt19-en-ru-293m-pytorch", wmt19),
    ]
    results, translations = [], {"source": {s.id: s.text for s in sample}}
    for name, fn in runners:
        try:
            result, translated = run_model(name, fn, sample, mem, total_chars)
        except Exception as exc:
            result, translated = {"model": name, "fatal_error": f"{type(exc).__name__}: {exc}", "qa_pass_rate": 0.0, "source_chars_per_second": 0.0}, {}
            print("[mt-bakeoff] fatal " + json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
        translations[name] = translated
        gc.collect()

    def score(row: dict) -> float:
        quality = float(row.get("qa_pass_rate") or 0.0)
        cps = float(row.get("source_chars_per_second") or 0.0)
        return quality * 100 + min(10.0, math.log10(max(1.0, cps)) * 2.5)

    ranking = sorted(results, key=score, reverse=True)
    report = {
        "sample_segments": len(sample),
        "sample_chars": sum(len(s.text) for s in sample),
        "book_segments": len(targets),
        "book_source_chars": total_chars,
        "results": results,
        "ranking": [row.get("model") for row in ranking],
        "winner_by_qa_then_speed": ranking[0].get("model") if ranking else None,
        "deepseek_calls": 0,
        "beam_size": 1,
        "nllb_license_note": "CC-BY-NC-4.0; benchmark only, not production default",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    SAMPLES.write_text(json.dumps(translations, ensure_ascii=False, indent=2), "utf-8")
    print("[mt-bakeoff-result] " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
