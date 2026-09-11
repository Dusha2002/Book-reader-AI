from __future__ import annotations

import concurrent.futures
import json
import os
import re
import resource
import time
from pathlib import Path

import httpx

from bookai.literary_context import locked_glossary_violations
from bookai.models import BookMemory, Segment
from bookai.parsers.base import load_book
from bookai.pipeline import _should_translate
from bookai.quality import batch_issues, hard_ids
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

SOURCE = Path(os.getenv("BOOKAI_SOURCE") or "Devices_and_Desires.fb2")
REPORT = Path(os.getenv("BOOKAI_RU_LLM_REPORT") or "russian-llm-bakeoff.json")
SAMPLES = Path(os.getenv("BOOKAI_RU_LLM_SAMPLES") or "russian-llm-bakeoff-samples.json")
BASE_URL = os.getenv("BOOKAI_LOCAL_LLM_URL") or "http://127.0.0.1:8080/v1"
MODEL_LABEL = os.getenv("BOOKAI_LOCAL_LLM_NAME") or "local-russian-llm"
WORKERS = max(1, int(os.getenv("BOOKAI_LOCAL_LLM_WORKERS") or "2"))
TIMEOUT = float(os.getenv("BOOKAI_LOCAL_LLM_TIMEOUT") or "180")


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


def model_id() -> str:
    with httpx.Client(timeout=30) as client:
        data = client.get(f"{BASE_URL}/models").json()
    rows = data.get("data") or []
    if rows and rows[0].get("id"):
        return str(rows[0]["id"])
    return "local-model"


def glossary_instruction(text: str) -> str:
    pairs = []
    low = text.casefold()
    for src, dst in REFERENCE_GLOSSARY_SEED.items():
        if src.casefold() in low:
            pairs.append(f"{src} → {dst}")
    return "; ".join(pairs)


def translate_one(seg: Segment, mid: str) -> tuple[str, str | None]:
    glossary = glossary_instruction(seg.text)
    system = (
        "Ты профессиональный литературный переводчик с английского на русский. "
        "Переводи точно и полно: не пропускай ни одного предложения, причинно-следственной связи, "
        "детали действия или технического описания. Русский должен звучать как естественная художественная проза, "
        "без кальки и канцелярита. Сохраняй сухую иронию, сдержанный тон, ритм длинных фраз, диалоги и намеренные повторы. "
        "Не добавляй пояснений, комментариев, кавычек вокруг всего перевода и не пересказывай текст. Верни только перевод."
    )
    user = "Переведи следующий фрагмент полностью на русский."
    if glossary:
        user += f" Обязательная терминология: {glossary}."
    user += "\n\n" + seg.text
    payload = {
        "model": mid,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": min(2200, max(256, int(len(seg.text) * 1.8))),
        "stream": False,
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BASE_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = str(data["choices"][0]["message"]["content"] or "").strip()
        text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        if not text:
            return seg.id, "ERROR: empty translation"
        return seg.id, text
    except Exception as exc:
        return seg.id, f"ERROR: {type(exc).__name__}: {exc}"


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?](?:[\"'”’)]*)\s|[.!?]$", text.strip())))


def qa(sample: list[Segment], translated: dict[str, str], mem: BookMemory) -> dict:
    hard = set(hard_ids(batch_issues(sample, translated, mem)))
    glossary = locked_glossary_violations(sample, translated, REFERENCE_GLOSSARY_SEED)
    missing = {s.id for s in sample if not str(translated.get(s.id) or "").strip()}
    abnormal: set[str] = set()
    english_tail: set[str] = set()
    sentence_loss: set[str] = set()
    for s in sample:
        text = str(translated.get(s.id) or "")
        if not text:
            continue
        ratio = len(text) / max(1, len(s.text))
        if ratio < 0.45 or ratio > 1.9:
            abnormal.add(s.id)
        latin = sum(ch.isascii() and ch.isalpha() for ch in text)
        if latin / max(1, len(text)) > 0.16:
            english_tail.add(s.id)
        src_sent = sentence_count(s.text)
        dst_sent = sentence_count(text)
        if src_sent >= 3 and dst_sent < max(1, int(src_sent * 0.55)):
            sentence_loss.add(s.id)
    bad = hard | set(glossary) | missing | abnormal | english_tail | sentence_loss
    return {
        "hard_qa": sorted(hard),
        "glossary_violations": glossary,
        "missing": sorted(missing),
        "abnormal_ratio": sorted(abnormal),
        "english_tail": sorted(english_tail),
        "sentence_loss": sorted(sentence_loss),
        "bad_ids": sorted(bad),
        "qa_pass": len(sample) - len(bad),
        "qa_pass_rate": round((len(sample) - len(bad)) / max(1, len(sample)), 4),
    }


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main() -> None:
    doc = load_book(SOURCE)
    targets = [s for s in doc.segments if _should_translate(s.text)]
    count = max(16, int(os.getenv("BOOKAI_BAKEOFF_SEGMENTS") or "32"))
    sample = choose_sample(targets, count)
    total_chars = sum(len(s.text) for s in targets)
    mem = memory()
    mid = model_id()

    started = time.perf_counter()
    translated: dict[str, str] = {}
    errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(translate_one, seg, mid): seg for seg in sample}
        for fut in concurrent.futures.as_completed(futures):
            seg = futures[fut]
            sid, text = fut.result()
            if text.startswith("ERROR:"):
                errors[sid] = text
            else:
                translated[sid] = text
            print(f"[ru-llm-bakeoff] done={len(translated)+len(errors)}/{len(sample)} id={seg.id}", flush=True)

    elapsed = max(0.001, time.perf_counter() - started)
    chars = sum(len(s.text) for s in sample if s.id in translated)
    cps = chars / elapsed
    quality = qa(sample, translated, mem)
    report = {
        "model": MODEL_LABEL,
        "api_model_id": mid,
        "runtime": "llama.cpp CPU OpenAI-compatible server",
        "segments": len(sample),
        "translated": len(translated),
        "errors": errors,
        "sample_chars": chars,
        "elapsed_seconds": round(elapsed, 3),
        "source_chars_per_second": round(cps, 2),
        "estimated_full_minutes": None if cps <= 0 else round(total_chars / cps * 1.15 / 60.0, 2),
        "process_peak_rss_mb": rss_mb(),
        "deepseek_calls": 0,
        "workers": WORKERS,
        **quality,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    SAMPLES.write_text(json.dumps({
        "model": MODEL_LABEL,
        "source": {s.id: s.text for s in sample},
        "translation": translated,
        "errors": errors,
    }, ensure_ascii=False, indent=2), "utf-8")
    print("[ru-llm-bakeoff-result] " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
