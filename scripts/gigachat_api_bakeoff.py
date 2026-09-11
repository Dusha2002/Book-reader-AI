from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bookai.literary_context import locked_glossary_violations
from bookai.models import BookMemory, Segment
from bookai.parsers.base import load_book
from bookai.pipeline import _should_translate
from bookai.quality import batch_issues, hard_ids
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

SOURCE = Path(os.getenv("BOOKAI_SOURCE") or "Devices_and_Desires.fb2")
REPORT = Path(os.getenv("BOOKAI_GIGACHAT_REPORT") or "gigachat-api-bakeoff.json")
SAMPLES = Path(os.getenv("BOOKAI_GIGACHAT_SAMPLES") or "gigachat-api-bakeoff-samples.json")
WORKERS = max(1, int(os.getenv("BOOKAI_GIGACHAT_WORKERS") or "2"))
SCOPE = os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS"
BASE_URL = os.getenv("GIGACHAT_BASE_URL") or "https://api.giga.chat/v1"
MODEL_OVERRIDE = (os.getenv("BOOKAI_GIGACHAT_MODEL") or "").strip()
CREDENTIALS = (os.getenv("GIGACHAT_AUTH_KEY") or "").strip()

_thread_local = threading.local()
_access_token = ""


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
            continue
        step = (len(rows) - 1) / max(1, per - 1)
        picked.extend(rows[round(i * step)] for i in range(per))
    order = {s.id: i for i, s in enumerate(targets)}
    unique = {s.id: s for s in picked}
    return sorted(unique.values(), key=lambda s: order[s.id])[:count]


def auth_client():
    from gigachat import GigaChat

    return GigaChat(
        credentials=CREDENTIALS,
        scope=SCOPE,
        base_url=BASE_URL,
        verify_ssl_certs=False,
        timeout=180,
        max_retries=6,
        retry_backoff_factor=1.2,
    )


def worker_client():
    from gigachat import GigaChat

    value = getattr(_thread_local, "gigachat", None)
    if value is None:
        if not _access_token:
            raise RuntimeError("GigaChat access token was not initialized")
        value = GigaChat(
            access_token=_access_token,
            base_url=BASE_URL,
            verify_ssl_certs=False,
            timeout=180,
            max_retries=6,
            retry_backoff_factor=1.2,
        )
        _thread_local.gigachat = value
    return value


def model_names(client) -> list[str]:
    response = client.get_models()
    names: list[str] = []
    for row in getattr(response, "data", []) or []:
        name = getattr(row, "id_", None) or getattr(row, "id", None) or getattr(row, "name", None)
        if name:
            names.append(str(name))
    return names


def choose_model(names: list[str]) -> str:
    if MODEL_OVERRIDE:
        if MODEL_OVERRIDE not in names:
            raise RuntimeError(f"Requested model {MODEL_OVERRIDE!r} is not available; available={names}")
        return MODEL_OVERRIDE
    preferences = [
        "GigaChat-3-Ultra",
        "GigaChat-3-Pro",
        "GigaChat-3-Lightning",
        "GigaChat-2-Max",
        "GigaChat-Max",
        "GigaChat-2-Pro",
        "GigaChat-Pro",
        "GigaChat-2",
        "GigaChat",
    ]
    for candidate in preferences:
        if candidate in names:
            return candidate
    if not names:
        raise RuntimeError("GigaChat API returned no models")
    return names[0]


def prompt_for(text: str) -> str:
    return f"""Переведи художественный фрагмент с английского на русский.

Требования:
- сохрани весь смысл без пропусков, сокращений и добавлений;
- русский должен звучать как естественная опубликованная проза, а не машинный перевод;
- стиль сдержанный, точный, сухо-ироничный; не усиливай эмоции;
- сохраняй устройство длинных предложений, если оно естественно по-русски;
- диалоги делай живыми, без канцелярита;
- имена и термины: Valens=Валенс, Orsea=Орсеа, Melancton=Меланктон, Syracoelus=Сиракоэл, Eremia=Эремия, Eremians=эремийцы, Perpetual Republic=Вечная Республика;
- верни ТОЛЬКО перевод, без комментария, кавычек и предисловия.

SOURCE:
{text}"""


def strip_wrapper(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text|russian|ru)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    for prefix in ("Перевод:", "Перевод", "Russian translation:", "Translation:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(" \n:-")
            break
    return text.strip()


def translate_one(segment: Segment, model: str) -> tuple[str, str, dict]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Ты литературный переводчик с английского на русский. Приоритеты: точность смысла, полнота, естественная русская проза и сохранение авторского тона.",
            },
            {"role": "user", "content": prompt_for(segment.text)},
        ],
        "temperature": 0.15,
        "top_p": 0.9,
        "max_tokens": 2200,
    }
    response = worker_client().chat(payload)
    text = strip_wrapper(str(response.choices[0].message.content or ""))
    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    if not text:
        raise RuntimeError("empty GigaChat translation")
    return segment.id, text, usage


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
        if ratio < 0.42 or ratio > 1.8:
            abnormal.add(s.id)
        latin = sum(ch.isascii() and ch.isalpha() for ch in text)
        if latin / max(1, len(text)) > 0.16:
            english_tail.add(s.id)
        source_sentences = len(re.findall(r"[.!?](?:[\"'’”)]?)(?:\s|$)", s.text))
        target_sentences = len(re.findall(r"[.!?](?:[\"'’”)]?)(?:\s|$)", text))
        if source_sentences >= 4 and target_sentences <= max(1, source_sentences // 2 - 1):
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


def main() -> None:
    global _access_token

    if not CREDENTIALS:
        raise SystemExit("GIGACHAT_AUTH_KEY secret is missing")

    auth = auth_client()
    token = auth.get_token()
    _access_token = str(getattr(token, "access_token", "") or "")
    if not _access_token:
        raise RuntimeError("GigaChat OAuth succeeded but access_token is empty")

    models = model_names(auth)
    selected = choose_model(models)
    print("[gigachat-api] auth=ok scope=" + SCOPE + " token_reused=1", flush=True)
    print("[gigachat-api] available_models=" + json.dumps(models, ensure_ascii=False), flush=True)
    print("[gigachat-api] selected_model=" + selected, flush=True)

    doc = load_book(SOURCE)
    targets = [s for s in doc.segments if _should_translate(s.text)]
    count = max(8, int(os.getenv("BOOKAI_BAKEOFF_SEGMENTS") or "32"))
    sample = choose_sample(targets, count)
    total_chars = sum(len(s.text) for s in targets)
    sample_chars = sum(len(s.text) for s in sample)
    mem = memory()

    started = time.perf_counter()
    translated: dict[str, str] = {}
    errors: dict[str, str] = {}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(translate_one, segment, selected): segment for segment in sample}
        done = 0
        for future in as_completed(futures):
            segment = futures[future]
            try:
                sid, text, usage = future.result()
                translated[sid] = text
                for key in token_usage:
                    token_usage[key] += int(usage.get(key) or 0)
            except Exception as exc:
                errors[segment.id] = f"{type(exc).__name__}: {exc}"
            done += 1
            print(f"[gigachat-api] done={done}/{len(sample)} id={segment.id}", flush=True)

    elapsed = max(0.001, time.perf_counter() - started)
    cps = sample_chars / elapsed
    metrics = qa(sample, translated, mem)
    report = {
        "provider": "GigaChat API",
        "scope": SCOPE,
        "base_url": BASE_URL,
        "available_models": models,
        "model": selected,
        "segments": len(sample),
        "translated": len(translated),
        "errors": errors,
        "workers": WORKERS,
        "sample_chars": sample_chars,
        "elapsed_seconds": round(elapsed, 3),
        "source_chars_per_second": round(cps, 2),
        "estimated_full_minutes": None if cps <= 0 else round(total_chars / cps * 1.15 / 60.0, 2),
        "book_source_chars": total_chars,
        "token_usage": token_usage,
        "deepseek_calls": 0,
        **metrics,
    }
    samples = {
        "model": selected,
        "source": {s.id: s.text for s in sample},
        "translation": translated,
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    SAMPLES.write_text(json.dumps(samples, ensure_ascii=False, indent=2), "utf-8")
    print("[gigachat-api-result] " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
