from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from bookai.gigachat_mt import GigaChatLightningBackend
from bookai.models import BookMemory, Segment
from bookai.parsers.base import load_book
from bookai.reference_harness import build_reference_harness
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

SOURCE = Path("Devices_and_Desires.fb2")
OUT_JSON = Path("literary-sample-eval.json")
OUT_MD = Path("literary-sample-eval.md")

# Short excerpts from the user's PDF benchmark Devices_and_Desires_RU_FULL_FINAL(3).pdf.
CASES = [
    {
        "case_id": "ch01-valens-irony",
        "chapter": "Chapter One",
        "keywords": ["valens", "father", "typical"],
        "reference": "Вполне в духе отца Валенса было заставить сына осваивать новомодное фехтование: стойки, туш, малую шпагу и рапиру. Всё это было изящно, утончённо, сложно, пожирало бездну времени и, разумеется, не годилось ни на что. Бригантина, да даже просто толстая зимняя куртка, остановила бы любое из этих изысканных остриёв; если хотелось причинить человеку хоть сколько-нибудь полезный вред, приходилось целиться в отверстия на лице — мишени размером не больше восьмимарочной монеты.",
    },
    {
        "case_id": "ch02-trial-portrait",
        "chapter": "Chapter Two",
        "keywords": ["ordinary", "man", "bald"],
        "reference": "По любым меркам это был совершенно заурядный человек: чуть ниже среднего роста, лысый, с пучками белых волос над ушами; круглолицый, малоподвижный, с яркими карими глазами. Зиани знал его много лет — по комитетам, приёмам и посещениям фабрик; дважды встречался с его женой и однажды — с дочерью.",
    },
    {
        "case_id": "ch05-hunting-routine",
        "chapter": "Chapter Five",
        "keywords": ["hunted", "three", "week"],
        "reference": "В отличие от отца, молодой герцог охотился три дня в неделю, всегда по одному и тому же распорядку. По вторникам — парфорсная охота с полной сворой: прочёсывали горные перелески в поисках косули, если был сезон, кабана, медведя и волка. По четвергам охотились «лук и стойка»: охотники пешие и неподвижные, пока свора выгоняла зверя из долинных посадок и вересковых пустошей у границы леса.",
    },
    {
        "case_id": "ch09-psellus-orthodoxy",
        "chapter": "Chapter Nine",
        "keywords": ["commissioner", "psell"],
        "reference": "Комиссар Лукайо Пселл за свою жизнь повидал немало странного, и то, что он пережил каждое из этих тревожащих испытаний, не позволив ни одному из них хоть сколько-нибудь повредить себе, служило лучшим доказательством его безупречной ортодоксальности. Ему доводилось, против воли, читать ересь и слушать мерзость — и вырванные пыткой признания человека, которого уже сломали, и гордые тирады нераскаявшегося мученика.",
    },
    {
        "case_id": "ch11-head-dark-irony",
        "chapter": "Chapter Eleven",
        "keywords": ["disadvantage", "head"],
        "reference": "Один из недостатков обычая посылать врагу голову в качестве жеста состоит в том, что всё остальное тело остаётся у тебя. Орсеа настоял, что хочет его увидеть. Миэль не был уверен зачем; по его мнению, потому что Орсеа всегда отличался некоторой брезгливостью. Раз он приказал казнить несчастную женщину, значит, обязан наказать себя видом её обезглавленного тела.",
    },
]


def _select(segments: list[Segment], case: dict) -> Segment:
    chapter = case["chapter"].casefold()
    in_chapter = [s for s in segments if chapter in (s.chapter or "").casefold()]
    pool = in_chapter or segments
    keys = [k.casefold() for k in case["keywords"]]
    ranked: list[tuple[int, int, Segment]] = []
    for s in pool:
        text = s.text.casefold()
        hits = sum(k in text for k in keys)
        length_score = -abs(min(len(s.text), 1800) - 700)
        ranked.append((hits, length_score, s))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if ranked and ranked[0][0] > 0:
        return ranked[0][2]
    candidates = [s for s in pool if 220 <= len(s.text) <= 1800]
    if candidates:
        return candidates[0]
    return pool[0]


def _style_memory(harness, selected: list[Segment]) -> BookMemory:
    # Compact DeepSeek analysis: only the five selected source passages, no full-book digests.
    sample = "\n\n".join(f"[{s.chapter}]\n{s.text}" for s in selected)
    memory = harness.analyze(sample[:16000])
    memory.glossary.update(REFERENCE_GLOSSARY_SEED)
    return apply_reference_profile(memory)


def _judge(harness, source: str, candidate: str, reference: str) -> dict:
    system = (
        "You are a strict bilingual literary translation evaluator. Return only valid JSON. "
        "Do not reward verbatim similarity by itself; judge meaning and literary effect."
    )
    user = f"""Evaluate the Russian candidate against the English source and an independent reference translation.
Return numeric 0-10 fields: meaning, natural_russian, author_voice, dry_irony, rhythm, terminology, overall; plus comment in Russian, max 35 words.

ENGLISH SOURCE:\n{source}\n\nCANDIDATE:\n{candidate}\n\nREFERENCE:\n{reference}"""
    raw = harness.analyzer.complete(system, user, temperature=0.0)
    text = str(raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except Exception:
        return {"raw": text[:1000]}


def main() -> None:
    document = load_book(SOURCE)
    selected = [_select(document.segments, case) for case in CASES]
    harness = build_reference_harness()

    t0 = time.perf_counter()
    memory = _style_memory(harness, selected)
    analysis_seconds = time.perf_counter() - t0

    bulk = GigaChatLightningBackend()
    if not bulk.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY missing")

    rows: list[dict] = []
    for case, segment in zip(CASES, selected):
        t1 = time.perf_counter()
        lightning_map, errors = bulk.translate_many([segment], memory, source_segments=document.segments)
        lightning = lightning_map.get(segment.id, "")
        lightning_seconds = time.perf_counter() - t1
        if not lightning:
            rows.append({**case, "segment_id": segment.id, "source": segment.text, "error": errors.get(segment.id, "GigaChat failed")})
            continue

        t2 = time.perf_counter()
        polished_map = harness.polish([segment], {segment.id: lightning}, memory, context=[])
        polished = polished_map.get(segment.id, lightning)
        polish_seconds = time.perf_counter() - t2

        row = {
            "case_id": case["case_id"],
            "chapter": case["chapter"],
            "segment_id": segment.id,
            "source": segment.text,
            "lightning": lightning,
            "deepseek_polished": polished,
            "reference": case["reference"],
            "lightning_score": _judge(harness, segment.text, lightning, case["reference"]),
            "polished_score": _judge(harness, segment.text, polished, case["reference"]),
            "lightning_seconds": round(lightning_seconds, 2),
            "polish_seconds": round(polish_seconds, 2),
        }
        rows.append(row)
        print(json.dumps({"case": case["case_id"], "segment": segment.id, "lightning_s": row["lightning_seconds"], "polish_s": row["polish_seconds"]}, ensure_ascii=False), flush=True)

    report = {
        "architecture": "GigaChat-3-Lightning draft -> DeepSeek V4.1 Flash polish/judge",
        "cases": len(rows),
        "style_analysis_seconds": round(analysis_seconds, 2),
        "gigachat_usage": bulk.usage.as_dict(),
        "deepseek_usage": harness.usage,
        "style_memory": asdict(memory.style),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    md = ["# Literary sample evaluation", "", f"Style analysis: {analysis_seconds:.1f}s", ""]
    for row in rows:
        md += [
            f"## {row.get('case_id')} — {row.get('segment_id','?')}", "",
            "### English source", row.get("source", ""), "",
            "### GigaChat-3-Lightning", row.get("lightning", row.get("error", "")), "",
            "### DeepSeek V4.1 Flash polish", row.get("deepseek_polished", ""), "",
            "### PDF reference", row.get("reference", ""), "",
            "### Scores", "```json", json.dumps({"lightning": row.get("lightning_score"), "polished": row.get("polished_score")}, ensure_ascii=False, indent=2), "```", "",
        ]
    OUT_MD.write_text("\n".join(md), "utf-8")
    print(json.dumps({"phase": "done", "cases": len(rows), "gigachat_usage": bulk.usage.as_dict()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
