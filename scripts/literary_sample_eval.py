from __future__ import annotations

import json
import time
from pathlib import Path

from bookai.gigachat_mt import GigaChatLightningBackend
from bookai.models import BookMemory, Segment, StyleGuide
from bookai.parsers.base import load_book
from bookai.reference_harness import build_reference_harness
from bookai.reference_profile import REFERENCE_GLOSSARY_SEED, apply_reference_profile

SOURCE = Path("Devices_and_Desires.fb2")
OUT_JSON = Path("literary-sample-eval.json")
OUT_MD = Path("literary-sample-eval.md")

CASES = [
    {
        "case_id": "ch01-valens-fencing",
        "must": ["brigandine", "fencing", "valens"],
        "reference": "Вполне в духе отца Валенса было заставить сына осваивать новомодное фехтование: стойки, туш, малую шпагу и рапиру. Всё это было изящно, утончённо, сложно, пожирало бездну времени и, разумеется, не годилось ни на что. Бригантина, да даже просто толстая зимняя куртка, остановила бы любое из этих изысканных остриёв; если хотелось причинить человеку хоть сколько-нибудь полезный вред, приходилось целиться в отверстия на лице — мишени размером не больше восьмимарочной монеты. А против батрака с садовым тесаком шансов не было вовсе. И всё же десять лет Валенс выгибался и тянулся туда-сюда вдоль меловой черты в продуваемом сарае, который не вычищали с тех пор, как он ещё был конюшней. Когда он научился попадать в яблоко, наставник подвесил сливу, потом тёрн. Теперь в тёрн он попадал девять раз из десяти, и его сменило кольцо. Освоит кольцо — интересно, что будет дальше? Ушко штопальной иглы, вероятно.",
    },
    {
        "case_id": "ch02-trial-portrait",
        "must": ["nondescript", "bald", "sphrantzes"],
        "reference": "По любым меркам это был совершенно заурядный человек: чуть ниже среднего роста, лысый, с пучками белых волос над ушами; круглолицый, малоподвижный, с яркими карими глазами. Зиани знал его много лет — по комитетам, приёмам и посещениям фабрик; дважды встречался с его женой и однажды — с дочерью. После этих встреч в памяти у него остался образ человека с громким, высоким голосом: деятельного и занятого, но достаточно вежливого, чтобы не забывать о субординации. Зиани знал, что тот занимает какой-то высокий пост в Гильдии, но лишь сегодня впервые узнал, чем именно занимается Лодоико Сфрантзес.",
    },
    {
        "case_id": "ch05-hunting-routine",
        "must": ["hunted three days a week", "parforce"],
        "reference": "В отличие от отца, молодой герцог охотился три дня в неделю, всегда по одному и тому же распорядку. По вторникам — парфорсная охота с полной сворой: прочёсывали горные перелески в поисках косули, если был сезон, кабана, медведя и волка. По четвергам охотились «лук и стойка»: охотники пешие и неподвижные, пока свора выгоняла зверя из долинных посадок и вересковых пустошей у границы леса. Суббота предназначалась для соколиной охоты, если только погода не становилась слишком сырой и холодной; тогда брали терьеров и шли по кроличьим норам либо пытали счастья, поднимая кроликов вокруг садов. Большие облавы теперь остались в прошлом; молодой герцог не любил беспокойство, которое они причиняли, и то, как после них дичь разбредалась с привычных участков.",
    },
    {
        "case_id": "ch09-psellus-orthodoxy",
        "must": ["commissioner", "psellus", "orthodoxy"],
        "reference": "Комиссар Лукайо Пселл за свою жизнь повидал немало странного, и то, что он пережил каждое из этих тревожащих испытаний, не позволив ни одному из них хоть сколько-нибудь повредить себе, служило лучшим доказательством его безупречной ортодоксальности. Ему доводилось, против воли, читать ересь и слушать мерзость — и вырванные пыткой признания человека, которого уже сломали, и гордые тирады нераскаявшегося мученика. Он видел то, чего видеть не должен никто: все мыслимые разновидности уродливого и ложного. И выдержал.",
    },
    {
        "case_id": "ch11-head-dark-irony",
        "chapter": "Chapter Eleven",
        "first_long_after_opening": True,
        "reference": "Один из недостатков обычая посылать врагу голову в качестве жеста состоит в том, что всё остальное тело остаётся у тебя. Орсеа настоял, что хочет его увидеть. Миэль не был уверен зачем; по его мнению, потому что Орсеа всегда отличался некоторой брезгливостью. Раз он приказал казнить несчастную женщину, значит, обязан наказать себя видом её обезглавленного тела. Если причина действительно была в этом, она была путаной, нелогичной, трудной для понимания всеми, кроме самого Орсеа, — то есть совершенно в его характере.",
    },
]

STYLE = StyleGuide(
    narrative_voice=(
        "Third-person limited, close to character perspective; restrained, precise, wry and dryly ironic. "
        "Let irony emerge through juxtaposition and understatement rather than explanation."
    ),
    rhythm=(
        "Preserve Parker's long accumulative sentences, enumerations, semicolons and delayed punch lines when functional; "
        "rebuild them as idiomatic Russian rather than chopping or mirroring English syntax."
    ),
    dialogue="Restrained natural Russian dialogue, controlled and period-neutral; preserve subtext and speaker register.",
    humor="Dry situational irony and understatement; never make the joke louder, slangier or more emotional than the source.",
    taboos=[
        "No English-syntax calques or bureaucratic Russian unless deliberately present in the source.",
        "Do not explain irony or character psychology.",
        "Do not intensify neutral narration into melodrama or slang.",
        "Preserve physical, technical and causal precision.",
        "Keep names and terminology consistent.",
    ],
)


def _select(segments: list[Segment], case: dict) -> Segment:
    if case.get("first_long_after_opening"):
        chapter_name = str(case["chapter"]).casefold()
        chapter = [s for s in segments if chapter_name in (s.chapter or "").casefold()]
        substantial = [s for s in chapter if len(s.text) >= 250]
        if not substantial:
            raise RuntimeError(f"No substantial source paragraph found in {case['chapter']}")
        chosen = substantial[0]
    else:
        keys = [str(k).casefold() for k in case["must"]]
        matches = [s for s in segments if all(k in s.text.casefold() for k in keys)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one source match for {case['case_id']}, found {len(matches)}")
        chosen = matches[0]
    print(json.dumps({"alignment": case["case_id"], "segment": chosen.id, "chapter": chosen.chapter, "source_start": chosen.text[:180]}, ensure_ascii=False), flush=True)
    return chosen


def _memory() -> BookMemory:
    memory = BookMemory(style=STYLE, glossary=dict(REFERENCE_GLOSSARY_SEED))
    return apply_reference_profile(memory)


def _judge_pair(harness, source: str, lightning: str, polished: str, reference: str) -> dict:
    system = (
        "You are a strict bilingual literary translation evaluator. Return only valid JSON. "
        "Do not reward verbatim similarity by itself; judge fidelity to English and achievement of the same literary effects as the independent reference."
    )
    user = f"""Score BOTH candidates independently. Return JSON with keys lightning and polished. Each object must contain numeric 0-10 fields meaning, natural_russian, author_voice, dry_irony, rhythm, terminology, overall, plus comment in Russian (max 35 words).

ENGLISH SOURCE:\n{source}\n\nLIGHTNING:\n{lightning}\n\nDEEPSEEK-POLISHED:\n{polished}\n\nREFERENCE:\n{reference}"""
    raw = harness.analyzer.complete(system, user, temperature=0.0)
    text = str(raw or "").strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text[:1500]}


def _write_report(rows: list[dict], bulk, harness, started: float) -> None:
    report = {
        "architecture": "GigaChat-3-Lightning draft -> DeepSeek V4.1 Flash polish/judge",
        "benchmark_alignment": "exact paragraph matching",
        "requested_cases": len(CASES),
        "completed_cases": sum(1 for r in rows if r.get("lightning")),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "gigachat_usage": bulk.usage.as_dict(),
        "deepseek_usage": harness.usage,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    md = ["# Literary sample evaluation — exact PDF alignment", ""]
    for row in rows:
        md += [f"## {row['case_id']} — {row['segment_id']}", "", "### English source", row["source"], ""]
        if row.get("error"):
            md += ["### Error", row["error"], ""]
        else:
            md += [
                "### GigaChat-3-Lightning", row.get("lightning", ""), "",
                "### DeepSeek V4.1 Flash polish", row.get("deepseek_polished", ""), "",
                "### PDF reference", row["reference"], "",
                "### Scores", "```json", json.dumps({"lightning": row.get("lightning_score"), "polished": row.get("polished_score")}, ensure_ascii=False, indent=2), "```", "",
            ]
    OUT_MD.write_text("\n".join(md), "utf-8")


def main() -> None:
    document = load_book(SOURCE)
    selected = [_select(document.segments, case) for case in CASES]
    memory = _memory()
    harness = build_reference_harness()
    bulk = GigaChatLightningBackend()
    if not bulk.available():
        raise RuntimeError("GIGACHAT_AUTH_KEY missing")

    rows: list[dict] = []
    started = time.perf_counter()
    for case, segment in zip(CASES, selected):
        base = {
            "case_id": case["case_id"],
            "chapter": segment.chapter,
            "segment_id": segment.id,
            "source": segment.text,
            "reference": case["reference"],
        }
        try:
            t1 = time.perf_counter()
            lightning_map, errors = bulk.translate_many([segment], memory, source_segments=document.segments)
            lightning = lightning_map.get(segment.id, "")
            lightning_seconds = time.perf_counter() - t1
            if not lightning:
                raise RuntimeError(errors.get(segment.id) or "GigaChat returned no translation")

            t2 = time.perf_counter()
            polished_map = harness.polish([segment], {segment.id: lightning}, memory, context=[])
            polished = polished_map.get(segment.id, lightning)
            polish_seconds = time.perf_counter() - t2
            try:
                scores = _judge_pair(harness, segment.text, lightning, polished, case["reference"])
            except Exception as exc:
                scores = {"judge_error": f"{type(exc).__name__}: {exc}"}

            row = {
                **base,
                "lightning": lightning,
                "deepseek_polished": polished,
                "lightning_score": scores.get("lightning", scores),
                "polished_score": scores.get("polished", scores),
                "lightning_seconds": round(lightning_seconds, 2),
                "polish_seconds": round(polish_seconds, 2),
            }
            print(json.dumps({"case": case["case_id"], "segment": segment.id, "lightning_s": row["lightning_seconds"], "polish_s": row["polish_seconds"]}, ensure_ascii=False), flush=True)
        except Exception as exc:
            row = {**base, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps({"case": case["case_id"], "segment": segment.id, "error": row["error"]}, ensure_ascii=False), flush=True)
        rows.append(row)
        _write_report(rows, bulk, harness, started)

    _write_report(rows, bulk, harness, started)
    print(json.dumps({"phase": "done", "requested_cases": len(CASES), "completed_cases": sum(1 for r in rows if r.get("lightning")), "gigachat_usage": bulk.usage.as_dict()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
