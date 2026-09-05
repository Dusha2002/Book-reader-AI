from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bookai.llm import OpenAICompatibleProvider, extract_json
from bookai.parsers.base import load_book
from bookai.pipeline import _analysis_sample, _chapter_groups, _should_translate
from bookai.reference_harness import PRO_MODEL, build_reference_harness
from literary_benchmark import BENCHMARK, run_group


SOURCE = Path("Devices_and_Desires.fb2")

# Small benchmark only. The full user-provided reference PDF is deliberately not
# committed or injected into generation. These passages are used solely AFTER a
# blind candidate is produced, to judge whether the harness reaches the target
# literary register rather than merely passing self-QA.
REFERENCE_RU = {
    "s000008": "Кратчайший путь к сердцу мужчины, — сказал наставник, — как гласит пословица, лежит через желудок. Но если вам нужно добраться до его мозга, рекомендую глазницу.",
    "s000009": "Словно хлестнул кнут, наставник мгновенно распрямился из ленивой сутулости, вытянувшись в тугие, безупречно прямые линии выпада. Предплечье выбросилось от локтя, как стрела; передняя нога рванулась вперёд, и остриё длинного тонкого клинка, с точностью детали хорошо собранного механизма, проскочило точно через середину кольца, висевшего на шнурке под потолочной балкой.",
    "s000010": "Вполне в духе отца Валенса было заставить сына осваивать новомодное фехтование: стойки, туш, малую шпагу и рапиру. Всё это было изящно, утончённо, сложно, пожирало бездну времени и, разумеется, не годилось ни на что. Бригантина, да даже просто толстая зимняя куртка, остановила бы любое из этих изысканных остриёв; если хотелось причинить человеку хоть сколько-нибудь полезный вред, приходилось целиться в отверстия на лице — мишени размером не больше восьмимарочной монеты. А против батрака с садовым тесаком шансов не было вовсе. И всё же десять лет Валенс выгибался и тянулся туда-сюда вдоль меловой черты в продуваемом сарае, который не вычищали с тех пор, как он ещё был конюшней. Когда он научился попадать в яблоко, наставник подвесил сливу, потом тёрн. Теперь в тёрн он попадал девять раз из десяти, и его сменило кольцо. Освоит кольцо — интересно, что будет дальше? Ушко штопальной иглы, вероятно.",
    "s000011": "Лучше, — сказал наставник, когда клинок Валенса задел край кольца и оно звякнуло, точно коровий колокольчик. — Ещё раз.",
    "s000012": "И вполне в духе самого Валенса было каждую неделю терпеливо отбывать этот урок — с застывшим лицом и убийством в сердце, изо всех сил стараясь стать лучше, хотя прекрасно понимал, что всё занятие представляет собой упражнение в бессмыслице. По понедельникам фехтование было предпоследним уроком; зато вечером в среду, когда у него действительно находился свободный час, он платил одному из гвардейцев четыре марки за то, чтобы тот учил его нормальной работе мечом и щитом, и ещё две — за молчание перед отцом. В настоящем фехтовании он, по словам гвардейца, был довольно хорош. Но у туша не было режущей кромки, только остриё, и потому нельзя было одним красивым обратным ударом срезать с лица наставника ухмылку, как Валенсу давно хотелось. Вместо этого он оставался привязан к этой дурацкой меловой линии, будто коза на верёвке.",
    "s000343": "Как только герцог Орсеа понял, что проиграл битву, войну и единственную надежду своей страны на выживание, он приказал начать общее отступление. Это было единственное разумное решение, принятое им за весь день.",
    "s000344": "Один час изменил всё. Час назад, когда он повёл войска в атаку, мир был совсем другим. У него была армия в двадцать пять тысяч человек — десятая часть населения герцогства Эремия. У него была выгодная позиция, полностью снаряжённый обоз с припасами и оборудованием, тщательно подготовленный план сражения, внезапность, любовь и доверие народа и надежда. Теперь, пока ревели рога, а рваные линии строя ломались и растворялись в роях бегущих точек, ему досталась жалкая работа: вывести как можно больше из четырнадцати тысяч ошеломлённых, растерянных и озлобленных уцелевших из-под удара вражеской конницы и вернуть их в относительную безопасность гор. Один час, чтобы изменить мир; немногие сумели бы сделать это настолько основательно. Нужно обладать особым гением, чтобы за столь короткое время так исчерпывающе уничтожить собственную жизнь.",
    "s000345": "Мимо пробежал капитан лучников, совершенно неузнаваемый из-за раны на лице, выкрикивая что-то, чего Орсеа не расслышал. Ещё одна плохая новость, или всего лишь подтверждение уже известного; а может, просто ругань. Это почти не имело значения: приказ был отдан, и теперь он мало на что мог повлиять. Если солдаты доберутся до зарослей колючего кустарника на краю болот, если остановятся там и снова построятся, вместо того чтобы слепо вбежать в трясину, и если после всего, во что он их втянул, они окажутся достаточно доверчивыми, чтобы всё ещё выполнять его приказы, тогда, возможно, он ещё будет зачем-то нужен. Сейчас же он был не более чем мишенью — причём весьма заметной: на дурацком белом коне, в дурацких нарядных доспехах.",
    "s003324": "Меланктон был реалистом. Он понимал, что шансов победить Вечную Республику у него ничуть не больше, чем у эремийцев. И всё же, просто ради того, чтобы довести дело до конца, решил ещё немного поупрямиться с дальнобойными машинами. В конце концов, он приложил немало усилий, чтобы протащить сюда огромный запас снарядов именно для них; лучше уж использовать, чем дать добру пропасть, подумал он. Даже если стены не рухнут, жизнь внутри города на какое-то время станет весьма неприятной. На войне помогает каждая мелочь.",
    "s003325": "Поэтому обстрел возобновили, как только из долины втащили наверх остальные трёхцентнеровые шары. Меланктон не остался смотреть или слушать; оставил Сиракоэла командовать и удалился в штаб в главном лагере. Без него машины снова вошли в свой терпеливый, невыносимый ритм.",
    "s003326": "Сиракоэл был человеком прямым, не обременённым лишним воображением. Он велел расчётам сосредоточиться на четырёх участках главных надвратных башен — местах, которые, по его профессиональному мнению, хуже всего перенесут долгое жестокое молочение. Как минимум, считал он, удастся где-нибудь дать трещину или что-то ослабить. Иногда достаточно одной трещины.",
}


def _candidate_from_generated() -> tuple[dict[str, str], dict]:
    doc = load_book(SOURCE)
    source_segments = [s for s in doc.segments if _should_translate(s.text)]
    by_id = {s.id: s for s in source_segments}
    harness = build_reference_harness()
    chapters = _chapter_groups(source_segments)
    memory = harness.analyze(_analysis_sample(chapters, 90000))
    candidates: dict[str, str] = {}
    scene_reports = {}
    for name, ids in BENCHMARK.items():
        report = run_group(harness, source_segments, by_id, memory, name, ids)
        scene_reports[name] = report
        for row in report["segments"]:
            candidates[row["id"]] = row["final"]
    return candidates, {"generation": scene_reports, "generation_usage": harness.usage}


def _candidate_from_fb2(path: Path) -> tuple[dict[str, str], dict]:
    doc = load_book(path)
    by_id = {s.id: s.text for s in doc.segments}
    ids = [sid for group in BENCHMARK.values() for sid in group]
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise RuntimeError(f"Candidate FB2 misses benchmark ids: {missing}")
    return {sid: by_id[sid] for sid in ids}, {"candidate_file": str(path)}


def _judge(candidates: dict[str, str]) -> tuple[dict, dict]:
    source_doc = load_book(SOURCE)
    source_by_id = {s.id: s.text for s in source_doc.segments}
    ids = [sid for group in BENCHMARK.values() for sid in group]
    pairs = {
        sid: {
            "source_en": source_by_id[sid],
            "candidate_ru": candidates[sid],
            "reference_ru": REFERENCE_RU[sid],
        }
        for sid in ids
    }

    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("BOOKAI_API_KEY")
    base = os.getenv("BOOKAI_BASE_URL") or "https://openrouter.ai/api/v1"
    model = os.getenv("BOOKAI_REFERENCE_JUDGE_MODEL") or PRO_MODEL
    judge = OpenAICompatibleProvider(key, base, model, "none", role="reference_judge")
    system = """You are a strict bilingual literary translation evaluator.
The REFERENCE_RU is a user-approved style benchmark, not a string-matching gold answer. Judge whether CANDIDATE_RU
reaches comparable quality while remaining faithful to SOURCE_EN. Do not reward copying for its own sake and do not
penalize a different wording when it is equally natural and faithful.

For each id score 0..10 on:
- fidelity: all propositions, relations, negation, modality, technical/physical meaning;
- natural_russian: idiomatic literary Russian, no calques, awkward cognates, bureaucratese or modern register drift;
- voice_irony: preservation of dry irony, understatement, character/narrator register and punch-line timing;
- rhythm: sentence architecture and cadence comparable in effectiveness to the reference;
- terminology: names and technical terms are precise and consistent.

A blocker is only a serious defect that would make the passage unacceptable in a published literary translation.
Return ONLY JSON exactly as:
{"segments":{"id":{"fidelity":0,"natural_russian":0,"voice_irony":0,"rhythm":0,"terminology":0,"blockers":[],"note":"..."}},"overall_note":"..."}
Use every supplied id exactly once."""
    obj = extract_json(judge.complete(system, "PAIRS:" + json.dumps(pairs, ensure_ascii=False), temperature=0.0))
    if not isinstance(obj, dict) or not isinstance(obj.get("segments"), dict):
        raise ValueError("Reference judge returned invalid schema")
    rows = obj["segments"]
    if set(rows) != set(ids):
        raise ValueError(f"Reference judge id mismatch: expected={set(ids)} got={set(rows)}")
    return obj, judge.usage


def _summarize(judgement: dict) -> dict:
    metrics = ["fidelity", "natural_russian", "voice_irony", "rhythm", "terminology"]
    rows = judgement["segments"]
    averages = {
        metric: sum(float(row[metric]) for row in rows.values()) / len(rows)
        for metric in metrics
    }
    minimums = {
        metric: min(float(row[metric]) for row in rows.values())
        for metric in metrics
    }
    blockers = [
        {"id": sid, "blocker": str(blocker)}
        for sid, row in rows.items()
        for blocker in (row.get("blockers") or [])
        if str(blocker).strip()
    ]
    passed = (
        averages["fidelity"] >= 8.5
        and averages["natural_russian"] >= 8.0
        and averages["voice_irony"] >= 8.0
        and averages["rhythm"] >= 7.8
        and averages["terminology"] >= 8.0
        and minimums["fidelity"] >= 7.5
        and minimums["natural_russian"] >= 7.0
        and not blockers
    )
    return {
        "passed": passed,
        "averages": averages,
        "minimums": minimums,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference-quality-report.json"))
    args = parser.parse_args()

    if set(REFERENCE_RU) != {sid for group in BENCHMARK.values() for sid in group}:
        raise RuntimeError("Reference benchmark ids do not match literary benchmark ids")

    if args.candidate:
        candidates, metadata = _candidate_from_fb2(args.candidate)
        mode = "postflight"
    else:
        candidates, metadata = _candidate_from_generated()
        mode = "preflight"

    judgement, judge_usage = _judge(candidates)
    summary = _summarize(judgement)
    report = {
        "mode": mode,
        "reference_used_in_generation": False,
        "summary": summary,
        "judgement": judgement,
        "candidates": candidates,
        "judge_usage": judge_usage,
        **metadata,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("[reference-gate] " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"[reference-gate] report={args.output}", flush=True)
    if not summary["passed"]:
        raise RuntimeError("Reference literary benchmark gate failed")


if __name__ == "__main__":
    main()
