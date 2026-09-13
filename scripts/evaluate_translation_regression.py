from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_LATIN = re.compile(r"\b[A-Za-z]{2,}\b")
_DIGIT = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?")
_NEG_SRC = re.compile(
    r"\b(?:not|never|none|neither|nor|without|cannot|can't|won't|wouldn't|didn't|"
    r"doesn't|isn't|aren't|wasn't|weren't|shouldn't|couldn't|mustn't)\b",
    re.I,
)
_NEG_RU = re.compile(r"\b(?:не|нет|ни|никогд|без|нельзя|невозмож)\w*", re.I)
_TIME_FACTS = (
    (
        re.compile(r"\bthis afternoon\b", re.I),
        re.compile(
            r"(?:сегодня[^.!?]{0,35}(?:дн[её]м|дн[яе]|после полудня|во второй половине дня)|"
            r"(?:дн[её]м|после полудня)[^.!?]{0,35}сегодня)",
            re.I,
        ),
        "this_afternoon",
    ),
    (
        re.compile(r"\btomorrow morning\b", re.I),
        re.compile(r"завтра[^.!?]{0,30}утр\w*", re.I),
        "tomorrow_morning",
    ),
    (
        re.compile(r"\btomorrow (?:night|evening)\b", re.I),
        re.compile(r"завтра[^.!?]{0,30}(?:вечер\w*|ноч\w*)", re.I),
        "tomorrow_night",
    ),
    (
        re.compile(r"\btonight\b", re.I),
        re.compile(
            r"(?:сегодня[^.!?]{0,30}(?:вечер\w*|ноч\w*)|"
            r"(?:вечер\w*|ноч\w*)[^.!?]{0,30}сегодня)",
            re.I,
        ),
        "tonight",
    ),
)

_NUMBER_REL_RE = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?%?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|dozen|hundred|thousand|million|half|quarter|third|twice|double|"
    r"triple|percent|per cent|times|more than|less than|fewer than|as many as|"
    r"outnumber(?:ed|ing)?)\b",
    re.I,
)
_TIME_RE = re.compile(
    r"\b(?:today|tomorrow|yesterday|tonight|morning|afternoon|evening|night|noon|"
    r"midnight|earlier|later|ago|before|after|until|during|while|meanwhile|then)\b",
    re.I,
)
_CAUSAL_RE = re.compile(
    r"\b(?:because|therefore|thus|hence|so that|as a result|owing to|due to|since|"
    r"unless|if|although|though|whereas|consequently)\b",
    re.I,
)
_REF_RE = re.compile(
    r"\b(?:either|neither|both|former|latter|the other|one of them|each of them|"
    r"all of them|one another|each other|who|whom|whose|which|they|them|their|he|"
    r"him|his|she|her|it|its|this|that|these|those)\b",
    re.I,
)
_PASSIVE_RE = re.compile(
    r"\b(?:was|were|is|are|been|being)\s+(?:\w+\s+){0,3}\w+(?:ed|en)\b[^.!?]{0,100}\bby\b",
    re.I,
)
_TERM_RE = re.compile(
    r"\b(?:brigandine|damson|darning[- ]needle|bushing|gear|spring|blade|alloy|"
    r"mechanism|advocate|counsel|prosecutor|attorney|court|prisoner|defen[cs]e|"
    r"offen[cs]e|engineer)\b",
    re.I,
)
_DIALOGUE_RE = re.compile(r"^\s*['\"“‘]")


def _categories(source: str) -> list[str]:
    out = []
    if _NUMBER_REL_RE.search(source):
        out.append("numbers")
    if _TIME_RE.search(source):
        out.append("time")
    if _NEG_SRC.search(source):
        out.append("negation")
    if _CAUSAL_RE.search(source):
        out.append("causality")
    if _REF_RE.search(source):
        out.append("reference_resolution")
    if _PASSIVE_RE.search(source):
        out.append("participant_roles")
    if _TERM_RE.search(source):
        out.append("terminology")
    if _DIALOGUE_RE.search(source):
        out.append("dialogue")
    return out


def _risk_score(source: str, cats: list[str]) -> float:
    score = 2.0 * len(cats) + min(3.0, len(source) / 250.0)
    if "terminology" in cats:
        score += 2.0
    if "participant_roles" in cats:
        score += 2.0
    if "reference_resolution" in cats:
        score += 1.0
    if "numbers" in cats or "time" in cats:
        score += 0.5
    return score


def _load_maps(paths: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pattern in paths:
        expanded = glob.glob(pattern)
        if not expanded and Path(pattern).exists():
            expanded = [pattern]
        for file_name in expanded:
            obj = json.loads(Path(file_name).read_text("utf-8"))
            if not isinstance(obj, list):
                continue
            for row in obj:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("id") or "")
                if sid:
                    rows[sid] = row
    return rows


def _auto_sample(
    mapping: dict[str, dict[str, Any]],
    curated_ids: set[str],
    target: int,
) -> list[dict[str, Any]]:
    candidates = []
    for sid, row in mapping.items():
        if sid in curated_ids:
            continue
        source = str(row.get("source") or "")
        if not source or source.casefold().startswith("chapter "):
            continue
        cats = _categories(source)
        if not cats:
            continue
        candidates.append(
            {
                "id": sid,
                "book": "runtime-map",
                "chapter": str(row.get("chapter") or ""),
                "categories": cats,
                "source": source,
                "curated": False,
                "_score": _risk_score(source, cats),
            }
        )
    candidates.sort(key=lambda x: (-float(x["_score"]), x["id"]))
    need = max(0, target - len(curated_ids))
    for row in candidates[:need]:
        row.pop("_score", None)
    return candidates[:need]


def _automatic_checks(source: str, target: str) -> list[str]:
    failures: list[str] = []
    if _LATIN.search(target):
        failures.append("latin_leak")
    digits = [x.replace(",", ".").rstrip("%") for x in _DIGIT.findall(source)]
    normalized_target = target.replace(",", ".")
    for token in digits:
        if token and token not in normalized_target:
            failures.append("explicit_number_missing")
            break
    if _NEG_SRC.search(source) and not _NEG_RU.search(target):
        failures.append("negation_marker_missing")
    for source_re, target_re, code in _TIME_FACTS:
        if source_re.search(source) and not target_re.search(target):
            failures.append("time_marker:" + code)
    return failures


def _curated_checks(case: dict[str, Any], target: str) -> list[str]:
    exp = dict(case.get("expectations") or {})
    failures = []
    for pattern in exp.get("required_regex") or []:
        if not re.search(pattern, target, re.I | re.S):
            failures.append("required:" + pattern)
    for pattern in exp.get("forbidden_regex") or []:
        if re.search(pattern, target, re.I | re.S):
            failures.append("forbidden:" + pattern)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="+")
    parser.add_argument("--dataset", default="eval/literary_regression_v1.json")
    parser.add_argument("--output", default="chapter-v9-regression.json")
    parser.add_argument("--fail-on-hard", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text("utf-8"))
    mapping = _load_maps(args.maps)
    curated = list(dataset.get("cases") or [])
    curated_ids = {str(x.get("id") or "") for x in curated}
    target_size = int(dataset.get("auto_candidate_target") or len(curated))
    cases = curated + _auto_sample(mapping, curated_ids, target_size)

    results = []
    hard_failures = []
    auto_counts = Counter()
    category_total = Counter()
    category_auto_fail = Counter()
    curated_pass = 0
    curated_total = 0
    missing = 0

    for case in cases:
        sid = str(case.get("id") or "")
        row = mapping.get(sid)
        source = str((row or {}).get("source") or case.get("source") or "")
        target = str((row or {}).get("translation") or "")
        cats = list(case.get("categories") or _categories(source))
        for cat in cats:
            category_total[cat] += 1

        if not row:
            missing += 1
            result = {
                "id": sid,
                "curated": bool(case.get("expectations")),
                "categories": cats,
                "status": "missing",
                "automatic_failures": ["missing_translation_map_row"],
                "curated_failures": [],
            }
            results.append(result)
            if bool((case.get("expectations") or {}).get("hard")):
                hard_failures.append(sid + ":missing")
            continue

        auto = _automatic_checks(source, target)
        for code in auto:
            auto_counts[code] += 1
        if auto:
            for cat in cats:
                category_auto_fail[cat] += 1

        curated_fail = _curated_checks(case, target) if case.get("expectations") else []
        if case.get("expectations"):
            curated_total += 1
            if not curated_fail:
                curated_pass += 1
            if curated_fail and bool(case["expectations"].get("hard")):
                hard_failures.append(sid + ":" + ",".join(curated_fail))

        results.append(
            {
                "id": sid,
                "curated": bool(case.get("expectations")),
                "hard": bool((case.get("expectations") or {}).get("hard")),
                "categories": cats,
                "status": "pass" if not auto and not curated_fail else "review",
                "automatic_failures": auto,
                "curated_failures": curated_fail,
                "translation": target,
            }
        )

    auto_pass = sum(1 for row in results if not row["automatic_failures"])
    summary = {
        "dataset_version": dataset.get("version"),
        "evaluated_cases": len(cases),
        "curated_cases": curated_total,
        "curated_pass": curated_pass,
        "curated_pass_rate": round(curated_pass / max(1, curated_total), 4),
        "hard_curated_failures": hard_failures,
        "automatic_objective_pass": auto_pass,
        "automatic_objective_pass_rate": round(auto_pass / max(1, len(cases)), 4),
        "missing": missing,
        "automatic_failures_by_code": dict(auto_counts),
        "by_category": {
            cat: {
                "total": category_total[cat],
                "automatic_failures": category_auto_fail[cat],
            }
            for cat in sorted(category_total)
        },
        "note": (
            "Automatic category checks are conservative report-only heuristics. "
            "Only curated hard assertions may fail CI. Gold/reference text is not used."
        ),
    }
    payload = {"summary": summary, "cases": results}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print("[literary-regression] " + json.dumps(summary, ensure_ascii=False))
    if args.fail_on_hard and hard_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
