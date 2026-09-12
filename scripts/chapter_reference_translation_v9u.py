from __future__ import annotations

import json, os, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import chapter_reference_translation_v9t as v9t

v9s, v9r, v9, v8, v6, v3 = v9t.v9s, v9t.v9s.v9r, v9t.v9, v9t.v8, v9t.v6, v9t.v3
_BASE_QUALITY = v9t._quality_v9t
_BASE_DETERMINISTIC = v9s._deterministic_v9s
_BASE_ENTITY_MAP = v8._entity_map
_V9U_STATS: dict[str, Any] = {}
_ENTITY_OVERRIDES: dict[str, str] = {}
_RISK_RE = re.compile(r"\b(?:outnumber(?:ed|ing)?|either|neither|both|former|latter|rather\s+than|instead\s+of|last\s+[^.!?]{0,35}\s+but\s+one|can(?:not|'t)\s+be\s+too|no\s+more\s+than|no\s+less\s+than|at\s+most|at\s+least|unless|hardly|scarcely)\b", re.I)
_AUX_RE = re.compile(r"\b(?:am|is|are|was|were|be|been|do|does|did|have|has|had|can|could|shall|should|will|would|may|might|must)\b", re.I)


def _blocking(row): return v9s._blocking(row)
def _semantic(rows): return v9t._semantic_rows(rows)


def _entity_map(memory):
    out = dict(_BASE_ENTITY_MAP(memory)); out.update(_ENTITY_OVERRIDES); return out


def _proper_name_material_only(source: str) -> bool:
    if re.search(r"\b(?:gear|spring|metal|blade|wire|ore|sheet|rod|bar|plate|alloy|material|cast|forg|tool|machine|mechanism|timber|lumber)\w*\b", source.casefold()):
        return False
    found = False
    for material in v9._MATERIALS:
        for m in re.finditer(rf"\b{re.escape(material)}\b", source, re.I):
            found = True
            if not m.group(0)[:1].isupper(): return False
            left = (source[:m.start()].rstrip().split() or [""])[-1].strip(".,;:!?()[]{}\"'")
            right = (source[m.end():].lstrip().split() or [""])[0].strip(".,;:!?()[]{}\"'")
            if not ((left[:1].isupper() and left[1:].islower()) or (right[:1].isupper() and right[1:].islower())):
                return False
    return found


def _deterministic(segment, candidate, memory):
    rows = [dict(x) for x in _BASE_DETERMINISTIC(segment, candidate, memory)]
    if _proper_name_material_only(segment.text):
        rows = [x for x in rows if x.get("code") != "material"]
    return rows


def _ctx(targets, translated, i):
    s = targets[i]
    return {"id": s.id, "source": s.text, "current_ru": str(translated.get(s.id) or ""),
            "before_en": [x.text for x in targets[max(0, i-2):i]], "after_en": [x.text for x in targets[i+1:i+3]],
            "before_ru": [str(translated.get(x.id) or "") for x in targets[max(0, i-2):i]],
            "after_ru": [str(translated.get(x.id) or "") for x in targets[i+1:i+3]]}


def _heuristic(targets):
    out = []
    for s in targets:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s.text)
        reason = None
        if _RISK_RE.search(s.text): reason = "verify exact comparison/quantifier/polarity/referent direction"
        elif len(words) <= 7 and _AUX_RE.search(s.text) and v9._is_dialogue(s.text): reason = "verify short elliptical dialogue and omitted predicate from context"
        if reason:
            out.append({"id": s.id, "severity": "major", "confidence": .80, "code": "challenge_risk", "source_span": s.text[:180], "target_span": "", "reason": reason, "repairability": "local"})
    return out


def _scan(harness, targets, translated):
    items = [_ctx(targets, translated, i) for i in range(len(targets))]
    cap = max(6500, int(os.getenv("BOOKAI_V9U_SCAN_BATCH_CHARS") or "13500")); batches=[]; cur=[]; size=0
    for item in items:
        w = len(item["source"])+len(item["current_ru"])+sum(map(len,item["before_en"]+item["after_en"]))
        if cur and size+w > cap: batches.append(cur); cur=[]; size=0
        cur.append(item); size += w
    if cur: batches.append(cur)
    system = """You are a conservative challenge scanner for EN→RU literary translation. Do not rewrite. Using source, current_ru and neighboring context, return only obvious publication defects broad QE can miss: idiom/pragmatics; reversed comparison/quantifier; negation/modality; short dialogue ellipsis; wrong actor/role/word sense; referent/gender; technical term corruption; meaning-changing calque; broken grammar/quotes; clearly distorted proper-name transliteration. Do not flag stylistic taste, valid paraphrase or equally plausible transliteration. Max one issue/item. ONLY JSON {\"issues\":[{\"id\":\"...\",\"code\":\"idiom|comparison|polarity|ellipsis|role|referent|gender|term|calque|grammar|name_transliteration\",\"severity\":\"critical|major\",\"confidence\":0.0,\"source_span\":\"...\",\"target_span\":\"...\",\"reason\":\"...\"}]}"""
    def one(batch):
        try: return (v8._complete_json(harness.gate, system, {"items": batch}).get("issues") or [])
        except Exception as e: print(f"[v9u-scan] {type(e).__name__}", flush=True); return []
    raw=[]; workers=max(1,min(4,int(os.getenv("BOOKAI_V9U_SCAN_WORKERS") or "4"),len(batches) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one,b) for b in batches]): raw.extend(f.result())
    ids={s.id for s in targets}; out=[]
    for r in raw:
        if not isinstance(r,dict) or str(r.get("id") or "") not in ids: continue
        sev=str(r.get("severity") or "major").lower()
        try: conf=float(r.get("confidence") or 0)
        except Exception: conf=0
        if sev not in {"critical","major"} or conf < .72: continue
        out.append({"id":str(r["id"]),"severity":sev,"confidence":min(1,max(0,conf)),"code":str(r.get("code") or "challenge"),"source_span":v9._norm_text(r.get("source_span") or "")[:220],"target_span":v9._norm_text(r.get("target_span") or "")[:220],"reason":v9._norm_text(r.get("reason") or "likely defect")[:500],"repairability":"local"})
    return out


def _verify(harness, targets, translated, rows):
    rows=[dict(r) for r in rows if _blocking(r) and r.get("code") not in {"entity","thread"}]
    if not rows: return []
    groups={}
    for r in rows: groups.setdefault(str(r.get("id") or ""),[]).append(r)
    gl=list(groups.values()); workers=max(1,min(5,int(os.getenv("BOOKAI_V9U_VERIFY_WORKERS") or "5"),len(gl)))
    bins=[[] for _ in range(workers)]; weights=[0]*workers
    for g in sorted(gl,key=lambda x:sum(len(str(y)) for y in x),reverse=True):
        i=min(range(workers),key=lambda n:weights[n]); bins[i]+=g; weights[i]+=sum(len(str(y)) for y in g)
    out=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(v9s._verify_llm_findings,harness,targets,translated,b) for b in bins if b]
        for f in as_completed(fs):
            try: out.extend(f.result())
            except Exception as e: print(f"[v9u-verify] {type(e).__name__}",flush=True)
    return out


def _review_entities(harness, targets, translated, memory):
    em=dict(_BASE_ENTITY_MAP(memory)); counts=Counter()
    for s in targets:
        for src in em:
            if len(src)>=4 and v8._source_mentions(s.text,src): counts[src]+=1
    items=[{"source":s,"current_ru":em[s],"occurrences":n} for s,n in counts.most_common(30) if n>=3 and em.get(s)]
    if not items: return {},set()
    system="""Review recurring EN proper-name canonicals in an EN→RU literary translation. Be very conservative. Correct only a clearly distorted Russian transliteration that drops/invents source syllables or is malformed; do not change equally plausible established forms. ONLY JSON {\"corrections\":[{\"source\":\"...\",\"old_ru\":\"...\",\"new_ru\":\"...\",\"confidence\":0.0}]}"""
    try: raw=v8._complete_json(harness.gate,system,{"entities":items}).get("corrections") or []
    except Exception as e: print(f"[v9u-entities] {type(e).__name__}",flush=True); raw=[]
    overrides={}; changed=set()
    for r in raw:
        if not isinstance(r,dict): continue
        src=str(r.get("source") or ""); old=v9._norm_text(r.get("old_ru") or em.get(src) or ""); new=v9._norm_text(r.get("new_ru") or "")
        try: conf=float(r.get("confidence") or 0)
        except Exception: conf=0
        if src not in em or conf<.88 or not old or not new or old==new or len(old.split())>3 or len(new.split())>3: continue
        overrides[src]=new; pat=re.compile(rf"(?<![А-Яа-яЁё]){re.escape(old)}(?![А-Яа-яЁё])",re.I)
        for s in targets:
            if not v8._source_mentions(s.text,src): continue
            val,n=pat.subn(new,str(translated.get(s.id) or ""))
            if n: translated[s.id]=val; changed.add(s.id)
    return overrides,changed


def _merge_verified(base, verified):
    by={}
    for r in verified: by.setdefault(str(r.get("id") or ""),[]).append(dict(r))
    out={}
    for sid,rows in base.items():
        keep=[dict(r) for r in rows if not _blocking(r)]+by.pop(sid,[])
        if keep: out[sid]=keep
    out.update({sid:rows for sid,rows in by.items() if rows}); return out


def _payload(targets, translated, memory, i, findings):
    p=_ctx(targets,translated,i); low=targets[i].text.casefold(); gl=[]
    for k,v in dict(getattr(memory,"glossary",{}) or {}).items():
        if str(k).casefold() in low: gl.append({"source":str(k),"ru":str(v)})
        if len(gl)>=12: break
    p.update({"issues":[{"code":r.get("code"),"reason":r.get("reason"),"source_span":r.get("source_span"),"target_span":r.get("target_span")} for r in findings[:6]],"glossary":gl,"book_context":v9._norm_text(getattr(memory,"rolling_summary",""))[:1200]}); return p


def _three(harness,p):
    system="""Retranslate ONE EN segment into publication-quality Russian because independent QE found a likely defect. Do not edit word-by-word from current_ru. Produce three complete fresh alternatives: faithful (exact relations/actor/polarity/numbers/idiom), literary (same exact meaning but natural professional Russian, free to split/reorder English syntax), contextual (resolve ellipsis/referents/speaker intent using neighbors). Preserve valid glossary terms; never add facts. ONLY JSON {\"faithful\":\"...\",\"literary\":\"...\",\"contextual\":\"...\"}."""
    try: obj=v8._complete_json(harness.gate,system,p)
    except Exception as e: print(f"[v9u-candidates] {p.get('id')} {type(e).__name__}",flush=True); return []
    out=[]
    for k in ("faithful","literary","contextual"):
        x=v9._norm_text(obj.get(k) or "")
        if x and x not in out: out.append(x)
    return out


def _judge(harness,p,cands):
    system="""Select the best complete EN→RU literary candidate. Priority: exact source meaning/actor/relation/polarity/chronology/numbers/technical facts; correct idiom/ellipsis/referent from context; then natural professional Russian literary prose without English calque; then continuity of names/register/voice. A fluent mistranslation always loses. ONLY JSON {\"best_index\":0}."""
    try:
        i=int(v8._complete_json(harness.gate,system,{**p,"candidates":cands}).get("best_index")); return i if 0<=i<len(cands) else 0
    except Exception: return 0


def _repair(harness,targets,translated,memory,fmap,scores):
    pos={s.id:i for i,s in enumerate(targets)}
    ids=[s.id for s in targets if any(_blocking(r) and r.get("code") not in {"entity","thread"} for r in fmap.get(s.id,[]))]
    ids.sort(key=lambda sid:scores.get(sid,100)); ids=ids[:max(0,int(os.getenv("BOOKAI_V9U_RETRANSLATE_MAX") or "24"))]
    workers=max(1,min(6,int(os.getenv("BOOKAI_V9U_RETRANSLATE_WORKERS") or "6"),len(ids) or 1))
    def one(sid):
        i=pos[sid]; seg=targets[i]; cur=str(translated.get(sid) or ""); rows=[r for r in fmap.get(sid,[]) if _blocking(r) and r.get("code") not in {"entity","thread"}]; p=_payload(targets,translated,memory,i,rows)
        safe=[cur]+[x for x in _three(harness,p) if v9._fatal_count(seg,x,memory)<=v9._fatal_count(seg,cur,memory)]
        c=[]; seen=set()
        for x in safe:
            k=v9._norm_text(x)
            if k and k not in seen: seen.add(k); c.append(x)
        if len(c)<2: return sid,""
        x=c[_judge(harness,p,c)]; return (sid,"") if v9._norm_text(x)==v9._norm_text(cur) else (sid,x)
    res={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one,sid) for sid in ids]):
            sid,x=f.result();
            if x: res[sid]=x
    changed=[]
    for sid in ids:
        if sid in res: translated[sid]=res[sid]; changed.append(sid)
    return changed,ids


def _quality(harness,targets,translated,memory):
    global _V9U_STATS,_ENTITY_OVERRIDES
    stats=dict(_BASE_QUALITY(harness,targets,translated,memory) or {})
    overrides,entity_changed=_review_entities(harness,targets,translated,memory); _ENTITY_OVERRIDES=dict(overrides); v8._entity_map=_entity_map
    sanitized=0
    for s in targets:
        x,n=v9s._format_dialogue_v9s(s,translated.get(s.id,"")); translated[s.id]=x; sanitized+=n
    base={sid:_semantic(rows) for sid,rows in v9._V9_FINAL_FINDINGS.items()}
    old_block=[r for rows in base.values() for r in rows if _blocking(r)]; old_verified=_verify(harness,targets,translated,old_block); semantic=_merge_verified(base,old_verified)
    scan=_scan(harness,targets,translated); heur=_heuristic(targets); confirmed=_verify(harness,targets,translated,[*scan,*heur])
    for r in confirmed: semantic.setdefault(str(r.get("id") or ""),[]).append(dict(r))
    fmap,scores,det_before=v9t._rebuild_full_map(targets,translated,memory,semantic); changed,selected=_repair(harness,targets,translated,memory,fmap,scores)
    changed_set=set(changed)|set(entity_changed)
    for sid in changed_set:
        s=next((x for x in targets if x.id==sid),None)
        if s: translated[sid]=v9s._format_dialogue_v9s(s,translated.get(sid,""))[0]
    remaining=[]
    if changed_set:
        fresh=v9t._selected_audit(harness,targets,translated,memory,changed_set); by={}
        for r in fresh: by.setdefault(str(r.get("id") or ""),[]).append(dict(r))
        for sid in changed_set: semantic[sid]=_semantic(by.get(sid,[]))
        alleged=[r for sid in changed_set for r in semantic.get(sid,[]) if _blocking(r)]; remaining=_verify(harness,targets,translated,alleged); vb={}
        for r in remaining: vb.setdefault(str(r.get("id") or ""),[]).append(dict(r))
        for sid in changed_set: semantic[sid]=[r for r in semantic.get(sid,[]) if not _blocking(r)]+vb.get(sid,[])
    for s in targets:
        x,n=v9s._format_dialogue_v9s(s,translated.get(s.id,"")); translated[s.id]=x; sanitized+=n
    fmap,scores,det_final=v9t._rebuild_full_map(targets,translated,memory,semantic); critical,major=v9s._publish_final_state(targets,fmap,scores); counts=Counter(r.get("severity") for rows in fmap.values() for r in rows)
    _V9U_STATS={"challenge_scan_all_segments":True,"challenge_scan_raw":len(scan),"heuristic_challenges":len(heur),"challenge_confirmed":len(confirmed),"existing_blockers_reverified":len(old_block),"existing_blockers_confirmed":len(old_verified),"priority_invariants_are_allegations":True,"entity_overrides":dict(overrides),"entity_segments_changed":len(entity_changed),"multi_candidate_selected":len(selected),"multi_candidate_changed":len(changed),"multi_candidate_changed_ids":changed,"delta_remaining_verified":len(remaining),"typography_segments_sanitized":sanitized,"deterministic_before_retranslation":det_before,"final_deterministic":det_final,"remaining_critical":len(critical),"remaining_major_high_confidence":len(major),"mean_quality_score":round(sum(scores.values())/max(1,len(scores)),2),"severity_counts":dict(counts),"quality_mode":"v9t-fast+all-segment-challenge+priority-consensus+3way-retranslation+literary-judge+delta-reaudit"}
    stats.update(_V9U_STATS); print("[bookai-v9u] "+json.dumps(_V9U_STATS,ensure_ascii=False),flush=True); return stats


def _annotate():
    if not v3.REPORT.exists(): return
    try: data=json.loads(v3.REPORT.read_text("utf-8"))
    except Exception: return
    data["architecture"]={**dict(data.get("architecture") or {}),"version":"quality-v9u-selective-mbr-literary-qe","gold_reference_available_to_pipeline":False,"challenge_scanner":"all final segments, source+RU+local context","priority_policy":"priority_invariant is an allegation requiring independent verification","repair_policy":"3 fresh candidates (faithful/literary/contextual) + source-aware literary selection","entity_policy":"conservative source-guided recurring-name canonical review","performance":"retain v9t fast path; expensive generation only for confirmed challenges","publication_gate":"confirmed critical or major after delta re-audit blocks completion"}; data["v9u_stats"]=dict(_V9U_STATS); v3.REPORT.write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def main():
    v9._configure_v9(); v9._MATERIALS["coal"]=("угл",); v9s._deterministic_v9s=_deterministic; v9._deterministic_findings=_deterministic; v9._thread_findings=v9s._thread_v9s; v8._fix_near_entity_typos=v9s._entity_fix_v9s; v9._format_dialogue_v9=v9s._format_dialogue_v9s; v8._normalize_dialogue_v8=v9s._format_dialogue_v9s; v3._semantic_short_repair=_quality
    try: v3.main()
    finally:
        v6._annotate_report(); v9r.v9j._annotate_v9j(); v9r.v9k._annotate_v9k(); v9r._annotate_v9r(); v9s._annotate_v9s(); v9t._annotate_v9t(); _annotate()

if __name__ == "__main__": main()
