from __future__ import annotations

import html as html_lib
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup


SOURCE_URL = "https://raw.githubusercontent.com/lbyshe/DeepLearningBook/master/Deep%20Learning%20-%20Goodfellow%2C%20Bengio.epub"
TARGET_MIN_CHARS = 1800
TARGET_MAX_CHARS = 3800
ANCHORS = ("second wave of neural networks", "CPU", "GPU", "LSTM")
VARIANT = (os.getenv("BOOKAI_SHORT_VARIANT") or "a").strip().casefold()
_LIGATURES = str.maketrans({"ﬁ":"fi","ﬂ":"fl","ﬀ":"ff","ﬃ":"ffi","ﬄ":"ffl","–":"-","—":"—","’":"'","“":"“","”":"”","\u200b":"","\u200c":"","\u200d":"","\ufeff":""})


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).translate(_LIGATURES).replace("\u00ad", "")
    value = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _download_epub() -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent":"Book-reader-AI cross-book benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"PK"):
        raise RuntimeError(f"Benchmark source is not an EPUB/ZIP: {data[:32]!r}")
    return data


def _epub_blocks(epub_bytes: bytes) -> list[str]:
    blocks: list[str] = []
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".html", ".xhtml", ".htm"))]
        if not names:
            raise RuntimeError("EPUB contains no HTML/XHTML content")
        for name in names:
            soup = BeautifulSoup(archive.read(name), "html.parser")
            for bad in soup(["script", "style", "svg", "math", "nav"]):
                bad.decompose()
            for node in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
                text = _norm(node.get_text(" ", strip=True))
                if len(text) >= 35:
                    blocks.append(text)
    return blocks


def _looks_corrupt(text: str) -> tuple[bool, list[str]]:
    low = str(text or "").casefold()
    sentinels = [bad for bad in ("machinelearning","neuralnetworks","deeplearning","generalpurpose","artificialneural") if bad in low]
    absurd = re.findall(r"\b[a-z]{30,}\b", low)
    return bool(sentinels or len(absurd) >= 2), sentinels + absurd[:5]


def _find_anchor_indices(blocks: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for anchor in ANCHORS:
        pos = next((i for i, block in enumerate(blocks) if anchor.casefold() in block.casefold()), None)
        if pos is None:
            raise RuntimeError(f"Goodfellow EPUB benchmark missing technical anchor {anchor!r}")
        indices[anchor] = pos
    return indices


def _merge_syntactic_continuations(rows: list[str]) -> list[str]:
    merged: list[str] = []
    for row in rows:
        current = _norm(row)
        if not current:
            continue
        if merged:
            previous = merged[-1]
            open_left = bool(re.search(r"[,;:]\s*$", previous)) or not bool(re.search(r"[.!?][\"'’”)]?\s*$", previous))
            if open_left and re.match(r"^[a-z]", current):
                merged[-1] = _norm(previous + " " + current)
                continue
        merged.append(current)
    return merged


def _short_blocks(blocks: list[str]):
    anchors = _find_anchor_indices(blocks)
    if max(anchors.values()) - min(anchors.values()) > 40:
        raise RuntimeError(f"Goodfellow smoke anchors are unexpectedly far apart: {anchors}")
    selected: set[int] = set()
    for index in anchors.values():
        selected.update(pos for pos in (index - 1, index, index + 1) if 0 <= pos < len(blocks))
    chosen = [blocks[i] for i in sorted(selected)]
    chars = sum(len(row) + 2 for row in chosen)
    radius = 2
    while chars < TARGET_MIN_CHARS and radius <= 4:
        candidate = set(selected)
        for index in anchors.values():
            for pos in (index - radius, index + radius):
                if 0 <= pos < len(blocks): candidate.add(pos)
        candidate_rows = [blocks[i] for i in sorted(candidate)]
        candidate_chars = sum(len(row) + 2 for row in candidate_rows)
        if candidate_chars <= TARGET_MAX_CHARS:
            selected, chosen, chars = candidate, candidate_rows, candidate_chars
        radius += 1
    ordered = sorted(selected)
    anchor_set = set(anchors.values())
    while sum(len(blocks[i]) + 2 for i in ordered) > TARGET_MAX_CHARS:
        removable = [i for i in ordered if i not in anchor_set]
        if not removable: break
        ordered.remove(max(removable, key=lambda i: len(blocks[i])))
    return _merge_syntactic_continuations([blocks[i] for i in ordered]), {"selection_mode":"targeted-regression-a","anchor_block_indices":anchors}, set(ordered)


def _eligible_indices(blocks: list[str]) -> list[int]:
    eligible=[]
    for i, block in enumerate(blocks):
        if not (180 <= len(block) <= 1200): continue
        if sum(ch.isalpha() for ch in block) / max(1, len(block)) < .60: continue
        if block.count(" ") < 28 or not re.search(r"[.!?]", block): continue
        if any(anchor.casefold() in block.casefold() for anchor in ANCHORS): continue
        eligible.append(i)
    return eligible


def _unseen_local_window_at(blocks: list[str], fraction: float, mode: str, excluded: set[int] | None = None):
    excluded=set(excluded or ())
    eligible=[idx for idx in _eligible_indices(blocks) if idx not in excluded]
    if len(eligible)<40: raise RuntimeError(f"Not enough clean Goodfellow prose blocks for unseen sample: {len(eligible)}")
    pivot=eligible[round(fraction*(len(eligible)-1))]
    nearby=sorted(eligible,key=lambda idx:(abs(idx-pivot),idx))
    selected=[]; chars=0
    for idx in nearby:
        if abs(idx-pivot)>20 and chars>=TARGET_MIN_CHARS: break
        size=len(blocks[idx])+2
        if chars+size>TARGET_MAX_CHARS: continue
        selected.append(idx); chars+=size
        if chars>=2600 and len(selected)>=5: break
    selected.sort()
    return _merge_syntactic_continuations([blocks[i] for i in selected]), {"selection_mode":mode,"eligible_blocks":len(eligible),"pivot_block":pivot,"selected_block_indices":selected,"excluded_known_anchors":True,"excluded_prior_variant_indices":len(excluded)}, set(selected)


def _unseen_local_window(blocks): return _unseen_local_window_at(blocks,.64,"unseen-local-window-b")
def _unseen_local_window_c(blocks):
    _,_,prior=_unseen_local_window(blocks)
    return _unseen_local_window_at(blocks,.28,"unseen-local-window-c",prior)


def _validate_selected(chosen: list[str]) -> str:
    text="\n\n".join(chosen); corrupt,evidence=_looks_corrupt(text)
    if corrupt: raise RuntimeError(f"Selected Goodfellow EPUB prose contains spacing corruption: {evidence}")
    if not (1200<=len(text)<=TARGET_MAX_CHARS+300): raise RuntimeError(f"Unexpected short Goodfellow benchmark size: {len(text)} chars")
    return text


def _fb2(target_paragraphs: list[str], memory_paragraphs: list[str]) -> str:
    target_body="\n".join(f"      <p>{html_lib.escape(row)}</p>" for row in target_paragraphs)
    memory_body="\n".join(f"      <p>{html_lib.escape(row)}</p>" for row in memory_paragraphs)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><description><title-info><genre>science</genre><author><first-name>Ian</first-name><last-name>Goodfellow</last-name></author><author><first-name>Yoshua</first-name><last-name>Bengio</last-name></author><author><first-name>Aaron</first-name><last-name>Courville</last-name></author><book-title>Deep Learning — benchmark</book-title><lang>en</lang></title-info><document-info><id>bookai-goodfellow</id><version>1.0</version></document-info></description><body><section><title><p>Chapter One</p></title>
{target_body}
</section><section><title><p>Chapter Two</p></title>
{memory_body}
</section></body></FictionBook>'''


def _memory_rows(blocks, selected_indices): return [_norm(block) for i,block in enumerate(blocks) if i not in selected_indices and len(_norm(block))>=45]


def main() -> None:
    out_dir=Path(sys.argv[1] if len(sys.argv)>1 else "crossbook-goodfellow"); out_dir.mkdir(parents=True,exist_ok=True)
    blocks=_epub_blocks(_download_epub())
    if VARIANT in {"c","fresh","unseen-c"}: selected,selection_meta,selected_indices=_unseen_local_window_c(blocks)
    elif VARIANT in {"b","alt","unseen"}: selected,selection_meta,selected_indices=_unseen_local_window(blocks)
    else: selected,selection_meta,selected_indices=_short_blocks(blocks)
    source_text=_validate_selected(selected); memory_rows=_memory_rows(blocks,selected_indices)
    if VARIANT not in {"b","alt","unseen","c","fresh","unseen-c"}:
        required=("deep learning","LSTM","CPU","GPU"); missing=[t for t in required if t.casefold() not in source_text.casefold()]
        if missing: raise RuntimeError(f"Short benchmark lost required technical coverage; missing={missing}")
    (out_dir/"sample.fb2").write_text(_fb2(selected,memory_rows),"utf-8")
    # Stable independent BookMemory source: every clean EPUB block in original order.
    stable_blocks=[_norm(block) for block in blocks if len(_norm(block))>=45]
    (out_dir/"memory-source.fb2").write_text(_fb2(stable_blocks,[]),"utf-8")
    (out_dir/"sample-source.txt").write_text(source_text,"utf-8")
    meta={"variant":VARIANT,"source_url":SOURCE_URL,"source_format":"epub-xhtml","source_chars":len(source_text),"segments":len(selected),"memory_scope":"stable-whole-book-epub","memory_segments":len(stable_blocks),"memory_chars":sum(len(r) for r in stable_blocks),"reference_text_embedded":False,**selection_meta}
    (out_dir/"sample-meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),"utf-8"); print(json.dumps(meta,ensure_ascii=False))


if __name__=="__main__": main()
