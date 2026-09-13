from __future__ import annotations

import json

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v9ah as v9ah
from bookai.bergamot_editor import BergamotGigaLiteraryEditorBackend

_BASE_BULK = v3.GigaChatLightningV3Backend


def _annotate() -> None:
    report = getattr(v3, "REPORT", None)
    if report is None or not report.exists():
        return
    try:
        data = json.loads(report.read_text("utf-8"))
    except Exception:
        return
    architecture = dict(data.get("architecture") or {})
    architecture.update(
        {
            "version": "quality-v9ai-bergamot-draft-gigachat-editor-ab",
            "primary_draft": "Firefox/Bergamot base en->ru local CPU model",
            "primary_editor": "GigaChat-3-Lightning source-grounded literary editor",
            "downstream": "v9ah adaptive sparse DeepSeek + v9ag validated sanitizer",
            "gold_reference_available_to_pipeline": False,
            "experiment": True,
        }
    )
    data["architecture"] = architecture
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    v3.GigaChatLightningV3Backend = BergamotGigaLiteraryEditorBackend
    try:
        v9ah.main()
    finally:
        v3.GigaChatLightningV3Backend = _BASE_BULK
        _annotate()


if __name__ == "__main__":
    main()
