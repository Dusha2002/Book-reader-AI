from __future__ import annotations

import json
import os

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v9 as v9
import chapter_reference_translation_v9ah as v9ah
from bookai.bergamot_editor import BergamotGigaLiteraryEditorBackend


_BASE_CONFIGURE_V9 = v9._configure_v9


class BergamotGigaV9EditorBackend(
    BergamotGigaLiteraryEditorBackend,
    v9.GigaChatLightningV9Backend,
):
    """v9 weighted self-TM + local Bergamot draft + GigaChat literary editor."""

    name = "bergamot-base-enru+gigachat-v9-literary-editor"
    last_experiment_stats: dict = {}

    def _editor_prompt(self, batch, memory, *, source_segments=None, minimal=False):
        base = BergamotGigaLiteraryEditorBackend._editor_prompt(
            self,
            batch,
            memory,
            source_segments=source_segments,
            minimal=minimal,
        )
        examples = v9.GigaChatLightningV9Backend._retrieve_weighted(
            batch,
            int(os.getenv("BOOKAI_TM_RETRIEVAL_K") or "4"),
        )
        if not examples:
            return base
        payload = [
            {
                "source_en": row["source"][:1100],
                "approved_ru": row["translation"][:1300],
                "tier": row["tier"],
                "quality": round(float(row.get("quality") or 0.8), 3),
                "relevance": row["retrieval_score"],
            }
            for row in examples
        ]
        return (
            base
            + "\n\nWEIGHTED_SELF_TRANSLATION_MEMORY (same source book only):\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\nUse this only for recurring names, terminology, register and authorial voice. "
              "Never override the current SOURCE_EN."
        )

    def _local_drafts(self, batch):
        result = BergamotGigaLiteraryEditorBackend._local_drafts(self, batch)
        type(self).last_experiment_stats = self.experiment_stats()
        return result

    def _translate_batch(self, batch, memory, *, source_segments=None, minimal=False):
        result = BergamotGigaLiteraryEditorBackend._translate_batch(
            self,
            batch,
            memory,
            source_segments=source_segments,
            minimal=minimal,
        )
        type(self).last_experiment_stats = self.experiment_stats()
        return result


def _configure_v9ai() -> None:
    # v9x/v9ah calls v9._configure_v9 immediately before v3.main(). Let the normal
    # production configuration happen first, then replace ONLY its bulk backend.
    _BASE_CONFIGURE_V9()
    v3.GigaChatLightningV3Backend = BergamotGigaV9EditorBackend
    print(
        "[v9ai-injection] backend=bergamot-base-enru+gigachat-v9-literary-editor "
        "weighted_self_tm=preserved",
        flush=True,
    )


def _annotate(stats: dict) -> None:
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
            "translation_memory": "v9 trusted+working weighted self-TM preserved",
            "downstream": "v9ah adaptive sparse DeepSeek + v9ag validated sanitizer",
            "gold_reference_available_to_pipeline": False,
            "experiment": True,
        }
    )
    data["architecture"] = architecture
    data["v9ai_experiment"] = stats
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    original = v9._configure_v9
    v9._configure_v9 = _configure_v9ai
    try:
        v9ah.main()
        stats = dict(BergamotGigaV9EditorBackend.last_experiment_stats or {})
        segments = int(dict(stats.get("bergamot") or {}).get("segments") or 0)
        if segments <= 0:
            raise RuntimeError(
                "v9ai A/B invalid: production chapter translated zero segments through Bergamot"
            )
        print("[v9ai-experiment] " + json.dumps(stats, ensure_ascii=False), flush=True)
        _annotate(stats)
    finally:
        v9._configure_v9 = original


if __name__ == "__main__":
    main()
