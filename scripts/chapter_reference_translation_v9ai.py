from __future__ import annotations

import json
import os

import chapter_reference_translation_v3 as v3
import chapter_reference_translation_v9 as v9
import chapter_reference_translation_v9ah as v9ah
import chapter_reference_translation_v9z as v9z
from bookai.bergamot_editor import BergamotGigaLiteraryEditorBackend


_BASE_CONFIGURE_V9 = v9._configure_v9
_BASE_DISABLE_REFERENCE = v9z._disable_reference_runtime

# These defects are deliberately allowed through the *immediate* bulk acceptance
# gate in the Bergamot-editor experiment. They are owned by later v9 mechanisms:
# book-wide name canon / entity fixer and the validated source-grounded sanitizer.
# Everything structural or semantic remains a hard early rejection.
_V9AI_LATE_OWNED_HARD_CODES = {
    "latin_residue",
    "character_name",
    "entity_consistency",
}


def _candidate_bad_v9ai(segment, candidate: str, memory, source_segments: list) -> bool:
    issues = v3.enhanced_candidate_issues(
        segment,
        candidate,
        memory,
        source_segments=source_segments,
    )
    hard = [issue for issue in issues if issue.severity == "hard"]
    blocking = [issue for issue in hard if issue.code not in _V9AI_LATE_OWNED_HARD_CODES]
    if blocking:
        print(
            "[v9ai-immediate-reject] "
            + json.dumps(
                {
                    "id": segment.id,
                    "codes": sorted({issue.code for issue in blocking}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return True
    if hard:
        print(
            "[v9ai-deferred-qa] "
            + json.dumps(
                {
                    "id": segment.id,
                    "codes": sorted({issue.code for issue in hard}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return False


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

    @staticmethod
    def _parse_complete_editor_response(response, batch):
        raw = str(response.choices[0].message.content or "")
        parsed = BergamotGigaLiteraryEditorBackend._parse_json_object(raw)
        expected = [segment.id for segment in batch]
        expected_set = set(expected)
        translated = {
            sid: str(parsed.get(sid) or "").strip()
            for sid in expected
            if str(parsed.get(sid) or "").strip()
        }
        actual = set(translated)
        if actual != expected_set:
            raise ValueError(
                "editor id contract violation: "
                f"missing={sorted(expected_set - actual)[:12]} "
                f"extra={sorted(set(parsed) - expected_set)[:12]}"
            )
        return translated

    def _record_editor_changes(self, batch, translated, drafts) -> None:
        for segment in batch:
            if segment.id not in translated:
                continue
            self._editor_seen_ids.add(segment.id)
            if " ".join(translated[segment.id].split()) != " ".join(drafts[segment.id].split()):
                self._editor_changed_ids.add(segment.id)
        if self._editor_seen_ids:
            print(
                "[bergamot-editor] "
                + json.dumps(
                    {
                        "seen_unique": len(self._editor_seen_ids),
                        "changed_unique": len(self._editor_changed_ids),
                        "changed_percent": round(
                            len(self._editor_changed_ids) / max(1, len(self._editor_seen_ids)) * 100,
                            1,
                        ),
                        "local_usage": self.local_usage,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    def _translate_batch(self, batch, memory, *, source_segments=None, minimal=False):
        """Retry the SAME batch in plain-JSON mode before recursive batch splitting."""
        drafts = self._local_drafts(batch)
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты литературный редактор русского перевода. Английский оригинал — источник истины; "
                    "машинный русский черновик — только заготовка. Верни публикационный русский текст."
                ),
            },
            {
                "role": "user",
                "content": self._editor_prompt(
                    batch,
                    memory,
                    source_segments=source_segments,
                    minimal=minimal,
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "response_format": self._strict_response_format(batch),
        }

        strict_response = None
        fallback_reason = ""
        try:
            strict_response = self._chat(payload, batch, strict=True)
            try:
                translated = self._parse_complete_editor_response(strict_response, batch)
            except Exception as exc:
                # This API call happened; account for it now because the inherited
                # resilient layer will account for the successful fallback response.
                self.usage.add(self._usage(strict_response), calls=1)
                fallback_reason = f"{type(exc).__name__}: {exc}"
                raise
        except Exception as exc:
            if strict_response is None and not self._schema_incompatibility(exc):
                raise
            if not fallback_reason:
                fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                "[bergamot-editor-json-retry] "
                + json.dumps(
                    {
                        "segments": len(batch),
                        "first": batch[0].id if batch else None,
                        "last": batch[-1].id if batch else None,
                        "reason": fallback_reason[:260],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            fallback = dict(payload)
            fallback.pop("response_format", None)
            fallback["messages"] = [dict(messages[0]), dict(messages[1])]
            fallback["messages"][0]["content"] += (
                " Верни ТОЛЬКО валидный JSON-объект: ключи — ровно все переданные ID, "
                "значения — полные русские тексты. Без markdown и комментариев."
            )
            response = self._chat(fallback, batch, strict=False)
            translated = self._parse_complete_editor_response(response, batch)
            usage = self._usage(response)
        else:
            response = strict_response
            usage = self._usage(response)

        self._record_editor_changes(batch, translated, drafts)
        return translated, usage

    def translate_many(self, segments, memory, *, source_segments=None):
        result = super().translate_many(
            segments,
            memory,
            source_segments=source_segments,
        )
        # Snapshot after the outer resilient layer has accounted for final usage.
        type(self).last_experiment_stats = self.experiment_stats()
        return result


def _configure_v9ai() -> None:
    # Let the normal production configuration happen first, then replace ONLY its
    # bulk backend so every downstream v9ah mechanism stays unchanged.
    _BASE_CONFIGURE_V9()
    v3.GigaChatLightningV3Backend = BergamotGigaV9EditorBackend
    print(
        "[v9ai-injection] backend=bergamot-base-enru+gigachat-v9-literary-editor "
        "weighted_self_tm=preserved",
        flush=True,
    )


def _disable_reference_runtime_v9ai() -> None:
    # Preserve v9z's source-only/gold-blind runtime cleanup, then install the
    # ownership-aware immediate gate for this experiment.
    _BASE_DISABLE_REFERENCE()
    v3._candidate_bad_v3 = _candidate_bad_v9ai
    print(
        "[v9ai-qa-ownership] immediate=catastrophic+semantic "
        "deferred=latin-residue+name/entity-consistency",
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
            "immediate_qa": "catastrophic/semantic only; latin/name/entity defects deferred to downstream owners",
            "editor_json_recovery": "same-batch plain-JSON retry before recursive split",
            "gold_reference_available_to_pipeline": False,
            "experiment": True,
        }
    )
    data["architecture"] = architecture
    data["v9ai_experiment"] = stats
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    original_configure = v9._configure_v9
    original_disable = v9z._disable_reference_runtime
    v9._configure_v9 = _configure_v9ai
    v9z._disable_reference_runtime = _disable_reference_runtime_v9ai
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
        v9._configure_v9 = original_configure
        v9z._disable_reference_runtime = original_disable


if __name__ == "__main__":
    main()
