from __future__ import annotations

from pathlib import Path

from bookai.bulk_mt import AzureTranslatorBackend, BulkMTClient, OPUSMTBackend, azure_dictionary_markup
from bookai.models import BookMemory


def test_azure_dictionary_marks_locked_proper_name():
    marked = azure_dictionary_markup(
        "Valens picked up the lever.",
        {"Valens": "Валенс", "Perpetual Republic": "Вечная Республика"},
    )
    assert '<mstrans:dictionary translation="Валенс">Valens</mstrans:dictionary>' in marked


def test_azure_dictionary_does_not_force_possessive_name_form():
    marked = azure_dictionary_markup("Valens's sword was on the table.", {"Valens": "Валенс"})
    assert "mstrans:dictionary" not in marked
    assert "Valens's" in marked


def test_azure_backend_is_optional_without_key(monkeypatch):
    monkeypatch.delenv("AZURE_TRANSLATOR_KEY", raising=False)
    backend = AzureTranslatorBackend()
    assert backend.available() is False


def test_opus_backend_requires_complete_model(tmp_path: Path):
    backend = OPUSMTBackend(tmp_path / "missing")
    assert backend.available() is False


def test_bulk_client_uses_opus_when_azure_is_absent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AZURE_TRANSLATOR_KEY", raising=False)
    model = tmp_path / "opus"
    model.mkdir()
    for name in ("model.bin", "source.spm", "target.spm"):
        (model / name).write_bytes(b"stub")
    client = BulkMTClient(azure=AzureTranslatorBackend(), opus=OPUSMTBackend(model))
    assert client.available() is True
    assert client.backend_name == "opus-mt-en-ru-ctranslate2-int8"


def test_bulk_client_prefers_azure_when_configured(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "test-key")
    client = BulkMTClient(
        azure=AzureTranslatorBackend(),
        opus=OPUSMTBackend(tmp_path / "missing"),
    )
    assert client.available() is True
    assert client.backend_name == "azure-translator-v3+opus-fallback"


def test_markup_escapes_plain_html():
    marked = azure_dictionary_markup("A < B & Valens > C", {"Valens": "Валенс"})
    assert "A &lt; B &amp;" in marked
    assert "&gt; C" in marked


def test_empty_memory_is_valid_for_bulk_layer():
    memory = BookMemory()
    assert isinstance(memory.glossary, dict)
