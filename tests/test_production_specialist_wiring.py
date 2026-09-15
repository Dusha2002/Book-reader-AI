from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import production_literary_translation as production


class DummyUltra:
    model = "GigaChat-3-Ultra"
    usage = {"requests": 0}


def test_specialist_repair_swaps_only_gate_and_restores_it():
    strategy = production.LiteraryTranslationStrategy()
    dummy = DummyUltra()
    strategy._specialist_provider = dummy
    seen = {}

    def base(harness, targets, translated, memory, routes):
        seen["during"] = harness.gate
        return ["s1"], 1

    strategy._base_deep_repair = base
    original_gate = object()
    harness = SimpleNamespace(gate=original_gate)

    changed, calls = strategy.specialist_repair(harness, [], {}, None, [{"id": "s1"}])

    assert seen["during"] is dummy
    assert harness.gate is original_gate
    assert changed == ["s1"]
    assert calls == 1


def test_specialist_verify_reuses_same_ultra_and_restores_gate():
    strategy = production.LiteraryTranslationStrategy()
    dummy = DummyUltra()
    strategy._specialist_provider = dummy
    seen = {}

    def base(harness, targets, translated, memory, routes):
        seen["during"] = harness.gate
        return [], 2, 1

    strategy._base_deep_verify = base
    original_gate = object()
    harness = SimpleNamespace(gate=original_gate)

    changed, confirmed, calls = strategy.specialist_verify(harness, [], {}, None, [{"id": "s1"}])

    assert seen["during"] is dummy
    assert harness.gate is original_gate
    assert changed == []
    assert confirmed == 2
    assert calls == 1


def test_objective_sanitizer_rejects_candidate_with_same_residual(monkeypatch):
    strategy = production.LiteraryTranslationStrategy()

    def issues(_targets, translated):
        if "anyway" in str(translated.get("s1") or ""):
            return [{"id": "s1", "index": 0, "codes": ["latin_leak"], "reason": "latin"}]
        return []

    monkeypatch.setattr(production.legacy_kernel.v9ag, "_objective_issues", issues)

    assert not strategy._candidate_clears_objective_issues([], {"s1": "было anyway"}, "s1", "всё ещё anyway")
    assert strategy._candidate_clears_objective_issues([], {"s1": "было anyway"}, "s1", "всё равно")


def test_release_fallback_accepts_only_validated_clean_candidate(monkeypatch):
    strategy = production.LiteraryTranslationStrategy()
    targets = [SimpleNamespace(id="s1", text="He had never wanted it anyway.")]
    translated = {"s1": "Он никогда этого не хотел anyway."}
    issue = {"id": "s1", "index": 0, "codes": ["latin_leak"], "reason": "latin"}

    def issues(_targets, values):
        if "anyway" in str(values.get("s1") or ""):
            return [issue]
        return []

    calls = {"n": 0}

    def complete_json(_provider, _system, _payload):
        calls["n"] += 1
        return {"items": [{"id": "s1", "corrected_ru": "Он всё равно никогда этого не хотел."}]}

    monkeypatch.setattr(production.legacy_kernel.v9ag, "_objective_issues", issues)
    monkeypatch.setattr(production.legacy_kernel.v9ag.v9ab.v8, "_complete_json", complete_json)
    harness = SimpleNamespace(hard_editor=SimpleNamespace(model="deepseek-flash"))

    changed, api_calls = strategy._release_residual_fallback(harness, targets, translated, [issue])

    assert changed == ["s1"]
    assert translated["s1"] == "Он всё равно никогда этого не хотел."
    assert api_calls == 1
    assert calls["n"] == 1


def test_run_patches_frozen_v9ah_specialist_callsites(monkeypatch):
    strategy = production.LiteraryTranslationStrategy()
    dummy = DummyUltra()
    strategy._specialist_provider = dummy
    original_gate = object()
    harness = SimpleNamespace(gate=original_gate)
    observed = {"repair": False, "verify": False}

    def frozen_repair(harness_arg, targets, translated, memory, routes):
        assert harness_arg.gate is dummy
        observed["repair"] = True
        return [], 1

    def frozen_verify(harness_arg, targets, translated, memory, routes):
        assert harness_arg.gate is dummy
        observed["verify"] = True
        return [], 0, 1

    monkeypatch.setattr(production.legacy_kernel, "_BASE_DEEP_REPAIR", frozen_repair)
    monkeypatch.setattr(production.legacy_kernel, "_BASE_DEEP_VERIFY", frozen_verify)
    monkeypatch.setattr(strategy, "annotate", lambda: None)
    monkeypatch.setattr(production, "install_gigachat_runtime_guard", lambda: None)

    def fake_kernel_main():
        production.legacy_kernel._BASE_DEEP_REPAIR(harness, [], {}, None, [{"id": "s1"}])
        production.legacy_kernel._BASE_DEEP_VERIFY(harness, [], {}, None, [{"id": "s1"}])

    monkeypatch.setattr(production.legacy_kernel, "main", fake_kernel_main)

    strategy.run()

    assert observed == {"repair": True, "verify": True}
    assert production.legacy_kernel._BASE_DEEP_REPAIR is frozen_repair
    assert production.legacy_kernel._BASE_DEEP_VERIFY is frozen_verify
    assert harness.gate is original_gate
