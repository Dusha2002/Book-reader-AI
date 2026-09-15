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
