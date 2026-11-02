"""Tests for GenerationConfig + how LLMClient applies it.

These tests exercise the config dataclass and the kwargs construction path that
each backend would use, without actually loading a model. Backend integration is
covered separately by scripts/smoke_test_llm.py.
"""

from __future__ import annotations

import sys
import types
from dataclasses import FrozenInstanceError, replace

import pytest

from src.reasoning.llm_client import (
    DETERMINISTIC,
    GenerationConfig,
    LLMClient,
    LLMResponse,
)


# ────────────────────────────────────────────────────────────────────────────
# GenerationConfig dataclass
# ────────────────────────────────────────────────────────────────────────────

def test_default_config_is_greedy_seeded():
    assert DETERMINISTIC.temperature == 0.0
    assert DETERMINISTIC.top_p == 1.0
    assert DETERMINISTIC.top_k == -1
    assert DETERMINISTIC.seed == 42
    assert DETERMINISTIC.max_tokens == 1024
    assert DETERMINISTIC.stop is None


def test_config_is_frozen():
    cfg = GenerationConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.temperature = 0.7  # type: ignore[misc]


def test_replace_overrides_one_field_only():
    cfg = replace(DETERMINISTIC, max_tokens=512)
    assert cfg.max_tokens == 512
    # Everything else unchanged.
    assert cfg.temperature == DETERMINISTIC.temperature
    assert cfg.seed == DETERMINISTIC.seed


def test_two_default_instances_are_equal():
    assert GenerationConfig() == GenerationConfig()


def test_config_with_stop_tokens_carries_them():
    cfg = GenerationConfig(stop=["</answer>", "###"])
    assert cfg.stop == ["</answer>", "###"]


# ────────────────────────────────────────────────────────────────────────────
# LLMResponse carries config
# ────────────────────────────────────────────────────────────────────────────

def test_llm_response_default_config_is_deterministic():
    r = LLMResponse(content="x", model="m", provider="vllm")
    # The factory default points at DETERMINISTIC.
    assert r.config == DETERMINISTIC


def test_llm_response_records_custom_config():
    cfg = replace(DETERMINISTIC, max_tokens=128, seed=7)
    r = LLMResponse(content="x", model="m", provider="vllm", config=cfg)
    assert r.config.max_tokens == 128
    assert r.config.seed == 7


# ────────────────────────────────────────────────────────────────────────────
# vLLM kwargs construction (no model load)
# ────────────────────────────────────────────────────────────────────────────

class _FakeSamplingParams:
    """Stand-in for vllm.SamplingParams that records kwargs."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_vllm(monkeypatch):
    """Inject a fake `vllm` module with just SamplingParams + a no-op LLM."""
    fake = types.ModuleType("vllm")
    fake.SamplingParams = _FakeSamplingParams
    fake.LLM = lambda **kw: None  # never called in these tests
    monkeypatch.setitem(sys.modules, "vllm", fake)
    # Also reset the cached reference in llm_client.
    import src.reasoning.llm_client as mod
    monkeypatch.setattr(mod, "_vllm_module", fake)


def _make_client_skipping_init(provider="vllm", config=None):
    """
    Construct an LLMClient instance without going through __init__ (which would
    try to load a model). Sufficient for testing _sampling_params / kwargs.
    """
    cfg = config or DETERMINISTIC
    client = LLMClient.__new__(LLMClient)
    client.model = "fake/model"
    client.dtype = "auto"
    client.quantization = None
    client.gpu_memory_utilization = 0.9
    client.max_model_len = None
    client.default_config = cfg
    client.provider = provider
    client._engine = None
    client._tokenizer = None
    return client


def test_vllm_sampling_params_default(monkeypatch):
    _install_fake_vllm(monkeypatch)
    client = _make_client_skipping_init(provider="vllm")
    params = client._sampling_params(DETERMINISTIC)
    assert params.kwargs["max_tokens"] == 1024
    assert params.kwargs["temperature"] == 0.0
    assert params.kwargs["top_p"] == 1.0
    assert params.kwargs["top_k"] == -1
    assert params.kwargs["seed"] == 42
    assert "stop" not in params.kwargs  # None → omitted


def test_vllm_sampling_params_custom(monkeypatch):
    _install_fake_vllm(monkeypatch)
    client = _make_client_skipping_init(provider="vllm")
    cfg = GenerationConfig(
        max_tokens=256, temperature=0.7, top_p=0.95, top_k=40, seed=7,
        stop=["</s>"],
    )
    params = client._sampling_params(cfg)
    assert params.kwargs["max_tokens"] == 256
    assert params.kwargs["temperature"] == 0.7
    assert params.kwargs["top_p"] == 0.95
    assert params.kwargs["top_k"] == 40
    assert params.kwargs["seed"] == 7
    assert params.kwargs["stop"] == ["</s>"]


# ────────────────────────────────────────────────────────────────────────────
# transformers gen_kwargs construction (mirror the if-temp>0 logic in code)
# ────────────────────────────────────────────────────────────────────────────

def _build_gen_kwargs_like_client(cfg: GenerationConfig) -> dict:
    """
    Reproduce the gen_kwargs logic from `LLMClient._complete_transformers`
    so we can assert the contract without running a model.
    """
    gen = {
        "max_new_tokens": cfg.max_tokens,
        "do_sample": cfg.temperature > 0,
        "pad_token_id": 0,
    }
    if cfg.temperature > 0:
        gen["temperature"] = cfg.temperature
        if cfg.top_p < 1.0:
            gen["top_p"] = cfg.top_p
        if cfg.top_k > 0:
            gen["top_k"] = cfg.top_k
    return gen


def test_transformers_kwargs_greedy():
    gen = _build_gen_kwargs_like_client(DETERMINISTIC)
    assert gen["do_sample"] is False
    # Greedy mode must NOT carry temperature/top_p/top_k — they trip the
    # transformers validator and produce the "may be ignored" warning.
    assert "temperature" not in gen
    assert "top_p" not in gen
    assert "top_k" not in gen


def test_transformers_kwargs_sampling():
    cfg = GenerationConfig(temperature=0.7, top_p=0.95, top_k=40)
    gen = _build_gen_kwargs_like_client(cfg)
    assert gen["do_sample"] is True
    assert gen["temperature"] == 0.7
    assert gen["top_p"] == 0.95
    assert gen["top_k"] == 40


def test_transformers_kwargs_sampling_top_p_at_one_omitted():
    # top_p=1.0 is the no-op default; emitting it is harmless but cluttered.
    cfg = GenerationConfig(temperature=0.5, top_p=1.0, top_k=-1)
    gen = _build_gen_kwargs_like_client(cfg)
    assert gen["do_sample"] is True
    assert gen["temperature"] == 0.5
    assert "top_p" not in gen
    assert "top_k" not in gen


# ────────────────────────────────────────────────────────────────────────────
# Backwards-compat: .seed property still works
# ────────────────────────────────────────────────────────────────────────────

def test_client_seed_property_reads_from_default_config():
    client = _make_client_skipping_init(config=replace(DETERMINISTIC, seed=99))
    assert client.seed == 99
