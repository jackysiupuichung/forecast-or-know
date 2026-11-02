#!/usr/bin/env python3
"""
Local open-source LLM client.

All inference is local — no external APIs, no tool use, no web access.
This is required by the experimental design: every model in the lineup must
reason from parametric knowledge plus the provided graph evidence only.

Backends (selected automatically, in priority order):
    1. vllm             — preferred for 8B+ models; high throughput, batched generation
    2. transformers     — fallback; works without a GPU server, slower
    3. llama_cpp_server — HTTP client for an external `llama-server` (llama.cpp).
                          Selected explicitly via provider='llama_cpp_server'; not
                          chosen by 'auto'. Useful for CPU-only smoke tests and
                          GGUF-quantised models. Caller is responsible for starting
                          and stopping the server process.

Generation is governed by a single `GenerationConfig` dataclass. All experiments
use `DETERMINISTIC` (temperature=0, fixed seed). Override only with explicit
intent — the resolved config is recorded on every `LLMResponse` so saved
predictions carry their own sampling provenance.

Public API:
    GenerationConfig(max_tokens, temperature, top_p, top_k, seed, stop)
    DETERMINISTIC = GenerationConfig(temperature=0.0, ...)        # frozen default

    LLMClient(provider='vllm', model='meta-llama/Llama-3.1-70B-Instruct')
        .complete(system, user, config=None) → LLMResponse
        .complete_batch(prompts, config=None) → List[LLMResponse]

    LLMResponse:
        .content    — str, the assistant's text response
        .model      — str
        .provider   — str ('vllm' | 'transformers')
        .usage      — Dict[str, int] with input_tokens, output_tokens
        .config     — GenerationConfig actually applied to this call
        .raw        — Raw backend response object (for debugging)

Supported model lineup:
    meta-llama/Llama-2-13b-chat-hf
    meta-llama/Llama-2-70b-chat-hf
    meta-llama/Llama-3.1-8B-Instruct
    meta-llama/Llama-3.1-70B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-72B-Instruct
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence


# Backends are imported lazily so a missing dependency doesn't break the module.
_vllm_module = None
_transformers_module = None


# ────────────────────────────────────────────────────────────────────────────────
# Generation config
# ────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationConfig:
    """
    Sampling/decoding parameters for one LLM call.

    The default (`DETERMINISTIC`) is greedy: temperature=0, fixed seed=42, no
    nucleus or top-k truncation. Use `replace(DETERMINISTIC, ...)` to override
    a single field while keeping the others fixed.

    Notes on determinism:
        - Greedy decoding (temperature=0) gives bitwise-stable outputs
          *within* a backend + hardware combination.
        - vLLM batched scheduling can introduce slight nondeterminism even
          at temp=0 because attention kernels run different batch sizes.
          The `seed` field still controls any tie-breaking RNG.
        - At temperature>0 the seed is the only knob that pins outputs.
    """
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1  # vLLM convention: -1 disables top-k
    seed: int = 42
    stop: Optional[Sequence[str]] = None
    request_logprobs: bool = False
    top_logprobs: int = 10


DETERMINISTIC: GenerationConfig = GenerationConfig()
"""Project-wide default. Greedy, seeded, no nucleus/top-k filtering."""


# ────────────────────────────────────────────────────────────────────────────────
# Response
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    config: GenerationConfig = field(default_factory=lambda: DETERMINISTIC)
    raw: Any = None
    logprobs: Optional[Dict[str, Any]] = None


class LLMClient:
    """
    Unified wrapper around local open-source LLM backends.

    Args:
        provider:     'vllm' | 'transformers' | 'llama_cpp_server' | 'auto'
                      (default: 'auto'; 'auto' never picks llama_cpp_server)
        model:        HuggingFace model ID, e.g. 'meta-llama/Llama-3.1-70B-Instruct'.
                      For provider='llama_cpp_server' this is a label only — the
                      actual weights live in the llama-server process.
        dtype:        'auto' | 'float16' | 'bfloat16' (default: 'auto')
        quantization: None | 'awq' | 'gptq' | 'bitsandbytes' (default: None)
                      Use 'awq' or 'gptq' for 70B+ models on a single 40GB GPU.
        gpu_memory_utilization: vLLM only; fraction of GPU memory to use (default: 0.90)
        max_model_len: Optional context length cap (default: model default)
        default_config: GenerationConfig used when callers pass `config=None`
                        (default: DETERMINISTIC).
        server_url:   Base URL of an external llama.cpp server, e.g.
                      'http://127.0.0.1:8080'. Required for provider='llama_cpp_server'.
        request_timeout: HTTP timeout in seconds for llama_cpp_server (default 600).
    """

    def __init__(
        self,
        provider: str = "auto",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: str = "auto",
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        default_config: GenerationConfig = DETERMINISTIC,
        server_url: Optional[str] = None,
        request_timeout: float = 600.0,
        adapter: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.default_config = default_config
        self.server_url = server_url.rstrip("/") if server_url else None
        self.request_timeout = request_timeout
        # Optional LoRA adapter dir (PEFT) — transformers provider only. The
        # finetuned treatment arm loads base + adapter here so the eval is
        # byte-identical to the zero-shot path except for the adapter weights.
        self.adapter = adapter

        self.provider = self._resolve_provider(provider)

        if self.provider == "vllm":
            self._engine, self._tokenizer = self._build_vllm()
        elif self.provider == "transformers":
            self._engine, self._tokenizer = self._build_transformers()
        elif self.provider == "llama_cpp_server":
            self._engine, self._tokenizer = self._build_llama_cpp_server()
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Use 'vllm', 'transformers', "
                f"'llama_cpp_server', or 'auto'."
            )

    @property
    def seed(self) -> int:
        """Backwards-compat alias used by `vllm.LLM(seed=...)` engine init."""
        return self.default_config.seed

    def _resolve_provider(self, provider: str) -> str:
        if provider != "auto":
            return provider.lower()
        try:
            import vllm  # noqa: F401
            return "vllm"
        except ImportError:
            return "transformers"

    # ──────────────────────────────────────────────────────────────────────────
    # Backend setup
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vllm(self):
        global _vllm_module
        if _vllm_module is None:
            import vllm as _vllm_module  # noqa: F811

        kwargs = {
            "model": self.model,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": self.default_config.seed,
            "trust_remote_code": True,
        }
        if self.quantization:
            kwargs["quantization"] = self.quantization
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len

        engine = _vllm_module.LLM(**kwargs)
        tokenizer = engine.get_tokenizer()
        return engine, tokenizer

    def _build_llama_cpp_server(self):
        if not self.server_url:
            raise ValueError(
                "provider='llama_cpp_server' requires server_url, e.g. "
                "'http://127.0.0.1:8080'."
            )
        import urllib.request
        import urllib.error

        # Probe the health endpoint so misconfiguration surfaces at construction
        # rather than on the first generate() call.
        try:
            with urllib.request.urlopen(
                f"{self.server_url}/health", timeout=5.0
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"llama-server health check returned HTTP {resp.status}"
                    )
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach llama-server at {self.server_url}: {e}. "
                f"Start it with `llama-server -m <model.gguf> --port 8080`."
            ) from e

        # No engine, no tokenizer — the server handles both. The dispatcher
        # below sends raw {role, content} messages and lets the server apply
        # the model's chat template.
        return None, None

    def _build_transformers(self):
        global _transformers_module
        if _transformers_module is None:
            import transformers as _transformers_module  # noqa: F811
        import torch

        tokenizer = _transformers_module.AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if self.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = "auto"

        if self.quantization == "bitsandbytes":
            # Newer transformers rejects a bare load_in_4bit kwarg; it must go
            # through BitsAndBytesConfig. NF4 + bf16 compute matches the trainer.
            model_kwargs["quantization_config"] = _transformers_module.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = _transformers_module.AutoModelForCausalLM.from_pretrained(
            self.model, **model_kwargs
        )
        # Attach a LoRA adapter (finetuned treatment arm) if requested.
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
            print(f"[transformers] loaded LoRA adapter: {self.adapter}", flush=True)
        model.eval()
        return model, tokenizer

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user: str,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Send a system+user prompt and return the assistant response.

        Tool use is structurally impossible: this is a local causal-LM call
        with no function-calling surface. The applied `GenerationConfig` is
        recorded on the returned `LLMResponse.config`.
        """
        cfg = config or self.default_config
        if self.provider == "llama_cpp_server":
            return self._complete_llama_cpp_server(system, user, cfg)
        prompt = self._format_prompt(system, user)
        if self.provider == "vllm":
            return self._complete_vllm(prompt, cfg)
        return self._complete_transformers(prompt, cfg)

    def complete_batch(
        self,
        prompts: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> List[LLMResponse]:
        """
        Batched generation. Each prompt is a dict {'system': str, 'user': str}.
        Significantly faster than serial complete() calls under vLLM.
        """
        cfg = config or self.default_config
        if self.provider == "llama_cpp_server":
            # llama-server has no batched chat-completions endpoint; serialise.
            return [
                self._complete_llama_cpp_server(p["system"], p["user"], cfg)
                for p in prompts
            ]
        formatted = [self._format_prompt(p["system"], p["user"]) for p in prompts]
        if self.provider == "vllm":
            return self._complete_batch_vllm(formatted, cfg)
        return self._complete_batch_transformers(formatted, cfg)

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt formatting (uses each model's chat template)
    # ──────────────────────────────────────────────────────────────────────────

    def _format_prompt(self, system: str, user: str) -> str:
        # Qwen3 is a reasoning model: it emits <think>...</think> before the
        # answer, pushing the Yes/No past the probe window. Disable thinking via
        # chat_template_kwargs (matches the zero-shot arm's llama.cpp path, which
        # sets enable_thinking=False) so the binary probe sees a bare Yes/No.
        tmpl_kwargs = {}
        if "qwen3" in str(self.model).lower():
            tmpl_kwargs["enable_thinking"] = False
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
        except Exception:
            # Some chat templates (e.g. Gemma-2) reject a separate `system` role
            # ("System role not supported"). Fall back to prepending the system
            # text to the user turn — same content, single user message.
            merged = f"{system}\n\n{user}" if system else user
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False, add_generation_prompt=True, **tmpl_kwargs,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # vLLM implementation
    # ──────────────────────────────────────────────────────────────────────────

    def _sampling_params(self, cfg: GenerationConfig):
        global _vllm_module
        kwargs: Dict[str, Any] = {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "seed": cfg.seed,
        }
        if cfg.stop:
            kwargs["stop"] = list(cfg.stop)
        return _vllm_module.SamplingParams(**kwargs)

    def _complete_vllm(self, prompt: str, cfg: GenerationConfig) -> LLMResponse:
        params = self._sampling_params(cfg)
        outputs = self._engine.generate([prompt], params, use_tqdm=False)
        out = outputs[0]
        return LLMResponse(
            content=out.outputs[0].text,
            model=self.model,
            provider="vllm",
            usage={
                "input_tokens": len(out.prompt_token_ids),
                "output_tokens": len(out.outputs[0].token_ids),
            },
            config=cfg,
            raw=out,
        )

    def _complete_batch_vllm(
        self, prompts: List[str], cfg: GenerationConfig
    ) -> List[LLMResponse]:
        params = self._sampling_params(cfg)
        outputs = self._engine.generate(prompts, params, use_tqdm=False)
        return [
            LLMResponse(
                content=o.outputs[0].text,
                model=self.model,
                provider="vllm",
                usage={
                    "input_tokens": len(o.prompt_token_ids),
                    "output_tokens": len(o.outputs[0].token_ids),
                },
                config=cfg,
                raw=o,
            )
            for o in outputs
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # transformers implementation (fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _scores_to_logprobs_dict(self, gen_ids, scores, cfg) -> Dict[str, Any]:
        """Convert HF `generate` per-step scores into the SAME OpenAI/llama.cpp
        `logprobs` dict shape that `parse_pointwise_binary` consumes:

            {"content": [{"token": str, "logprob": float,
                          "top_logprobs": [{"token": str, "logprob": float}, ...]}, ...]}

        `scores` is a tuple of length n_generated, each [vocab] logits for one
        step (batch already indexed out). Building this shape means the binary
        probe is byte-identical to the llama.cpp path — no reimplementation."""
        import torch

        k = max(1, int(cfg.top_logprobs))
        content = []
        for step, logits in enumerate(scores):
            logp = torch.log_softmax(logits.float(), dim=-1)
            sampled_id = int(gen_ids[step])
            top = torch.topk(logp, k)
            top_lps = [
                {"token": self._tokenizer.decode([int(tid)]),
                 "logprob": float(lp)}
                for lp, tid in zip(top.values.tolist(), top.indices.tolist())
            ]
            content.append({
                "token": self._tokenizer.decode([sampled_id]),
                "logprob": float(logp[sampled_id]),
                "top_logprobs": top_lps,
            })
            # The probe only ever needs the answer position (early tokens); cap
            # to keep memory bounded on long generations.
            if step + 1 >= min(len(scores), cfg.max_tokens):
                break
        return {"content": content}

    def _complete_batch_transformers(
        self, prompts: List[str], cfg: GenerationConfig
    ) -> List[LLMResponse]:
        """Batched transformers generation with per-sequence logprob extraction.

        LEFT-pads the batch (decoder-only: left padding keeps the real tokens
        right-aligned so `generate` continues from the true last token). Each
        sequence's scores are indexed out and passed through the SAME
        `_scores_to_logprobs_dict`, so batched and serial results are identical.
        This is the throughput fix — one forward per batch, not per prompt."""
        import torch

        torch.manual_seed(cfg.seed)
        tok = self._tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        try:
            enc = tok(prompts, return_tensors="pt", padding=True).to(self._engine.device)
        finally:
            tok.padding_side = prev_side
        n_input = enc["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "do_sample": cfg.temperature > 0,
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if cfg.temperature > 0:
            gen_kwargs["temperature"] = cfg.temperature
            if cfg.top_p < 1.0:
                gen_kwargs["top_p"] = cfg.top_p
            if cfg.top_k > 0:
                gen_kwargs["top_k"] = cfg.top_k
        if cfg.request_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            out = self._engine.generate(**enc, **gen_kwargs)

        sequences = out.sequences if cfg.request_logprobs else out
        results: List[LLMResponse] = []
        for b in range(len(prompts)):
            gen_ids = sequences[b, n_input:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            logprobs = None
            if cfg.request_logprobs:
                # out.scores[step] is [batch, vocab]; index this sequence out.
                logprobs = self._scores_to_logprobs_dict(
                    gen_ids, tuple(s[b] for s in out.scores), cfg
                )
            results.append(LLMResponse(
                content=text,
                model=self.model,
                provider="transformers",
                usage={"input_tokens": int(n_input),
                       "output_tokens": int(gen_ids.shape[0])},
                config=cfg,
                logprobs=logprobs,
            ))
        return results

    def _complete_transformers(self, prompt: str, cfg: GenerationConfig) -> LLMResponse:
        import torch

        torch.manual_seed(cfg.seed)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._engine.device)
        n_input = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "do_sample": cfg.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if cfg.temperature > 0:
            gen_kwargs["temperature"] = cfg.temperature
            if cfg.top_p < 1.0:
                gen_kwargs["top_p"] = cfg.top_p
            if cfg.top_k > 0:
                gen_kwargs["top_k"] = cfg.top_k
        if cfg.request_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            out = self._engine.generate(**inputs, **gen_kwargs)

        if cfg.request_logprobs:
            output_ids = out.sequences
            gen_ids = output_ids[0, n_input:]
            logprobs = self._scores_to_logprobs_dict(
                gen_ids, tuple(s[0] for s in out.scores), cfg
            )
        else:
            output_ids = out
            gen_ids = output_ids[0, n_input:]
            logprobs = None

        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)

        return LLMResponse(
            content=text,
            model=self.model,
            provider="transformers",
            usage={
                "input_tokens": int(n_input),
                "output_tokens": int(gen_ids.shape[0]),
            },
            config=cfg,
            raw=output_ids,
            logprobs=logprobs,
        )


    # ──────────────────────────────────────────────────────────────────────────
    # llama.cpp server implementation (HTTP, OpenAI-compatible)
    # ──────────────────────────────────────────────────────────────────────────

    def _complete_llama_cpp_server(
        self, system: str, user: str, cfg: GenerationConfig
    ) -> LLMResponse:
        import json as _json
        import urllib.request
        import urllib.error

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "seed": cfg.seed,
            "stream": False,
        }
        # llama-server uses -1 to disable top_k, matching vLLM. Only forward a
        # positive value.
        if cfg.top_k > 0:
            payload["top_k"] = cfg.top_k
        if cfg.stop:
            payload["stop"] = list(cfg.stop)
        if cfg.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = cfg.top_logprobs
        # Qwen3 is a hybrid reasoning model: by default its first generated
        # token is `<think>`, which breaks the single-token Yes/No logprob
        # probe (max_tokens=1 → only `<think>` is emitted, coverage=0). Disable
        # the thinking block via llama-server's chat_template_kwargs so it emits
        # a bare Yes/No. Gated on the model name so other models are untouched.
        if "qwen3" in str(self.model).lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Optional assistant PREFILL (response-prefix): append a partial assistant
        # turn so the server continues it instead of opening a fresh turn. Forces
        # format-locked models (Llama-3.1-70B emits tool-call `{"name":...}` JSON
        # rather than a bare Yes/No) to commit to the answer token — the FIRST
        # generated token then IS the Yes/No the probe reads. Env-gated so all
        # other models/runs are byte-identical. Requires llama-server's
        # continue-final-message behaviour (set via chat_template_kwargs).
        import os as _os
        _prefill = _os.environ.get("PROBE_PREFILL")
        if _prefill:
            payload["messages"].append({"role": "assistant", "content": _prefill})
            ck = payload.get("chat_template_kwargs") or {}
            ck["continue_final_message"] = True
            ck["add_generation_prompt"] = False
            payload["chat_template_kwargs"] = ck

        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server returned HTTP {e.code}: {detail}"
            ) from e

        data = _json.loads(body)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            content=content,
            model=self.model,
            provider="llama_cpp_server",
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
            config=cfg,
            raw=data,
            logprobs=choice.get("logprobs"),
        )

    def health_check(self, timeout: float = 2.0) -> bool:
        """Ping the llama.cpp server's /health endpoint. True iff 200 OK.

        Falls back to /v1/models on 404 to support older llama-server builds.
        """
        if self.provider != "llama_cpp_server" or not self.server_url:
            return True
        import urllib.request
        import urllib.error

        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(
                    f"{self.server_url}{path}", timeout=timeout
                ) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                return False
            except Exception:
                return False
        return False


# ────────────────────────────────────────────────────────────────────────────────
# Public re-exports
# ────────────────────────────────────────────────────────────────────────────────

__all__ = [
    "LLMClient",
    "LLMResponse",
    "GenerationConfig",
    "DETERMINISTIC",
    "replace",  # convenient: from src.reasoning.llm_client import replace
]
