"""Provider-agnostic LLM access.

Public surface:
    LLMClient      - the text protocol every adapter satisfies
    VLMClient      - the vision protocol (text + attached images)
    LLMError       - raised on any model/transport failure
    get_client     - build a text client from environment configuration
    get_vlm_client - build a vision client ("primary" or "secondary" role)
    describe_configured_models - snapshot of the resolved model config, for
                     recording "which model/context parsed this" into
                     output (IR metadata, pipeline summaries, ...)

Environment (read from the real environment, then from a `.env` file at the repo
root if present — real environment variables always win). Every variable exists
in three prefixes; `VLM_*` falls back to `LLM_*` per variable (a single
configured multimodal model serves both roles), `VLM2_*` never falls back (the
secondary verifier must be an independently chosen model):
    {LLM,VLM,VLM2}_PROVIDER  "openai" (default, any OpenAI-compatible server)
                             | "openrouter" | "ollama" | "local" | "anthropic"
    {LLM,VLM,VLM2}_MODEL     model id (required for openai; defaults to opus
                             for anthropic)
    {LLM,VLM,VLM2}_BASE_URL  API root, required for the openai provider (.../v1)
    {LLM,VLM,VLM2}_API_KEY   optional; local servers need none, Anthropic can
                             use its own credential chain when omitted
    {LLM,VLM,VLM2}_THINKING_ON  "1" to keep hybrid-reasoning models (Qwen3/3.5,
                             gpt-oss, ...) thinking (default off). Off by
                             default because a reasoning model can burn the
                             whole token budget on hidden reasoning and leave
                             the visible answer empty on short-answer tasks.
                             Works the same way for both "openrouter" and
                             "ollama" providers. Models with no thinking mode
                             simply ignore/reject the toggle either way (the
                             client retries once without it if the server
                             errors on the unknown parameter), so setting
                             this has no effect on non-reasoning models.
    VLM_CLASSIFY_*           optional independent role for classify_crop's
                             "what type of visual region is this" calls (see
                             images/visual_classify.py) -- a cheap short-answer
                             question that doesn't need the same model, or
                             even the same server, as the page read / OCR /
                             describe calls. Same PROVIDER/MODEL/BASE_URL/
                             API_KEY fields as above, each falling back to its
                             VLM_* counterpart -- so setting only
                             VLM_CLASSIFY_MODEL keeps the primary VLM's own
                             server and just swaps the model id, while also
                             setting VLM_CLASSIFY_PROVIDER/BASE_URL/API_KEY
                             points classify at an entirely different server
                             (e.g. classify locally on Ollama while the
                             primary VLM/describe runs on a cloud provider).
                             No VLM_CLASSIFY_* set at all -> reuses the
                             primary VLM client outright, no behavior change.
    {LLM,VLM,VLM2}_NUM_CTX   ollama only: context window (tokens) to force via
                             Ollama's native /api/chat instead of /v1 -- the
                             only way to raise num_ctx per request (the /v1
                             endpoint has none; an over-budget prompt is
                             otherwise truncated silently, not rejected). Left
                             unset/0 on an ollama provider, _DEFAULT_OLLAMA_NUM_CTX
                             is applied automatically (and written back to the
                             env var, so it's observable afterward) --
                             enforcement is the default, not opt-in. No effect
                             on any other provider.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .base import LLMClient, LLMError, VLMClient

__all__ = [
    "LLMClient",
    "LLMError",
    "VLMClient",
    "describe_configured_models",
    "get_client",
    "get_vlm_classify_client",
    "get_vlm_client",
    "vlm_classify_configured",
]

# Applied automatically whenever an ollama-provider client is built with no
# explicit {prefix}_NUM_CTX -- context enforcement should be the default
# behavior for every caller (pipeline, benchmark, table/image OCR calls),
# not something each one has to remember to opt into. 16384 is a
# middle-ground guess: enough for a good number of rendered page images plus
# a full document's Markdown, without assuming more VRAM headroom than a
# 20-40GB-class vision model already leaves free. Override per-role via
# {prefix}_NUM_CTX in parser/.env (or, for benchmark, config.json/"num_ctx").
_DEFAULT_OLLAMA_NUM_CTX = 16384


def get_client() -> LLMClient:
    """Construct a text `LLMClient` from `LLM_*` environment variables."""
    return _build_client("LLM")


def get_vlm_client(role: str = "primary", model: str | None = None) -> VLMClient:
    """Construct a vision `VLMClient` from environment variables.

    role="primary"    reads `VLM_*`, falling back to `LLM_*` per variable.
    role="secondary"  reads `VLM2_*` only; raises `LLMError` when unset. The
                      secondary model independently re-reads suspicious regions
                      (consensus check), so silently reusing the primary would
                      defeat its purpose.
    role="classify"   reads `VLM_CLASSIFY_*`, falling back to `VLM_*` per
                      variable -- so setting only VLM_CLASSIFY_MODEL keeps
                      everything else (provider/base_url/api_key) on the
                      primary VLM's own server, but setting
                      VLM_CLASSIFY_PROVIDER/BASE_URL/API_KEY too points
                      classify_crop at a completely different server (e.g.
                      classify locally on Ollama while the primary VLM/
                      describe runs on a cloud provider). Prefer
                      `get_vlm_classify_client()` over calling this directly:
                      it also handles "no VLM_CLASSIFY_* set at all" by
                      reusing the primary client outright, at zero cost.
    model             overrides the env-configured model id, same
                      provider/base_url/api_key otherwise -- for a caller
                      whose task differs enough from the role's usual job
                      (e.g. tables/structure/vlm_adapter.py's grid/bbox
                      extraction vs. this role's normal OCR transcription)
                      that a different model on the same server may fit
                      better. Leave unset to use the role's own model as
                      configured.
    """
    if role == "primary":
        return _build_client("VLM", fallback="LLM", model_override=model)
    if role == "secondary":
        load_dotenv()
        if not (os.getenv("VLM2_PROVIDER") or os.getenv("VLM2_MODEL")):
            raise LLMError("secondary VLM not configured (set VLM2_* variables)")
        return _build_client("VLM2", model_override=model)
    if role == "classify":
        return _build_client("VLM_CLASSIFY", fallback="VLM", model_override=model)
    raise ValueError(f"unknown VLM role: {role!r}")


def get_vlm_classify_client(vlm: VLMClient | None) -> VLMClient | None:
    """The client `classify_crop` calls should actually use, given an already-
    resolved primary VLM client `vlm` (or None if unconfigured/unavailable).

    No `VLM_CLASSIFY_*` variable set at all: returns `vlm` itself unchanged --
    classification just rides the primary VLM, exactly like before this role
    existed, with no second client ever built. Set `VLM_CLASSIFY_MODEL` alone
    to keep the same provider/base_url/api_key and only swap the model id
    (e.g. a smaller/faster model for classify_crop's cheap "what type is
    this" question, while the primary VLM's own model stays reserved for the
    page read, OCR and describe_crop calls that actually need it). Also set
    `VLM_CLASSIFY_PROVIDER`/`VLM_CLASSIFY_BASE_URL`/`VLM_CLASSIFY_API_KEY` to
    point classify at an ENTIRELY different server -- e.g. classify locally
    on Ollama while the primary VLM (describe) runs on a cloud provider.

    `vlm is None` (primary VLM unconfigured/down) still tries a fully
    independent `VLM_CLASSIFY_*` config if one is set (classify doesn't have
    to depend on the primary VLM being reachable at all), else returns None.
    """
    if not vlm_classify_configured():
        return vlm
    try:
        return get_vlm_client("classify")
    except LLMError:
        return vlm


def vlm_classify_configured() -> bool:
    """True iff VLM_CLASSIFY_* actually diverges from VLM (any of PROVIDER/
    MODEL/BASE_URL/API_KEY explicitly set) -- same check
    get_vlm_classify_client uses to decide whether to build a second client
    at all. A caller that wants to pre-warm a genuinely different classify
    model (run_parse_pipeline.py's phase 0) uses this to skip that pass
    entirely when there's nothing distinct to warm -- classify would just
    reuse the primary VLM, and phase 1 already classifies as part of its own
    work in that case."""
    load_dotenv()
    return any(os.getenv(f"VLM_CLASSIFY_{name}")
              for name in ("PROVIDER", "MODEL", "BASE_URL", "API_KEY"))


def describe_configured_models() -> dict[str, dict]:
    """A snapshot of the model config currently resolved into the environment
    -- for recording "which model, with how much context, actually produced
    this output" into a document's own metadata or a run's summary, since
    that's otherwise lost once the run finishes and the env vars go away.

    Call this AFTER get_client()/get_vlm_client() for the roles you care
    about, so a role's num_ctx auto-default (see _build_client) has already
    been resolved and written back to the env var this reads. Returns one
    entry per role that has a provider or model configured at all (VLM2 and
    VLM_CLASSIFY are included only if actually set -- most runs don't
    override them); each entry is {"provider", "model", "num_ctx", "thinking"}.
    """
    def _role(prefix: str, fallback: str | None = None) -> dict | None:
        provider = _env(prefix, "PROVIDER", fallback)
        model = _env(prefix, "MODEL", fallback)
        if not provider and not model:
            return None
        num_ctx_raw = _env(prefix, "NUM_CTX", fallback)
        thinking_raw = _env(prefix, "THINKING_ON", fallback)
        return {
            "provider": provider,
            "model": model,
            "num_ctx": int(num_ctx_raw) if num_ctx_raw else None,
            "thinking": (thinking_raw or "").strip().lower() in ("1", "true", "yes", "on"),
        }

    out: dict[str, dict] = {}
    llm = _role("LLM")
    if llm:
        out["LLM"] = llm
    vlm = _role("VLM", fallback="LLM")
    if vlm:
        out["VLM"] = vlm
    vlm2 = _role("VLM2")
    if vlm2:
        out["VLM2"] = vlm2
    # Only recorded when it actually diverges from VLM (see
    # get_vlm_classify_client) -- otherwise it's the same model/config as
    # "VLM" above and a second identical entry would just be noise.
    if vlm_classify_configured():
        vlm_classify = _role("VLM_CLASSIFY", fallback="VLM")
        if vlm_classify:
            out["VLM_CLASSIFY"] = vlm_classify
    return out


def _env(prefix: str, name: str, fallback: str | None) -> str | None:
    """`{prefix}_{name}`, else `{fallback}_{name}`, else None."""
    val = os.getenv(f"{prefix}_{name}")
    if val is None and fallback:
        val = os.getenv(f"{fallback}_{name}")
    return val or None


def _build_client(prefix: str, fallback: str | None = None,
                  model_override: str | None = None):
    """Build an adapter from `{prefix}_*` environment variables."""
    load_dotenv()  # no-op if there's no .env file; never overrides a set env var
    provider = (_env(prefix, "PROVIDER", fallback) or "openai").strip().lower()
    model = model_override or _env(prefix, "MODEL", fallback)
    api_key = _env(prefix, "API_KEY", fallback)

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=model, api_key=api_key)

    # Known OpenAI-compatible hosts get a default base URL so *_BASE_URL is optional.
    # "ollama" is its own recognized name (not just an alias for "local") so
    # scripts/config that key off the literal provider string -- e.g.
    # ensure_ollama_models.py's is_ollama_target -- agree with what actually
    # builds the client.
    _COMPAT_DEFAULT_URL = {
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama": "http://localhost:11434/v1",
    }
    if provider in ("openai", "openai-compat", "openai_compatible", "local",
                    "openrouter", "ollama"):
        base_url = _env(prefix, "BASE_URL", fallback) or _COMPAT_DEFAULT_URL.get(provider)
        if not base_url:
            raise LLMError(f"{prefix}_BASE_URL is required for the openai provider")
        if not model:
            raise LLMError(f"{prefix}_MODEL is required for the openai provider")
        from .openai_compat import OpenAICompatClient

        # Hybrid-reasoning models hide their answer behind hidden "thinking"
        # tokens and can burn the whole max_tokens budget on it, leaving the
        # visible content empty on short-answer tasks (both OpenRouter and
        # Ollama default this ON). Off by default; set {prefix}_THINKING_ON=1
        # to keep it. Each provider exposes the toggle under its own field
        # name, so the two need separate handling here.
        thinking_on = (_env(prefix, "THINKING_ON", fallback) or "").strip().lower() in (
            "1", "true", "yes", "on")
        extra_body: dict = {}
        if provider == "openrouter":
            extra_body["reasoning"] = {"enabled": thinking_on}
        elif provider == "ollama" and not thinking_on:
            extra_body["reasoning_effort"] = "none"

        num_ctx_raw = _env(prefix, "NUM_CTX", fallback)
        num_ctx = None
        if num_ctx_raw:
            try:
                num_ctx = int(num_ctx_raw)
            except ValueError:
                raise LLMError(f"{prefix}_NUM_CTX must be an integer, got {num_ctx_raw!r}")
        elif provider == "ollama":
            # No explicit choice anywhere (env/.env/config) -- fall back to a
            # safe default rather than silently running unenforced. Written
            # back to the env var so it's observable afterward (e.g. a
            # caller's own startup log) exactly like an explicit value would be.
            num_ctx = _DEFAULT_OLLAMA_NUM_CTX
            os.environ[f"{prefix}_NUM_CTX"] = str(num_ctx)

        return OpenAICompatClient(base_url=base_url, model=model, api_key=api_key,
                                  extra_body=extra_body, provider=provider,
                                  num_ctx=num_ctx, think_on=thinking_on)

    raise LLMError(f"unknown {prefix}_PROVIDER: {provider!r}")
