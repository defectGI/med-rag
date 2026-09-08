"""OpenAI-compatible Chat Completions adapter over raw HTTP.

This is the lingua franca that lets one client reach almost anything: Ollama,
vLLM, llama.cpp, LM Studio, TGI and most hosted APIs all speak the
`POST {base_url}/chat/completions` shape. Implemented with stdlib `urllib` so
the core has no third-party HTTP dependency.

Ollama's `/v1` endpoint has no per-request way to raise its context window
(`num_ctx`) -- it's a server/Modelfile-only setting, and an over-budget
prompt is silently truncated rather than rejected
(github.com/ollama/ollama/issues/5356). When `num_ctx` is configured AND the
provider is "ollama", this client routes through Ollama's own native
`/api/chat` instead, which DOES accept `options.num_ctx` per request -- the
one case where staying "provider-agnostic" would mean staying unable to
guarantee the model actually saw the whole prompt. Every other provider
(openrouter, hosted openai, anthropic-via-elsewhere, ollama without
`num_ctx` set) is completely unaffected; this is opt-in and additive.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Sequence

from . import ollama_recovery
from .base import LLMError

try:
    from medrag.pipeline.parser import progress
except ImportError:  # llm/ reused standalone -- the in-flight gauge just turns off
    progress = None

# A local Ollama model that starts repeating a single character ("??????...")
# never recovers on its own -- see ollama_recovery.py. Retried a few times
# first (a one-off sampling glitch clears by itself more often than not);
# only once it PERSISTS across those retries is the server assumed wedged
# and worth killing/restarting (and remote servers are never touched, see
# ollama_recovery.restart_local_ollama).
_GARBAGE_RETRY_ATTEMPTS = 3


class OpenAICompatClient:
    """Talks to any OpenAI-compatible `/chat/completions` endpoint.

    `base_url` is the API root without the trailing path, e.g.
    `http://localhost:11434/v1` (Ollama) or `https://api.openai.com/v1`.
    `api_key` is optional — local servers usually need none.

    `provider`/`num_ctx`/`think_on` only matter together: when `provider`
    is exactly "ollama" and `num_ctx` is set, every call is routed through
    Ollama's native `/api/chat` (see module docstring) instead of `/v1`.
    Leave `num_ctx` unset to keep using `/v1` as before.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        extra_body: dict | None = None,
        provider: str | None = None,
        num_ctx: int | None = None,
        think_on: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # Merged into every /v1 request body — e.g. {"reasoning_effort": "none"}
        # to stop a reasoning model from spending the token budget on hidden
        # thinking (which leaves message.content null for short-answer tasks).
        # Not used on the native-Ollama path (see _chat_ollama_native).
        self.extra_body = extra_body or {}
        self.provider = provider
        self.num_ctx = num_ctx
        self.think_on = think_on
        # The `usage` block from the most recent response, normalized to
        # {"prompt_tokens": int, "completion_tokens": int} regardless of which
        # path served the request. Not part of the VLMClient/LLMClient
        # protocol -- an opt-in extra a caller can read via getattr() right
        # after a call, e.g. to notice prompt_tokens landing suspiciously
        # close to a small context window (evidence of silent truncation on
        # the /v1 path, or a sanity check that num_ctx was actually honored
        # on the native path).
        self.last_usage: dict | None = None

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        return self._chat(system=system, user_text=user, images=None, max_tokens=max_tokens)

    def complete_vision(self, *, system: str, user: str,
                        images: Sequence[tuple[str, bytes]],
                        max_tokens: int = 2048) -> str:
        return self._chat(system=system, user_text=user, images=images, max_tokens=max_tokens)

    def _chat(self, *, system: str, user_text: str,
             images: Sequence[tuple[str, bytes]] | None, max_tokens: int) -> str:
        # Single choke point for every request either path serves -- feeds
        # the "model requests in flight" gauge (see parser/progress.py).
        if progress is not None:
            progress.request_started()
        try:
            return self._chat_with_garbage_guard(system=system, user_text=user_text,
                                                 images=images, max_tokens=max_tokens)
        finally:
            if progress is not None:
                progress.request_finished()

    def _dispatch(self, *, system: str, user_text: str,
                  images: Sequence[tuple[str, bytes]] | None, max_tokens: int) -> str:
        if self.provider == "ollama" and self.num_ctx:
            return self._chat_ollama_native(system=system, user_text=user_text,
                                            images=images, max_tokens=max_tokens)
        return self._chat_openai_compat(system=system, user_text=user_text,
                                        images=images, max_tokens=max_tokens)

    def _chat_with_garbage_guard(self, *, system: str, user_text: str,
                                 images: Sequence[tuple[str, bytes]] | None,
                                 max_tokens: int) -> str:
        content = self._dispatch(system=system, user_text=user_text,
                                 images=images, max_tokens=max_tokens)
        if self.provider != "ollama" or not ollama_recovery.is_garbage_output(content):
            return content

        # Degenerate output ("??????...") from Ollama -- retry plainly first,
        # a one-off sampling glitch usually clears on the next call.
        for _ in range(_GARBAGE_RETRY_ATTEMPTS - 1):
            content = self._dispatch(system=system, user_text=user_text,
                                     images=images, max_tokens=max_tokens)
            if not ollama_recovery.is_garbage_output(content):
                return content

        # Persisted across every plain retry -- the server itself is likely
        # wedged. Kill + relaunch it (local only, no-op and returns False for
        # a remote host) and try once more before giving up.
        if ollama_recovery.restart_local_ollama(self.base_url):
            content = self._dispatch(system=system, user_text=user_text,
                                     images=images, max_tokens=max_tokens)
            if not ollama_recovery.is_garbage_output(content):
                return content
            raise LLMError(
                f"ollama at {self.base_url} still returning degenerate output "
                f"(repeated character) after {_GARBAGE_RETRY_ATTEMPTS} retries "
                "and a server restart"
            )

        raise LLMError(
            f"ollama at {self.base_url} returning degenerate output "
            f"(repeated character) after {_GARBAGE_RETRY_ATTEMPTS} retries; "
            "server is remote or could not be restarted"
        )

    # -- default path: OpenAI-compatible /v1, works for any provider --------

    def _chat_openai_compat(self, *, system: str, user_text: str,
                            images: Sequence[tuple[str, bytes]] | None,
                            max_tokens: int) -> str:
        if images:
            # Data-URI `image_url` shape is the de-facto standard understood
            # by OpenAI, vLLM, Ollama, LM Studio and OpenRouter alike.
            content: list[dict] | str = [{"type": "text", "text": user_text}]
            for mime, data in images:
                b64 = base64.b64encode(data).decode("ascii")
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}})
        else:
            content = user_text
        base_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        # Retry once without the thinking/reasoning toggle if the server
        # rejects it outright -- a model with no thinking mode has no such
        # parameter to begin with, so this should behave like "off" rather
        # than hard-failing the whole call.
        try:
            return self._post({**base_payload, **self.extra_body})
        except LLMError as exc:
            if self.extra_body and self._rejected_extra_body(exc):
                return self._post(base_payload)
            raise

    def _rejected_extra_body(self, exc: LLMError) -> bool:
        detail = str(exc).lower()
        return any(key.lower() in detail for key in self.extra_body)

    def _post(self, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            self.last_usage = payload.get("usage")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"cannot reach {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"non-JSON response from {self.base_url}: {exc}") from exc

        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {payload!r:.500}") from exc

    # -- opt-in path: Ollama's native /api/chat, only for per-request num_ctx --

    def _chat_ollama_native(self, *, system: str, user_text: str,
                            images: Sequence[tuple[str, bytes]] | None,
                            max_tokens: int) -> str:
        """Only reached when provider == "ollama" and num_ctx is set (see
        _chat). Ollama's native API is the only way to force `num_ctx` per
        request; the message/image/response shapes all differ from `/v1` so
        this is a separate request/parse path, not a flag on the other one."""
        user_msg: dict = {"role": "user", "content": user_text}
        if images:
            # Native API wants raw base64 in an "images" array on the
            # message -- no data-URI prefix, no content-block wrapping.
            user_msg["images"] = [base64.b64encode(data).decode("ascii") for _, data in images]
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user_msg],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": 0},
        }
        if self.think_on is not None:
            payload["think"] = self.think_on

        root = self.base_url.removesuffix("/v1")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{root}/api/chat", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMError(f"HTTP {exc.code} from {root}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"cannot reach {root}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"non-JSON response from {root}: {exc}") from exc

        # Native API reports counts at the top level under different names --
        # normalized to the same shape _post uses so a caller (e.g.
        # benchmark's prompt_tokens reporting) doesn't need to know which
        # path served the request.
        if "prompt_eval_count" in reply or "eval_count" in reply:
            self.last_usage = {
                "prompt_tokens": reply.get("prompt_eval_count"),
                "completion_tokens": reply.get("eval_count"),
            }

        try:
            return reply["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {reply!r:.500}") from exc
