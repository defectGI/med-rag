"""JSON extraction and retry helpers. Used by both stages."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, TypeVar

from ..config import RetrySettings
from ..errors import ProviderError, StructuredOutputError
from ..logging_utils import get_logger

logger = get_logger("pipeline.json")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

T = TypeVar("T")


def extract_json(text: str, *, strict: bool) -> dict[str, Any]:
    """Parse a JSON object from raw model text.

    Args:
        text: The raw response body.
        strict: If ``True``, only try a direct ``json.loads``. If ``False``,
            also strip markdown fences and grab the first ``{...}`` block.

    Raises:
        StructuredOutputError: If no JSON object can be parsed.
    """
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise StructuredOutputError("Expected a JSON object, got something else.")
    except json.JSONDecodeError:
        if strict:
            raise StructuredOutputError(
                "Response was not valid JSON (strict_json is on)."
            )

    # Lenient path: strip fences, then take the first {...} block.
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise StructuredOutputError("Could not find a JSON object in the response.")


def with_retry(
    func: Callable[[], T],
    *,
    retry: RetrySettings,
    description: str,
) -> T:
    """Run ``func`` with retry and backoff.

    Retries on :class:`StructuredOutputError` and transient
    :class:`ProviderError`. Re-raises the last error when attempts run out.

    Args:
        func: Zero-arg callable. One attempt.
        retry: Retry config.
        description: Label used in log messages.
    """
    attempts = max(1, retry.max_attempts)
    delay = retry.backoff_seconds
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except (StructuredOutputError, ProviderError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.warning(
                "%s failed on attempt %d/%d (%s). Retrying in %.1fs.",
                description,
                attempt,
                attempts,
                type(exc).__name__,
                delay,
            )
            if delay > 0:
                time.sleep(delay)
            delay *= retry.backoff_factor

    assert last_exc is not None
    raise last_exc
