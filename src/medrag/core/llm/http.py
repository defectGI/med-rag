"""Shared OpenAI-compatible HTTP POST + JSON helper.

Was defined separately in `urun/api/answering_model.py`,
`urun/api/retrieval/modules/top_n/embedder.py`, and
`urun/pipeline/vectorize/embedder/openai_compat.py` -- 3 copies, differing
only in docstring/whitespace, otherwise byte-identical. Consolidated here.

Consumers import this as `post_json` and bind it locally as `_post_json`
(`from medrag.core.llm.http import post_json as _post_json`) -- this keeps
the module-level name each test suite's `monkeypatch.setattr(<module>,
"_post_json", ...)` targets unchanged (~58 call sites across
test_answering_model.py, test_preflight.py, test_continuation.py,
test_slot_selection.py, test_reconciler.py, and vectorize's
test_embedder_openai_compat.py). Not renamed to `_post_json` here because
a leading underscore on a *public*, cross-component symbol is misleading.

Excluded from this consolidation:
`urun/pipeline/parser/llm/openai_compat.py`'s own `_post_json` (parser is
sys.path-imported, not an installed dependent of `medrag` yet -- folding it in
here now would make an uninstalled component depend on an installed one).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from medrag.core.llm.errors import ProviderError


def post_json(
    url: str, payload: dict, *, api_key: str | None, timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    """POST + JSON response. The only place this touches HTTP -- tests stub this.

    `extra_headers` (Anthropic support): merged in on top of the
    default `Authorization: Bearer` header -- Anthropic's Messages API uses
    `x-api-key`/`anthropic-version` instead, so `core/llm/anthropic.py` calls
    this with `api_key=None` (skip Bearer) and its own headers here. Existing
    ~58 call sites that never pass this keep the exact same behavior."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    for key, value in (extra_headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"cannot reach {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response from {url}: {exc}") from exc
