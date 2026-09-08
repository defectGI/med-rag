"""Deterministic intent -> Flow selection: the answering model never picks
its own strategy at runtime. `config/default.toml` [routing] provides the
mapping, so adding a new path is one config line + a new Flow; this class is
untouched.
"""

from __future__ import annotations

from medrag.api.config import Routing
from medrag.api.flows.base import Flow
from medrag.api.retrieval.core import IntentLabel
from medrag.api.trace import TraceFn


class Router:
    """Verifies at construction that every `routing` reference actually exists
    in the `flows` registry (first_parse "fail loudly at the boundary"
    principle) -- a mistyped flow name doesn't wait silently until runtime."""

    def __init__(self, flows: dict[str, Flow], routing: Routing) -> None:
        if routing.default_flow not in flows:
            raise KeyError(
                f"routing.default_flow={routing.default_flow!r} kayıtlı flow'larda "
                f"yok: {sorted(flows)}"
            )
        for label, flow_name in routing.intents.items():
            if flow_name not in flows:
                raise KeyError(
                    f"routing.intents[{label!r}]={flow_name!r} kayıtlı flow'larda "
                    f"yok: {sorted(flows)}"
                )
        self._flows = flows
        self._routing = routing

    def flow_for(self, label: IntentLabel | str, on_trace: TraceFn | None = None) -> Flow:
        """`on_trace` is fully optional/observational (same general principle
        as `trace.py`) -- with `None` (the default), behavior/return value is
        exactly as before. When given, it makes the path DISTINCTLY
        identifiable as either a route mapped in `routing.intents` or a silent
        fall-through to `default_flow`: on a match `"routed"`, when `label` is
        NOT in `routing.intents` (fell back to default_flow)
        `"routed_default_fallback"` -- both carry the selected `flow_name` as
        data."""
        key = label.value if isinstance(label, IntentLabel) else label
        flow_name = self._routing.intents.get(key, self._routing.default_flow)
        if on_trace:
            event = "routed" if key in self._routing.intents else "routed_default_fallback"
            on_trace(event, flow_name)
        return self._flows[flow_name]

    def flow_by_name(self, name: str) -> Flow:
        """Lookup by direct flow name -- a pinned session is tied to a flow
        directly, not to an intent (see `SessionState.pinned` /
        `PinnedFlow.flow`), INDEPENDENT of the `routing.intents` mapping. An
        unknown name raises `KeyError` (first_parse "fail loudly at the
        boundary" principle -- same style as the validation in `__init__`, but
        since pin names are runtime data rather than construction-time data,
        it's validated here, at call time)."""
        try:
            return self._flows[name]
        except KeyError:
            raise KeyError(
                f"pinlenmiş flow adı kayıtlı flow'larda yok: {name!r} "
                f"(bilinenler: {sorted(self._flows)})"
            ) from None


__all__ = ["Router"]
