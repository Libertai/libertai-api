"""Guardrails for the shipped Grafana dashboard (grafana/libertai-api-dashboard.json).

prometheus-client renders histogram bucket bounds as floats (le="60.0"), and
Prometheus label matchers compare strings exactly, so an le selector that
disagrees with the rendering (le="60") matches no series and the affected
panel silently renders "No data" or its fallback 0. This renders the real
exposition and checks every le selector in the dashboard against it.
"""

import json
import re
from pathlib import Path

from src import metrics

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "grafana" / "libertai-api-dashboard.json"
LE_SELECTOR = re.compile(r'le="([^"]+)"')


def _dashboard_expressions() -> list[str]:
    """Every target expression in the dashboard, including nested row panels."""
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    expressions = []
    panels = list(dashboard["panels"])
    while panels:
        panel = panels.pop()
        panels.extend(panel.get("panels", []))
        expressions.extend(target["expr"] for target in panel.get("targets", []))
    return expressions


def _rendered_le_values() -> set[str]:
    """The le label values the real exposition renders, across all histograms.

    Each histogram needs one child series for its buckets to render; the child
    label values themselves are arbitrary.
    """
    metrics.http_request_duration_seconds.labels("chat_completions", "POST").observe(0.5)
    metrics.upstream_request_duration_seconds.labels("qwen3.8-27b").observe(0.5)
    metrics.gate_wait_seconds.labels("qwen3.8-27b").observe(0.5)
    return set(LE_SELECTOR.findall(metrics._render_metrics().decode()))


def test_dashboard_le_selectors_match_rendered_buckets() -> None:
    expressions = _dashboard_expressions()
    assert not any("le=~" in expr for expr in expressions), (
        "regex le selectors bypass this test's exact-match validation; keep exact le selectors"
    )
    selectors = {value for expr in expressions for value in LE_SELECTOR.findall(expr)}
    assert selectors, "expected the dashboard to contain le selectors"
    rendered = _rendered_le_values()
    assert selectors <= rendered, (
        f"le selectors that match no rendered bucket: {sorted(selectors - rendered)}; rendered: {sorted(rendered)}"
    )
