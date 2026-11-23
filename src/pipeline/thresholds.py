# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Single source of truth for the demo operating point.

The shipped demo defaults (Gun 0.44 / Knife 0.43, fallback 0.40) used
to live in five places — ``RunConfig`` defaults, the Pydantic schema,
the frontend inputs, the CLI constants, and three ``0.40`` literals in
the runner — so a v5 retune could silently desync them. Everything
imports from here now: the runner, the API schema, and the CLI. The
frontend fetches the same values from ``GET /api/defaults``.

Values: val-tuned operating point for the v2 weights (Gun: constrained
F1-optimal at precision >= 0.90; Knife: F1-optimal) — see
``sentinel-dataset-v2/thresholds/operating_point.md``. Re-tune after a
retrain and change ONLY this file (plus the frontend fetch, which reads
it through the API).
"""

from __future__ import annotations

# Per-class confidence thresholds for the shipped demo.
CONF_GUN = 0.44
CONF_KNIFE = 0.43

# Generic fallback for any class without a per-class threshold.
DEFAULT_CONF = 0.40


def as_dict() -> dict[str, float]:
    """The operating point as a JSON-able dict (served by /api/defaults)."""
    return {
        "conf_gun": CONF_GUN,
        "conf_knife": CONF_KNIFE,
        "conf_fallback": DEFAULT_CONF,
    }
