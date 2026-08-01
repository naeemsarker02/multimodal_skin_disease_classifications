"""Shared guard preventing a dataset's held-out test split from being
touched more than once, across every script that can read it
(`evaluate.py`, `evaluate_fairness.py`, `evaluate_cross_dataset.py`).

Added 2026-07-25 after `evaluate_fairness.py` read PAD-UFES-20's
`metadata_test.csv` without going through `evaluate.py`'s
`--confirm-final` flag - that flag only gated *that one script*, so a
second script touching the same file bypassed the "test split used only
once" discipline (Project_Tracking.md decision 4) undetected. The marker
file this module writes/checks is dataset-scoped, not script-scoped, so
no future script can repeat that gap by construction.

Usage: call `check_test_split_available(ds_config, caller)` before any
code path that reads `ds_config.test_csv`; call
`mark_test_split_consumed(...)` once, immediately after the sanctioned
one-time run that consumes it.
"""

import json
from pathlib import Path


def marker_path(ds_config) -> Path:
    return ds_config.test_csv.parent / "TEST_SPLIT_CONSUMED.json"


def check_test_split_available(ds_config, caller: str) -> None:
    marker = marker_path(ds_config)
    if not marker.exists():
        return
    info = json.loads(marker.read_text())
    # Two marker schemas coexist: the original single-event schema
    # (consumed_on/consumed_by/reference at top level) and the
    # consumption_events list schema (added 2026-08-01 when PAD_UFES20's
    # guard was reopened for a second sanctioned read - see
    # Project_Tracking.md "PAD-UFES-20 Test-Split Guard Reopened for Step
    # 4"). Report the most recent event either way, most-severe/most-recent
    # first is not needed here - just show what most recently locked it.
    if "consumption_events" in info:
        latest = info["consumption_events"][-1]
    else:
        latest = info
    raise SystemExit(
        f"Refusing to let {caller} read {ds_config.name}'s test split "
        f"({ds_config.test_csv}): already consumed on "
        f"{latest.get('consumed_on')} by {latest.get('consumed_by')} "
        f"(see {latest.get('reference')}). The test split is single-use per "
        "Project_Tracking.md decision 4 and is now locked for this "
        f"dataset. Marker file: {marker}. Do not delete this marker to "
        "work around this check - if the user explicitly wants to reopen "
        "this decision, that itself must be logged in Project_Tracking.md "
        "before the marker is removed."
    )


def mark_test_split_consumed(ds_config, caller: str, consumed_on: str, reference: str, detail: dict) -> Path:
    marker = marker_path(ds_config)
    marker.write_text(json.dumps({
        "dataset": ds_config.name,
        "consumed_on": consumed_on,
        "consumed_by": caller,
        "reference": reference,
        "detail": detail,
    }, indent=2))
    return marker
