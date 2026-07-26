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
    raise SystemExit(
        f"Refusing to let {caller} read {ds_config.name}'s test split "
        f"({ds_config.test_csv}): already consumed on "
        f"{info.get('consumed_on')} by {info.get('consumed_by')} "
        f"(see {info.get('reference')}). The test split is single-use per "
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
