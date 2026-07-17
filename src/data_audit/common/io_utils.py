"""Shared, safety-guarded I/O helpers for the audit pipeline.

Every function that writes anything to disk routes through
`assert_not_raw_path`, which raises if the target path falls inside
data/raw/. This turns the "never modify raw data" rule into something
enforced at runtime, not just a convention the scripts are expected to
follow.
"""

from pathlib import Path

import pandas as pd

from src.data_audit.config import RAW_ROOT


class RawDataWriteError(RuntimeError):
    """Raised if any code attempts to write inside data/raw/."""


def assert_not_raw_path(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == RAW_ROOT.resolve() or RAW_ROOT.resolve() in resolved.parents:
        raise RawDataWriteError(
            f"Refusing to write to '{resolved}': path is inside the "
            f"read-only raw data directory '{RAW_ROOT}'."
        )


def ensure_dir(path: Path) -> Path:
    assert_not_raw_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    assert_not_raw_path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    assert_not_raw_path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def list_files(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    """Recursively list files under `directory` (read-only walk)."""
    if not directory.exists():
        return []
    files = [p for p in directory.rglob("*") if p.is_file()]
    if extensions is not None:
        files = [p for p in files if p.suffix.lower() in extensions]
    return sorted(files)
