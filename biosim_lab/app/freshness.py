"""Detect a process running code older than the files on disk.

Streamlit re-runs the entry script on every interaction, but a module already in
``sys.modules`` is *not* re-imported. After a deploy that replaces the source
without restarting the process, the app therefore keeps executing the previous
bytecode while tracebacks are rendered from the new file. The result is the
worst kind of bug report: a stack trace whose line numbers belong to the old
code and whose source text belongs to the new, so a fix appears to have been
applied and to have changed nothing.

This module records a hash of each watched source file at import time and
re-reads them on demand. A mismatch means the running process is stale and only
a restart will pick the change up --- nothing in the app can force a re-import.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

#: Modules whose staleness would silently invalidate results, not just styling.
WATCHED: tuple[str, ...] = (
    "biosim_lab",
    "biosim_lab.core.materials",
    "biosim_lab.core.particles",
    "biosim_lab.instruments.saw_sorter.simulate",
    "biosim_lab.instruments.cell_tracker.tracking",
    "biosim_lab.instruments.cell_counter.counting",
    "biosim_lab.app.runners",
)


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def _current() -> dict[str, str]:
    """Hash every watched module's file as it stands on disk right now."""
    out: dict[str, str] = {}
    for name in WATCHED:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        digest = _digest(Path(file))
        if digest is not None:
            out[name] = digest
    return out


def _seed_baseline() -> dict[str, str]:
    """Import the watched modules, then hash them.

    Importing here is the whole point rather than a side effect: a hash is only
    evidence of staleness if it was taken when the bytecode was compiled. The
    runners import these lazily on first use, so without this the baseline would
    be empty for exactly the modules whose staleness matters. They are imported
    on the first page view anyway, so this moves the cost rather than adding it.

    An import failure is not fatal --- the module is simply not watched. A
    missing optional back-end must not take the app down.
    """
    for name in WATCHED:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - an unimportable module is just unwatched
            continue
    return _current()


#: Captured at process start, when the watched modules were compiled.
AT_IMPORT: dict[str, str] = _seed_baseline()


def stale_modules() -> list[str]:
    """Watched modules whose file on disk differs from the running bytecode."""
    now = _current()
    return sorted(
        name for name, digest in now.items()
        if name in AT_IMPORT and AT_IMPORT[name] != digest
    )


__all__ = ["AT_IMPORT", "WATCHED", "stale_modules"]
