"""Shared machinery for the literature benchmarks: comparison records and reporting.

Every number a benchmark reports against a paper goes through a :class:`Check`,
so the report cannot quietly mix a fitted value with a prediction, or a trend
with a quantity. There are five kinds, and they are reported differently:

``quantity``
    A number the paper states in its text, compared with ours. PASS when the
    absolute difference is within ``tolerance``; otherwise **DEVIATION** --- it
    is reported with its likely cause, never turned into a test failure.
``range``
    The paper states an interval ("83-96 %"). PASS inside it, DEVIATION with
    the distance to the nearer end outside it.
``trend``
    A qualitative statement ("the optimum tilt angle decreases as the flow rate
    increases"). PASS or FAIL: a model that gets a stated trend backwards is
    wrong, and the tests assert these.
``claim``
    A qualitative conclusion of the paper that the model may or may not
    support. REPRODUCED or NOT REPRODUCED --- the latter is a finding.
``calibration``
    A paper value the model was *fitted* to. Reported so the reader can see it
    was used, and so nobody counts it as agreement.

Reference values come only from numbers stated in the papers' text and tables,
never from reading their plots (see ``reference.yaml`` in each benchmark).
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

PASS = "PASS"
DEVIATION = "DEVIATION"
FAIL = "FAIL"
REPRODUCED = "REPRODUCED"
NOT_REPRODUCED = "NOT REPRODUCED"
CALIBRATION = "CALIBRATION"


class BenchmarkDeviation(UserWarning):
    """A quantitative benchmark value outside tolerance: reported, not failed."""


@dataclass
class Check:
    """One comparison between this model and a paper."""

    case: str
    quantity: str
    kind: str
    ours: float | str | None
    paper: float | str | Sequence[float] | None
    unit: str = ""
    tolerance: float | None = None
    note: str = ""
    cause: str = ""
    status: str = field(default="", init=True)
    passed: bool | None = None

    def __post_init__(self) -> None:
        if self.status:
            return
        if self.kind == "calibration":
            self.status = CALIBRATION
        elif self.kind == "trend":
            self.status = PASS if self.passed else FAIL
        elif self.kind == "claim":
            self.status = REPRODUCED if self.passed else NOT_REPRODUCED
        elif self.kind == "range":
            lo, hi = self.paper  # type: ignore[misc]
            ok = self.ours is not None and lo <= float(self.ours) <= hi  # type: ignore[arg-type]
            self.status = PASS if ok else DEVIATION
        elif self.kind == "quantity":
            ok = (
                self.ours is not None and self.paper is not None
                and not _nan(self.ours)
                and abs(float(self.ours) - float(self.paper)) <= float(self.tolerance or 0.0)  # type: ignore[arg-type]
            )
            self.status = PASS if ok else DEVIATION
        else:
            raise ValueError(f"unknown check kind {self.kind!r}")
        if self.status == DEVIATION and not self.cause:
            raise ValueError(
                f"{self.case}/{self.quantity}: a DEVIATION must state its likely cause"
            )

    @property
    def deviation(self) -> float | None:
        """Signed ours-minus-paper for quantities; distance outside for ranges."""
        if _nan(self.ours) or self.ours is None or isinstance(self.ours, str):
            return None
        if self.kind == "quantity" and isinstance(self.paper, (int, float)):
            return float(self.ours) - float(self.paper)
        if self.kind == "range":
            lo, hi = self.paper  # type: ignore[misc]
            v = float(self.ours)
            return 0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo)
        return None

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["deviation"] = self.deviation
        if isinstance(self.paper, (list, tuple)):
            d["paper"] = f"{self.paper[0]:g}-{self.paper[1]:g}"
        return d


def _nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def fmt(value: Any, digits: int = 1) -> str:
    """Compact rendering for report tables."""
    if value is None or _nan(value):
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "–".join(fmt(v, digits) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def checks_table(checks: Iterable[Check]) -> str:
    """Markdown table: case | quantity | ours | paper | deviation | status."""
    rows = [
        "| Case | Quantity | Ours | Paper | Deviation | Status |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        dev = c.deviation
        dev_s = "—" if dev is None else f"{dev:+.1f}"
        unit = f" {c.unit}" if c.unit and not isinstance(c.ours, str) else ""
        rows.append(
            f"| {c.case} | {c.quantity} | {fmt(c.ours)}{unit} | {fmt(c.paper)}{unit} "
            f"| {dev_s} | **{c.status}** |"
        )
    return "\n".join(rows)


def warn_deviations(checks: Iterable[Check]) -> None:
    """Emit one :class:`BenchmarkDeviation` per DEVIATION (pytest shows them)."""
    for c in checks:
        if c.status == DEVIATION:
            warnings.warn(
                f"{c.case} {c.quantity}: ours {fmt(c.ours)} vs paper {fmt(c.paper)} "
                f"{c.unit} — {c.cause}",
                BenchmarkDeviation,
                stacklevel=2,
            )


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parallel_map(fn: Callable[[Any], Any], items: Sequence[Any], *,
                 workers: int | None = None) -> list[Any]:
    """``[fn(x) for x in items]`` across processes; serial when asked or tiny.

    ``BIOSIM_BENCH_WORKERS=1`` forces serial execution (useful under a
    debugger or in a constrained container).
    """
    env = os.environ.get("BIOSIM_BENCH_WORKERS")
    n = int(env) if env else (workers or min(8, os.cpu_count() or 1))
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, items))


def save_figure(fig: Any, directory: Path, stem: str) -> dict[str, str]:
    """Write ``stem.html`` always and ``stem.png`` when kaleido can."""
    directory.mkdir(parents=True, exist_ok=True)
    out = {"html": str(directory / f"{stem}.html")}
    fig.write_html(out["html"], include_plotlyjs="cdn")
    try:
        fig.write_image(str(directory / f"{stem}.png"), width=1100, height=560, scale=2)
        out["png"] = str(directory / f"{stem}.png")
    except Exception as exc:  # noqa: BLE001 - kaleido is optional
        out["png_error"] = str(exc)
    return out


def write_summary(directory: Path, payload: dict[str, Any], checks: Sequence[Check]) -> None:
    """Persist a benchmark's machine-readable outcome for :mod:`benchmarks.make_report`."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([c.as_row() for c in checks]).to_csv(directory / "comparison.csv", index=False)
    with open(directory / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload | {"checks": [c.as_row() for c in checks]}, fh, indent=2,
                  default=_json_default)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


__all__ = [
    "ROOT",
    "PASS",
    "DEVIATION",
    "FAIL",
    "REPRODUCED",
    "NOT_REPRODUCED",
    "CALIBRATION",
    "BenchmarkDeviation",
    "Check",
    "fmt",
    "checks_table",
    "warn_deviations",
    "load_yaml",
    "parallel_map",
    "save_figure",
    "write_summary",
]
