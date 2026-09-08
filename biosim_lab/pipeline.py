"""Chaining instruments into one workflow, and propagating the error through it.

Each instrument in this package answers one question well. A real experiment is
a *sequence* of them --- sort a suspension, count what came out, then watch the
survivors --- and two things only become visible when they are chained.

**The population changes as it moves.** Acoustic sorting is a size filter: the
radiation force scales with the cell's volume while drag scales with its radius,
so migration speed goes as ``r^2`` and the collected fraction is enriched in
large cells. The suspension that reaches the counter therefore has a *different
mean diameter and a narrower spread* than the one that was loaded. Run the two
instruments separately and you will calibrate the counter on the wrong
distribution; run them chained and the shift is measured for you.

**The uncertainties compose.** Each stage adds its own, from a different source:

* sorting adds *binomial* error, because a finite number of cells either make it
  to the collection outlet or do not;
* counting adds *Poisson* error, because a finite number of cells land in the
  chamber's field of view (``1/sqrt(N)``);
* any downstream assay adds *sample-to-sample* error across replicates.

These are independent, so they add in quadrature --- the total is *not* the sum,
and it is usually dominated by one stage. Knowing which one is the actionable
part: counting more fields of view cannot help if the sorter is the bottleneck.

This module is deliberately thin. It does not reimplement any instrument; it
carries a :class:`Sample` between them and keeps the audit trail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StageReport:
    """What one stage did to the sample, and what it cost in certainty."""

    stage: str
    n_in: int
    n_out: int
    #: Relative (1-sigma) uncertainty this stage contributes to the final count.
    relative_uncertainty: float
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def yield_fraction(self) -> float:
        return self.n_out / self.n_in if self.n_in else float("nan")


@dataclass
class Sample:
    """A cell suspension in transit between instruments.

    ``cells`` is one row per cell and must carry at least ``radius_m`` and
    ``label``; ``alive`` is used when present. This is the same shape the
    sorter's outlet table already has, so no translation layer is needed.
    """

    cells: pd.DataFrame
    volume_ml: float = 1.0
    history: list[StageReport] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.cells))

    @property
    def concentration_per_ml(self) -> float:
        return self.n / self.volume_ml if self.volume_ml else float("nan")

    def radius_stats(self) -> dict[str, float]:
        """Mean, spread and coefficient of variation of the cell radii [m]."""
        radii = np.asarray(self.cells["radius_m"], dtype=float)
        radii = radii[np.isfinite(radii)]
        if radii.size == 0:
            return {"mean_m": float("nan"), "std_m": float("nan"), "cv": float("nan")}
        mean = float(radii.mean())
        std = float(radii.std(ddof=1)) if radii.size > 1 else 0.0
        return {"mean_m": mean, "std_m": std, "cv": std / mean if mean else float("nan")}

    def total_relative_uncertainty(self) -> float:
        """Stage uncertainties combined in quadrature.

        Valid because the stages' error sources are independent: which cells the
        sorter collects says nothing about which ones land in the counting
        chamber's field of view. Correlated stages would need a covariance term.
        """
        contributions = [
            r.relative_uncertainty for r in self.history
            if np.isfinite(r.relative_uncertainty)
        ]
        return float(np.sqrt(np.sum(np.square(contributions)))) if contributions else 0.0

    def dominant_uncertainty(self) -> StageReport | None:
        """The stage contributing most of the total error --- what to fix first."""
        usable = [r for r in self.history if np.isfinite(r.relative_uncertainty)]
        return max(usable, key=lambda r: r.relative_uncertainty) if usable else None


class Stage(ABC):
    """One instrument's worth of work on a sample."""

    name: str = "stage"

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        """Transform *sample*, appending exactly one :class:`StageReport`."""


class Pipeline:
    """Run stages in order, threading one sample through them."""

    def __init__(self, stages: list[Stage]) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = stages

    def run(self, sample: Sample) -> Sample:
        for stage in self.stages:
            n_before = len(sample.history)
            sample = stage.apply(sample)
            if len(sample.history) != n_before + 1:
                raise RuntimeError(
                    f"stage {stage.name!r} must append exactly one StageReport; "
                    "the audit trail is what makes the error budget meaningful"
                )
        return sample

    def summary(self, sample: Sample) -> str:
        """The error budget, as a table a methods section can quote."""
        lines = [
            f"{'stage':<14} {'n in':>8} {'n out':>8} {'yield':>8} {'rel. sigma':>11}",
        ]
        for report in sample.history:
            lines.append(
                f"{report.stage:<14} {report.n_in:>8d} {report.n_out:>8d} "
                f"{report.yield_fraction:>7.1%} {report.relative_uncertainty:>10.2%}"
            )
        total = sample.total_relative_uncertainty()
        lines.append(f"{'-' * 52}")
        lines.append(f"{'combined':<14} {'':>8} {sample.n:>8d} {'':>8} {total:>10.2%}")
        dominant = sample.dominant_uncertainty()
        if dominant is not None and total > 0:
            share = (dominant.relative_uncertainty / total) ** 2
            lines.append(
                f"dominated by {dominant.stage!r} "
                f"({share:.0%} of the variance) --- improve that stage first"
            )
        return "\n".join(lines)


__all__ = ["Sample", "Stage", "StageReport", "Pipeline"]
