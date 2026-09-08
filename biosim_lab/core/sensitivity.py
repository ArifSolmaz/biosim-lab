"""Which unsourced numbers actually matter?

The material library flags every value it could not source to a DOI as an
ASSUMPTION. That list is honest but not actionable: it says *what* is unknown,
not what the ignorance costs. Most assumptions turn out to be irrelevant to the
answer, and one or two dominate it. This module finds which is which, so a user
with limited time knows what to go and measure.

The measure: normalised elasticity
----------------------------------
For a model output ``Y`` and an input ``X``,

    S = (dY / Y) / (dX / X)

is dimensionless, so densities, viscosities and radii can be ranked against each
other on one axis. ``S = 1`` means a 10 % error in the input gives a 10 % error
in the answer; ``S = 0`` means the input does not matter. This is the standard
local sensitivity coefficient (see e.g. Saltelli et al., *Global Sensitivity
Analysis: The Primer*, doi:10.1002/9780470725184, ch. 1, which also explains what
a local index does *not* tell you --- see the caveat below).

Two things that are easy to get wrong
-------------------------------------
**Common random numbers.** The sorting model is stochastic: it samples cell radii
and inlet positions. Re-running it with a different seed changes the efficiency
by a percent or so even with every input fixed. If baseline and perturbed runs
use different seeds, that sampling noise lands in the numerator of ``S`` and the
ranking becomes a ranking of noise. Every run here therefore uses the *same*
seed, so the difference is caused by the parameter and nothing else.

**Judging significance on the right quantity.** The measured quantity is the
*paired difference*, so its uncertainty is the spread of that difference across
seeds --- not the spread of the raw metric. The two differ by a large factor
here, and using the raw spread rejects real effects: at the marginal operating
point the assumed MCF7 density moves efficiency by 7.4 points while the metric
itself wanders by 9 points from resampling alone. The paired difference is
nonetheless stable to a fraction of a point, and a paired t-test over the seeds
sees it. An unresolved entry means "too small to see at this ensemble size and
seed count", not "zero".

Caveat: this is a *local* index
-------------------------------
Elasticity is a derivative at one operating point. It will miss an input that
does nothing at 6.6 MHz but dominates at 20 MHz, and it cannot see interactions
between inputs. Re-run the scan at each operating point you care about. For a
full variance decomposition you want Sobol indices, which cost thousands of runs
rather than the few dozen here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .materials import audit, get_cell, get_fluid, get_substrate, perturbed

_GETTERS = {"fluid": get_fluid, "cell": get_cell, "substrate": get_substrate}


@dataclass(frozen=True)
class Target:
    """One library value to vary."""

    group: str
    material: str
    prop: str
    provenance: str = ""

    @property
    def label(self) -> str:
        return f"{self.material}.{self.prop}"

    def current(self) -> float:
        value = getattr(_GETTERS[self.group](self.material), self.prop)
        return float(value.magnitude)


def assumption_targets(
    groups: Sequence[str] | None = None,
    materials: Sequence[str] | None = None,
) -> list[Target]:
    """Every ASSUMPTION-flagged value in the library, as scan targets.

    Filter with *groups* / *materials* to scan only what a given experiment
    actually touches --- scanning a cell type that is not in the suspension
    wastes runs and produces a row of exact zeros.
    """
    targets = []
    for row in audit(only_assumptions=True):
        if groups is not None and row["group"] not in groups:
            continue
        if materials is not None and row["material"] not in materials:
            continue
        targets.append(
            Target(row["group"], row["material"], row["property"], row["provenance"])
        )
    return targets


@dataclass
class SensitivityRow:
    """Elasticity of one metric with respect to one input."""

    material: str
    prop: str
    group: str
    metric: str
    baseline_value: float
    baseline_metric: float
    elasticity: float
    delta_metric: float
    resolved: bool
    provenance: str = ""

    @property
    def label(self) -> str:
        return f"{self.material}.{self.prop}"


@dataclass
class SensitivityReport:
    """The ranked result of a scan."""

    rows: list[SensitivityRow] = field(default_factory=list)
    noise_floor: dict[str, float] = field(default_factory=dict)
    fraction: float = 0.05
    n_runs: int = 0

    def ranked(self, metric: str, resolved_only: bool = False) -> list[SensitivityRow]:
        """Rows for *metric*, largest absolute elasticity first."""
        rows = [r for r in self.rows if r.metric == metric]
        if resolved_only:
            rows = [r for r in rows if r.resolved]
        def key(row: SensitivityRow) -> float:
            return -abs(row.elasticity) if np.isfinite(row.elasticity) else 0.0

        return sorted(rows, key=key)

    def to_dataframe(self) -> Any:
        import pandas as pd

        return pd.DataFrame([vars(r) for r in self.rows])

    def is_saturated(self, metric: str) -> bool:
        """True when nothing moved the metric, because the device is at a rail.

        A sorter running at 100 % efficiency has a genuinely zero local
        derivative with respect to everything: the next cell cannot be collected
        any harder. That is a real result --- the design point is robust --- but
        it produces a table of exact zeros that reads like a broken scan, so it
        is worth naming explicitly.
        """
        rows = [r for r in self.rows if r.metric == metric]
        if not rows:
            return False
        return all(r.delta_metric == 0.0 for r in rows)

    def summary(self, metric: str, top: int = 10) -> str:
        """A plain-text ranking, for a methods section or the CLI."""
        rows = self.ranked(metric, resolved_only=False)[:top]
        if not rows:
            return f"no sensitivity rows for metric {metric!r}"
        floor = self.noise_floor.get(metric, 0.0)
        if self.is_saturated(metric):
            baseline = rows[0].baseline_metric
            return (
                f"Sensitivity of {metric!r}: SATURATED at {baseline:g}.\n"
                f"  No perturbation of any of the {len(rows)} inputs changed it, because the\n"
                f"  device is at a rail at this operating point and the local derivative is\n"
                f"  genuinely zero. The design point is robust to every unsourced number\n"
                f"  tested. To learn which inputs matter, re-scan nearer the decision\n"
                f"  boundary --- that is where an assumption error changes the outcome."
            )
        out = [
            f"Sensitivity of {metric!r} to unsourced inputs "
            f"(+/-{self.fraction:.0%}, {self.n_runs} runs)",
            f"  metric spread across seeds at fixed inputs: {floor:.4g}",
            f"  {'input':<38} {'elasticity':>11}  {'d(metric)':>10}  note",
        ]
        for r in rows:
            if r.resolved:
                note = ""
            elif r.delta_metric == 0.0:
                # Not a measurement that came out too small to see: the model
                # never read this value, so its influence is exactly nothing.
                note = "unused by this model"
            else:
                note = "not resolved"
            out.append(
                f"  {r.group + ':' + r.label:<38} {r.elasticity:>11.3f}  "
                f"{r.delta_metric:>10.4g}  {note}"
            )
        return "\n".join(out)


# Per-row false-positive rate the resolution test targets. At 0.5 % a scan over
# the whole 38-value assumption library is expected to invent ~0.2 findings.
_NOISE_ALPHA = 0.005

DEFAULT_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)


def _critical_value(dof: int) -> float:
    """Two-sided t critical value at :data:`_NOISE_ALPHA`."""
    if dof < 1:
        return float("inf")
    from scipy.stats import t as student_t

    return float(student_t.ppf(1.0 - _NOISE_ALPHA / 2.0, dof))


def scan(
    run: Callable[[int], dict[str, float]],
    targets: Iterable[Target],
    *,
    metrics: Sequence[str],
    fraction: float = 0.05,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> SensitivityReport:
    """Rank *targets* by how much they move *metrics*.

    Parameters
    ----------
    run:
        ``run(seed)`` executes the model once with the library as it currently
        stands and returns a metric dict. It must be deterministic given the
        seed --- that is what makes the design below work.
    targets:
        Which library values to vary, e.g. from :func:`assumption_targets`.
    metrics:
        Keys of the dict *run* returns to track.
    fraction:
        Relative perturbation, applied both up and down. The default 5 % is
        small enough to stay in the linear regime for a smooth output and large
        enough to move a discrete one. Central differencing cancels the leading
        curvature term, so it is second-order accurate in *fraction*.
    seeds:
        Ensembles to repeat the whole measurement over. At least three, or
        nothing can be resolved.

    The design: paired differences
    ------------------------------
    For each seed the up- and down-perturbed runs use **that same seed**, so the
    two ensembles are identical and their difference isolates the parameter.
    That is the classic common-random-numbers variance reduction, and it is what
    makes a 5 % perturbation measurable at all against a model whose output
    moves by several percent from resampling alone.

    It also decides how significance must be judged. The quantity of interest is
    the paired difference, so the uncertainty on it is the spread of that
    difference **across seeds** --- not the spread of the raw metric, which is
    the wrong yardstick by a large factor and rejects real effects. A finding is
    resolved when its mean difference clears ``t * sem`` over the seeds, which
    is a paired t-test.
    """
    targets = list(targets)
    seeds = list(seeds)
    if len(seeds) < 3:
        raise ValueError(f"need at least three seeds to resolve anything, got {len(seeds)}")

    baselines = {seed: run(seed) for seed in seeds}
    n_runs = len(seeds)
    dof = len(seeds) - 1
    critical = _critical_value(dof)

    def mean_metric(results: dict[int, dict[str, float]], metric: str) -> float:
        values = [float(results[s].get(metric, np.nan)) for s in seeds]
        return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")

    rows: list[SensitivityRow] = []
    noise: dict[str, float] = {}
    for metric in metrics:
        spread = [float(baselines[s].get(metric, np.nan)) for s in seeds]
        finite = np.asarray([v for v in spread if np.isfinite(v)])
        noise[metric] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0

    for target in targets:
        x0 = target.current()
        if x0 == 0.0:
            # A relative perturbation of zero is zero; elasticity is undefined.
            for metric in metrics:
                rows.append(SensitivityRow(
                    target.material, target.prop, target.group, metric, x0,
                    mean_metric(baselines, metric), float("nan"), 0.0, False,
                    target.provenance,
                ))
            continue

        up: dict[int, dict[str, float]] = {}
        down: dict[int, dict[str, float]] = {}
        for sign, store in ((+1.0, up), (-1.0, down)):
            change = (target.group, target.material, target.prop,
                      x0 * (1.0 + sign * fraction))
            with perturbed([change]):
                for seed in seeds:
                    store[seed] = run(seed)
                    n_runs += 1

        for metric in metrics:
            y0 = mean_metric(baselines, metric)
            deltas = np.asarray([
                float(up[s].get(metric, np.nan)) - float(down[s].get(metric, np.nan))
                for s in seeds
            ], dtype=float)
            good = deltas[np.isfinite(deltas)]
            if good.size == 0 or not np.isfinite(y0) or y0 == 0.0:
                rows.append(SensitivityRow(
                    target.material, target.prop, target.group, metric, x0, y0,
                    float("nan"), float("nan"), False, target.provenance,
                ))
                continue

            mean_delta = float(np.mean(good))
            sem = (
                float(np.std(good, ddof=1) / np.sqrt(good.size)) if good.size > 1 else 0.0
            )
            # A difference that is identically zero across every seed is a
            # genuine null, not an unresolved measurement.
            resolved = bool(
                mean_delta != 0.0
                and (sem == 0.0 or abs(mean_delta) > critical * sem)
            )
            rows.append(SensitivityRow(
                target.material, target.prop, target.group, metric, x0, y0,
                mean_delta / (2.0 * fraction * y0), mean_delta, resolved,
                target.provenance,
            ))

    return SensitivityReport(rows=rows, noise_floor=noise, fraction=fraction, n_runs=n_runs)


__all__ = [
    "Target",
    "SensitivityRow",
    "SensitivityReport",
    "assumption_targets",
    "scan",
]
