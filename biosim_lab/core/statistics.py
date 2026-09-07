"""Uncertainty on simulated measurements.

A single simulation run reports ``efficiency = 100.0 %`` and says nothing about
how much to trust it. Two different uncertainties are hiding behind that number
and they answer different questions.

**Counting uncertainty (within one run).** Even a perfectly deterministic device
sorts a *finite* number of cells, and a proportion measured on 300 cells is not
the proportion of the population. This is binomial and it is present even if you
never change the seed.

**Between-replicate uncertainty (across runs).** Each run draws fresh radii from
the log-normal size distribution, fresh inlet positions, and fresh viability
outcomes. Re-running with a new seed gives a different answer, and the spread
tells you how much of your result is the device and how much is the particular
sample you happened to draw.

Reporting only one of the two is misleading in opposite directions: the first
alone ignores sample-to-sample variability; the second alone, with few
replicates, badly underestimates the total.

Why Wilson and not the textbook interval
----------------------------------------
The interval most people reach for is ``p +- z sqrt(p(1-p)/n)``. For a sorter it
fails exactly where it matters: at ``p = 1`` it gives a width of **zero**, so
300 out of 300 is reported as ``100 % +- 0``, which is nonsense --- the next 300
cells will not all be captured with certainty. It can also produce bounds below
0 or above 1.

The Wilson score interval has neither pathology, is well behaved for small ``n``
and for proportions near the boundaries, and is the interval recommended for
routine use.

Reference: Brown, Cai & DasGupta (2001), *Interval estimation for a binomial
proportion*, Statist. Sci. 16:101, doi:10.1214/ss/1009213286. Original:
Wilson (1927), JASA 22:209, doi:10.1080/01621459.1927.10502953.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Two-sided normal quantiles for the confidence levels people actually use.
_Z = {0.80: 1.2815515655, 0.90: 1.6448536270, 0.95: 1.9599639845, 0.99: 2.5758293035}


def _z_for(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    from scipy.stats import norm

    return float(norm.ppf(0.5 + confidence / 2.0))


@dataclass(frozen=True)
class Proportion:
    """A measured proportion with its confidence interval, all as percentages."""

    successes: int
    trials: int
    confidence: float
    point_percent: float
    low_percent: float
    high_percent: float

    @property
    def half_width_percent(self) -> float:
        """Half the interval width --- the ``+-`` people expect to see."""
        return 0.5 * (self.high_percent - self.low_percent)

    def __str__(self) -> str:
        return (f"{self.point_percent:.1f} % "
                f"[{self.low_percent:.1f}, {self.high_percent:.1f}] "
                f"({self.successes}/{self.trials})")


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> Proportion:
    """Wilson score interval for a binomial proportion, in percent.

    Unlike the normal approximation this stays inside ``[0, 100]`` and has
    non-zero width at ``p = 0`` and ``p = 1``, which is the case a cell sorter
    hits constantly.

    >>> str(wilson_interval(300, 300))          # doctest: +ELLIPSIS
    '100.0 % [98.7, 100.0] (300/300)'
    """
    n = int(trials)
    k = int(successes)
    if n <= 0:
        return Proportion(k, n, confidence, float("nan"), float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError(f"successes={k} must lie in [0, {n}]")

    z = _z_for(confidence)
    p = k / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = (z / denominator) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    # In exact arithmetic centre - half is 0 when k == 0 and centre + half is 1
    # when k == n. In floating point they land a few 1e-17 away, which breaks the
    # invariant low <= point <= high that every caller is entitled to assume.
    low = min(max(0.0, centre - half), p)
    high = max(min(1.0, centre + half), p)
    return Proportion(
        successes=k,
        trials=n,
        confidence=confidence,
        point_percent=100.0 * p,
        low_percent=100.0 * low,
        high_percent=100.0 * high,
    )


@dataclass(frozen=True)
class ReplicateSummary:
    """Mean and spread of one metric across independent runs."""

    name: str
    values: np.ndarray
    mean: float
    std: float
    sem: float
    low: float
    high: float
    confidence: float

    @property
    def n(self) -> int:
        return int(self.values.size)

    def __str__(self) -> str:
        return (f"{self.mean:.2f} ± {self.sem:.2f} "
                f"[{self.low:.2f}, {self.high:.2f}] (n={self.n})")


def summarise_replicates(
    name: str, values: Sequence[float], confidence: float = 0.95
) -> ReplicateSummary:
    """Mean, standard deviation and a t-based confidence interval on the mean.

    Student's t rather than the normal quantile, because replicate counts are
    small --- with 5 runs the difference is 30 %, and using z would understate
    the interval.
    """
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = data.size
    if n == 0:
        nan = float("nan")
        return ReplicateSummary(name, data, nan, nan, nan, nan, nan, confidence)
    mean = float(data.mean())
    if n == 1:
        return ReplicateSummary(name, data, mean, 0.0, 0.0, mean, mean, confidence)

    std = float(data.std(ddof=1))
    sem = std / np.sqrt(n)
    from scipy.stats import t as student_t

    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return ReplicateSummary(
        name=name, values=data, mean=mean, std=std, sem=sem,
        low=mean - critical * sem, high=mean + critical * sem, confidence=confidence,
    )


def replicate(
    run: Callable[[int], dict[str, Any]],
    *,
    n_replicates: int = 5,
    base_seed: int = 0,
    metrics: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Run something *n_replicates* times with different seeds and summarise it.

    Parameters
    ----------
    run:
        ``run(seed) -> metrics dict``. Anything that returns a flat mapping of
        scalars works, so this is not specific to one instrument.
    metrics:
        Which keys to summarise. ``None`` takes every numeric key of the first
        run.

    Returns
    -------
    dict
        ``{"n_replicates": int, "seeds": [...], "per_replicate": [dict, ...],
        "summary": {name: ReplicateSummary}}``
    """
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1")

    seeds = [base_seed + i for i in range(n_replicates)]
    runs = [run(seed) for seed in seeds]

    if metrics is None:
        metrics = [
            key for key, value in runs[0].items()
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        ]

    summary = {
        name: summarise_replicates(
            name, [r.get(name, float("nan")) for r in runs], confidence
        )
        for name in metrics
    }
    return {
        "n_replicates": n_replicates,
        "seeds": seeds,
        "per_replicate": runs,
        "summary": summary,
    }


def summary_table(result: dict[str, Any]) -> Any:
    """Replicate summaries as a DataFrame, ready to print or download."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "metric": name,
                "mean": s.mean,
                "std": s.std,
                "sem": s.sem,
                f"ci{int(s.confidence * 100)}_low": s.low,
                f"ci{int(s.confidence * 100)}_high": s.high,
                "n": s.n,
            }
            for name, s in result["summary"].items()
        ]
    )


__all__ = [
    "Proportion",
    "ReplicateSummary",
    "wilson_interval",
    "summarise_replicates",
    "replicate",
    "summary_table",
]
