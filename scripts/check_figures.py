#!/usr/bin/env python3
"""Fail if any generated figure is blank or near-blank.

Why this exists
---------------
``examples/03_parameter_sweep.py`` once shipped ``03_purity_percent_frequency_19p89.png``
as a completely empty plot: every value in that slice was NaN, so Plotly drew no
data, invented its own axes, and wrote a perfectly valid 76 kB PNG. The script
exited 0. Nothing in the test suite noticed, because nothing looked.

A unit test now covers that specific case. This covers the general one: after the
examples run, every figure they produced must actually contain something.

Usage
-----
    python scripts/check_figures.py assets
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

#: A figure whose pixels are this uniform is almost certainly empty. Axes,
#: labels and a title alone land around 0.02-0.05 by this measure.
MIN_STD = 6.0

#: Fraction of pixels allowed to be the single most common colour.
MAX_UNIFORM_FRACTION = 0.985


def inspect(path: Path) -> tuple[bool, str]:
    """Return ``(ok, explanation)`` for one image."""
    try:
        import imageio.v3 as iio
    except ImportError:  # pragma: no cover
        return True, "imageio unavailable, skipped"

    image = np.asarray(iio.imread(path), dtype=float)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)

    std = float(image.std())
    values, counts = np.unique(image.round().astype(int), return_counts=True)
    uniform = float(counts.max() / image.size)

    if std < MIN_STD:
        return False, f"near-uniform (std {std:.2f} < {MIN_STD})"
    if uniform > MAX_UNIFORM_FRACTION:
        return False, f"{uniform:.1%} of pixels are one colour"
    return True, f"std {std:.1f}, most common colour {uniform:.1%}"


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "assets")
    figures = sorted(root.glob("*.png"))
    if not figures:
        print(f"no PNGs found in {root}/ — did the examples run?", file=sys.stderr)
        return 1

    failures = []
    for path in figures:
        ok, why = inspect(path)
        print(f"{'ok  ' if ok else 'BLANK'}  {path.name:<48s} {why}")
        if not ok:
            failures.append((path, why))

    print(f"\n{len(figures) - len(failures)}/{len(figures)} figures contain data")
    if failures:
        print("\nBlank or near-blank figures:", file=sys.stderr)
        for path, why in failures:
            print(f"  {path}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
