"""Time-switched drive for the alternating-frequency BAW sorter.

One transducer, several frequencies, played in a fixed cycle
(Zhang et al. 2023, doi:10.3390/ijms24043338, Sec. 4.1: two sine sources fed
through an SPDT relay driven by a square wave). Each phase excites a different
transverse resonance of the channel, so the node pattern --- and with it the
direction of the force on a given cell --- jumps at every switch.

The one thing that makes this awkward numerically is that cells do not enter
the acoustic region in step with the relay. A cell's *own* clock starts when it
enters, so the phase it sees at simulation time ``t`` is the one at
``t + entry_offset``. :class:`Schedule` evaluates that per cell in one
vectorised call and lists every instant at which any cell's forcing jumps, so
the integrator can land on each of them (:mod:`biosim_lab.core.particles`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Schedule:
    """A periodic sequence of drive phases.

    Attributes
    ----------
    durations:
        Seconds per phase, in cycle order.
    mode_numbers:
        Transverse resonance index excited in each phase.
    energy_densities:
        ``E_ac`` in each phase [J/m^3].
    frequencies:
        Drive frequency of each phase [Hz] (for labelling and heating only).
    labels:
        Display names, e.g. ``"1 MHz"``.
    """

    durations: np.ndarray
    mode_numbers: np.ndarray
    energy_densities: np.ndarray
    frequencies: np.ndarray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.durations) < 2:
            raise ValueError("a switching schedule needs at least two phases")
        if np.any(np.asarray(self.durations) <= 0):
            raise ValueError("every phase duration must be positive")

    @property
    def period(self) -> float:
        """Length of one full switching cycle [s]."""
        return float(np.sum(self.durations))

    @property
    def phase_starts(self) -> np.ndarray:
        """Start of each phase within the cycle [s]."""
        return np.concatenate([[0.0], np.cumsum(self.durations)[:-1]])

    def phase_index(self, t: np.ndarray | float) -> np.ndarray:
        """Index of the phase active at cycle time(s) *t* (any real, wrapped)."""
        tau = np.mod(np.asarray(t, dtype=float), self.period)
        edges = np.cumsum(self.durations)
        return np.minimum(np.searchsorted(edges, tau, side="right"), len(self.durations) - 1)

    def switch_times(
        self, t0: float, t1: float, offsets: np.ndarray | float = 0.0
    ) -> np.ndarray:
        """Every simulation time in ``(t0, t1)`` at which some cell's phase changes.

        A cell with entry offset ``o`` switches at ``t = s + m * period - o``
        for each phase start ``s`` and integer ``m``.
        """
        offs = np.unique(np.atleast_1d(np.asarray(offsets, dtype=float)))
        out: list[np.ndarray] = []
        starts = self.phase_starts
        for o in offs:
            m_lo = int(np.floor((t0 + o) / self.period)) - 1
            m_hi = int(np.ceil((t1 + o) / self.period)) + 1
            m = np.arange(m_lo, m_hi + 1)
            t = (starts[None, :] + m[:, None] * self.period - o).ravel()
            out.append(t[(t > t0) & (t < t1)])
        return np.unique(np.concatenate(out)) if out else np.array([])

    def timeline(
        self, t0: float, t1: float, offset: float = 0.0
    ) -> list[dict[str, float | str]]:
        """Phase intervals seen by a cell with *offset*, for shading a time axis."""
        edges = np.unique(np.concatenate([[t0, t1], self.switch_times(t0, t1, offset)]))
        rows: list[dict[str, float | str]] = []
        for a, b in zip(edges[:-1], edges[1:]):
            i = int(self.phase_index(0.5 * (a + b) + offset))
            rows.append({
                "start": float(a), "end": float(b), "phase": i,
                "label": self.labels[i], "frequency_Hz": float(self.frequencies[i]),
            })
        return rows

    def cycles_in(self, duration: float) -> float:
        """How many full switching cycles fit into *duration* seconds."""
        return float(duration) / self.period


__all__ = ["Schedule"]
