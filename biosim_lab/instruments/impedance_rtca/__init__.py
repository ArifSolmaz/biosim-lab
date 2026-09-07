"""Real-time cell impedance analyser (xCELLigence-style) — Stage 2 instrument."""

from biosim_lab.instruments.impedance_rtca.instrument import (
    ImpedanceRTCA,
    RTCAParams,
    logistic_coverage,
)

__all__ = ["ImpedanceRTCA", "RTCAParams", "logistic_coverage"]
