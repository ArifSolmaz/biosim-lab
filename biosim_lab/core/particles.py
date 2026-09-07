"""Lagrangian particle tracking with a pluggable force registry.

Regimes
-------
A cell of radius ``r`` in a viscous liquid has a velocity relaxation time

    tau_p = 2 * rho_p * r^2 / (9 * mu)        [s]

For an 18 um cell in water that is ~2e-5 s, five orders of magnitude below the
~1 s channel transit time, so the **overdamped** limit is the right default:
inertia is dropped and the particle velocity follows the instantaneous force
balance

    v = u_fluid + F_ext / (6 * pi * mu * r)

The full inertial ODE (``m dv/dt = sum(F)``) is available via
``mode="inertial"`` for completeness and for validating the overdamped
assumption; it is stiff and much slower.

Reference for the drag law and its validity range:
Batchelor, *An Introduction to Fluid Dynamics*, doi:10.1017/CBO9780511800955,
section 4.9 (Stokes' law, valid for particle Reynolds number Re_p << 1).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import solve_ivp

from biosim_lab.core.plugin import RegimeWarning

#: A force model maps the current state onto an ``(n, dim)`` force array [N].
ForceFunction = Callable[["ParticleState"], np.ndarray]


@dataclass
class ParticleState:
    """Instantaneous state of the whole ensemble.

    Attributes
    ----------
    t:
        Time [s].
    x:
        Positions, shape ``(n, dim)`` [m].
    v:
        Velocities, shape ``(n, dim)`` [m/s].
    radius:
        Radii, shape ``(n,)`` [m].
    density:
        Particle densities, shape ``(n,)`` [kg/m^3].
    compressibility:
        Particle compressibilities, shape ``(n,)`` [1/Pa].
    label:
        Population label per particle (e.g. ``"mcf7"``), shape ``(n,)``.
    extra:
        Free-form per-particle arrays a force model may need.
    """

    t: float
    x: np.ndarray
    v: np.ndarray
    radius: np.ndarray
    density: np.ndarray
    compressibility: np.ndarray
    label: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def volume(self) -> np.ndarray:
        """Particle volumes [m^3]."""
        return 4.0 / 3.0 * np.pi * self.radius**3

    @property
    def mass(self) -> np.ndarray:
        """Particle masses [kg]."""
        return self.density * self.volume


class ForceModel(Protocol):
    """Structural type for anything the registry can hold."""

    name: str

    def __call__(self, state: ParticleState) -> np.ndarray: ...


@dataclass
class _Entry:
    name: str
    fn: ForceFunction
    enabled: bool = True
    description: str = ""


class ForceRegistry:
    """Ordered, individually switchable collection of force models.

    Every force contributes an ``(n, dim)`` array in newtons.  Keeping them
    separate (rather than summing inside one big function) is what makes the
    per-force diagnostics and the "turn secondary effects off by default"
    policy possible.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        name: str,
        fn: ForceFunction,
        *,
        enabled: bool = True,
        description: str = "",
    ) -> None:
        """Add (or replace) a force model."""
        self._entries[name] = _Entry(name, fn, enabled, description)

    def enable(self, name: str, enabled: bool = True) -> None:
        """Switch a registered force on or off."""
        if name not in self._entries:
            raise KeyError(f"no force named {name!r}; have: {sorted(self._entries)}")
        self._entries[name].enabled = enabled

    def disable(self, name: str) -> None:
        """Switch a registered force off."""
        self.enable(name, False)

    @property
    def active(self) -> list[str]:
        """Names of the currently enabled forces, in registration order."""
        return [e.name for e in self._entries.values() if e.enabled]

    def evaluate(self, state: ParticleState) -> dict[str, np.ndarray]:
        """Return ``{name: force_array}`` for every enabled force."""
        return {e.name: np.asarray(e.fn(state), dtype=float)
                for e in self._entries.values() if e.enabled}

    def total(self, state: ParticleState) -> np.ndarray:
        """Sum of all enabled forces, shape ``(n, dim)`` [N]."""
        out = np.zeros_like(state.x)
        for force in self.evaluate(state).values():
            out = out + force
        return out

    def describe(self) -> pd.DataFrame:
        """Tabular summary for reports and dashboards."""
        return pd.DataFrame(
            [
                {"force": e.name, "enabled": e.enabled, "description": e.description}
                for e in self._entries.values()
            ]
        )

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ForceRegistry active={self.active}>"


@dataclass
class TrackResult:
    """Output of :meth:`LagrangianTracker.run`.

    Attributes
    ----------
    trajectories:
        ``xarray.Dataset`` with ``x(particle, time, axis)`` in metres.
    final:
        ``pandas.DataFrame``, one row per particle, with final position,
        radius, label and the reason the trajectory stopped.
    """

    trajectories: xr.Dataset
    final: pd.DataFrame
    solver_info: dict[str, Any] = field(default_factory=dict)


class LagrangianTracker:
    """Integrate particle trajectories under a :class:`ForceRegistry`.

    Parameters
    ----------
    registry:
        Force models to apply.
    fluid_velocity:
        ``f(t, x) -> (n, dim)`` array [m/s] giving the undisturbed fluid
        velocity at each particle position.  Return zeros for a quiescent fluid.
    viscosity:
        Dynamic viscosity [Pa*s], used by the overdamped mobility.
    mode:
        ``"overdamped"`` (default) or ``"inertial"``.
    bounds:
        Optional ``((lo_0, hi_0), (lo_1, hi_1), ...)`` per axis [m]; ``None`` for
        an unbounded axis.  The position used to evaluate forces and the fluid
        velocity is clamped into the domain, and the sampled trajectories are
        clamped again on output.

        Clamping the *evaluation point* rather than the state keeps the
        right-hand side continuous, which matters: zeroing the outward velocity
        at the wall would make the RHS discontinuous and stall the adaptive
        integrator. It also keeps field data (the FEM Gor'kov force, defined
        only inside the channel) from being extrapolated, which diverges.
    """

    def __init__(
        self,
        registry: ForceRegistry,
        fluid_velocity: Callable[[float, np.ndarray], np.ndarray],
        viscosity: float,
        *,
        mode: Literal["overdamped", "inertial"] = "overdamped",
        bounds: tuple[tuple[float, float] | None, ...] | None = None,
    ) -> None:
        if viscosity <= 0:
            raise ValueError("viscosity must be positive")
        if mode not in ("overdamped", "inertial"):
            raise ValueError("mode must be 'overdamped' or 'inertial'")
        self.registry = registry
        self.fluid_velocity = fluid_velocity
        self.viscosity = float(viscosity)
        self.mode = mode
        self.bounds = bounds

    # -- diagnostics ------------------------------------------------------
    def relaxation_time(self, state: ParticleState) -> np.ndarray:
        """Particle velocity relaxation time ``tau_p = 2 rho_p r^2 / (9 mu)`` [s]."""
        return 2.0 * state.density * state.radius**2 / (9.0 * self.viscosity)

    def check_regime(self, state: ParticleState, transit_time: float) -> dict[str, float]:
        """Warn if the overdamped assumption or Stokes' law is being stretched.

        Returns the worst-case Stokes number and particle Reynolds number.
        """
        tau = self.relaxation_time(state)
        stokes = float(np.max(tau) / max(transit_time, 1e-30))
        u = np.linalg.norm(self.fluid_velocity(0.0, state.x), axis=-1)
        rho_f = float(state.extra.get("fluid_density", np.array([997.0]))[0])
        re_p = float(np.max(rho_f * u * 2.0 * state.radius / self.viscosity))
        if stokes > 0.1:
            warnings.warn(
                f"Stokes number {stokes:.3g} > 0.1: particle inertia is not negligible, "
                "the overdamped model under-predicts lag. Use mode='inertial'.",
                RegimeWarning,
                stacklevel=2,
            )
        if re_p > 1.0:
            warnings.warn(
                f"particle Reynolds number {re_p:.3g} > 1: Stokes drag is no longer valid, "
                "a Schiller-Naumann or Oseen correction is required.",
                RegimeWarning,
                stacklevel=2,
            )
        return {"stokes_number": stokes, "particle_reynolds": re_p}

    # -- integration ------------------------------------------------------
    def run(
        self,
        state: ParticleState,
        t_span: tuple[float, float],
        *,
        n_samples: int = 101,
        t_eval: np.ndarray | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-11,
        method: str = "LSODA",
        check_regime: bool = True,
    ) -> TrackResult:
        """Integrate from ``t_span[0]`` to ``t_span[1]``.

        The whole ensemble is one flattened ODE system, so SciPy sees a single
        vectorised right-hand side rather than ``n`` separate integrations.
        """
        t0, t1 = float(t_span[0]), float(t_span[1])
        if t1 <= t0:
            raise ValueError("t_span must be increasing")
        if check_regime:
            self.check_regime(state, t1 - t0)

        n, dim = state.n, state.dim
        if t_eval is None:
            t_eval = np.linspace(t0, t1, int(n_samples))
        else:
            t_eval = np.clip(np.unique(np.asarray(t_eval, dtype=float)), t0, t1)

        if self.mode == "overdamped":
            y0 = state.x.reshape(-1)

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                x = self._clip(y.reshape(n, dim))
                st = _with(state, t, x, np.zeros_like(x))
                u = np.asarray(self.fluid_velocity(t, x), dtype=float)
                mobility = 1.0 / (6.0 * np.pi * self.viscosity * state.radius)[:, None]
                return (u + self.registry.total(st) * mobility).reshape(-1)

        else:
            y0 = np.concatenate([state.x.reshape(-1), state.v.reshape(-1)])
            mass = state.mass[:, None]

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                x = self._clip(y[: n * dim].reshape(n, dim))
                v = y[n * dim :].reshape(n, dim)
                st = _with(state, t, x, v)
                u = np.asarray(self.fluid_velocity(t, x), dtype=float)
                drag = 6.0 * np.pi * self.viscosity * state.radius[:, None] * (u - v)
                accel = (self.registry.total(st) + drag) / mass
                return np.concatenate([v.reshape(-1), accel.reshape(-1)])

        sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=rtol, atol=atol, method=method)
        if not sol.success:  # pragma: no cover - integrator failure path
            raise RuntimeError(f"trajectory integration failed: {sol.message}")

        xs = sol.y[: n * dim].reshape(n, dim, -1).transpose(0, 2, 1)  # (n, t, dim)
        xs = self._apply_bounds(xs)
        vs = None
        if self.mode == "inertial":
            vs = sol.y[n * dim :].reshape(n, dim, -1).transpose(0, 2, 1)

        return self._package(state, sol.t, xs, vs, sol)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Clamp positions into the domain before any field is evaluated."""
        if self.bounds is None:
            return x
        out = x.copy()
        for axis, limits in enumerate(self.bounds):
            if limits is None or axis >= out.shape[1]:
                continue
            lo, hi = limits
            if hi > lo:
                np.clip(out[:, axis], lo, hi, out=out[:, axis])
        return out

    def _apply_bounds(self, xs: np.ndarray) -> np.ndarray:
        """Clamp the sampled trajectories into the domain.

        In a working device the acoustic force near a wall points inwards, so
        this only removes small integrator overshoot; it also guarantees the
        particle-number conservation the mass-balance test checks.
        """
        if self.bounds is None:
            return xs
        out = xs.copy()
        for axis, limits in enumerate(self.bounds):
            if limits is None or axis >= out.shape[2]:
                continue
            lo, hi = limits
            if hi > lo:
                np.clip(out[:, :, axis], lo, hi, out=out[:, :, axis])
        return out

    def _package(
        self,
        state: ParticleState,
        t: np.ndarray,
        xs: np.ndarray,
        vs: np.ndarray | None,
        sol: Any,
    ) -> TrackResult:
        axes = ["x", "y", "z"][: state.dim]
        data = {"position": (("particle", "time", "axis"), xs)}
        if vs is not None:
            data["velocity"] = (("particle", "time", "axis"), vs)
        ds = xr.Dataset(
            data,
            coords={
                "particle": np.arange(state.n),
                "time": t,
                "axis": axes,
                "radius": ("particle", state.radius),
                "label": ("particle", state.label.astype(str)),
                "density": ("particle", state.density),
            },
        )
        ds["position"].attrs["units"] = "m"
        if vs is not None:
            ds["velocity"].attrs["units"] = "m/s"
        ds["time"].attrs["units"] = "s"

        final = pd.DataFrame(
            {
                "particle": np.arange(state.n),
                "label": state.label.astype(str),
                "radius_m": state.radius,
                "density_kg_m3": state.density,
                "compressibility_1_Pa": state.compressibility,
                **{f"{ax}_final_m": xs[:, -1, i] for i, ax in enumerate(axes)},
                **{f"{ax}_initial_m": xs[:, 0, i] for i, ax in enumerate(axes)},
            }
        )
        info = {
            "mode": self.mode,
            "n_rhs_evaluations": int(sol.nfev),
            "n_steps": int(len(sol.t)),
            "active_forces": self.registry.active,
        }
        return TrackResult(trajectories=ds, final=final, solver_info=info)


def _with(state: ParticleState, t: float, x: np.ndarray, v: np.ndarray) -> ParticleState:
    """Cheap copy of *state* at a new time/position (no array duplication)."""
    return ParticleState(
        t=t,
        x=x,
        v=v,
        radius=state.radius,
        density=state.density,
        compressibility=state.compressibility,
        label=state.label,
        extra=state.extra,
    )


def make_state(
    positions: np.ndarray,
    radii: np.ndarray,
    densities: np.ndarray,
    compressibilities: np.ndarray,
    labels: np.ndarray | list[str],
    *,
    velocities: np.ndarray | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> ParticleState:
    """Build a :class:`ParticleState`, broadcasting scalars to the ensemble size."""
    x = np.atleast_2d(np.asarray(positions, dtype=float))
    n = x.shape[0]

    def _b(a: Any) -> np.ndarray:
        return np.broadcast_to(np.asarray(a, dtype=float), (n,)).copy()

    return ParticleState(
        t=0.0,
        x=x,
        v=np.zeros_like(x) if velocities is None else np.asarray(velocities, dtype=float),
        radius=_b(radii),
        density=_b(densities),
        compressibility=_b(compressibilities),
        label=np.asarray(labels, dtype=object) if not np.isscalar(labels)
        else np.array([labels] * n, dtype=object),
        extra=extra or {},
    )


__all__ = [
    "ParticleState",
    "ForceModel",
    "ForceRegistry",
    "LagrangianTracker",
    "TrackResult",
    "make_state",
]
