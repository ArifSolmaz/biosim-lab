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

Integrators
-----------
Two integrators, selected with ``integrator=``:

* ``"solve_ivp"`` (default) --- SciPy's adaptive LSODA. Accurate and cheap for
  smooth forcing.
* ``"rk4"`` --- classical fourth-order Runge--Kutta at a fixed step ``dt``.
  This is the method the time-switched acoustofluidic literature uses (Zhang
  et al. 2023, doi:10.3390/ijms24043338, Sec. 4.2), and it is the right tool
  when the force is *piecewise* in time: an adaptive integrator either steps
  straight over a switch or grinds its step size down to resolve it, whereas
  a fixed grid that is made to land exactly on every switching instant has no
  discontinuity inside any step at all.

Both accept ``breakpoints``: instants at which the right-hand side jumps. The
adaptive path restarts the integration at each one; the RK4 path shortens the
step so that it lands on each one. ``tests/test_particles.py`` checks the two
against each other and RK4 against a closed-form trajectory.

Reference for the drag law and its validity range:
Batchelor, *An Introduction to Fluid Dynamics*, doi:10.1017/CBO9780511800955,
section 4.9 (Stokes' law, valid for particle Reynolds number Re_p << 1).
Classical RK4: Butcher, *Numerical Methods for Ordinary Differential
Equations*, 3rd ed., doi:10.1002/9781119121534, section 23.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
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
    wall_clearance:
        When ``True``, each particle's *centre* is held at least one radius
        inside every bound: a sphere touches the wall at ``x = r``, not at
        ``x = 0``. This matters wherever a force presses particles into a wall.
        With a no-slip wall the fluid velocity at the wall plane itself is
        zero, so a centre clamped onto it is stopped dead and never reaches
        the outlet --- a numerical artefact that a tilted-SSAW device, which
        drives cells across the full channel width, runs straight into.
    """

    def __init__(
        self,
        registry: ForceRegistry,
        fluid_velocity: Callable[[float, np.ndarray], np.ndarray],
        viscosity: float,
        *,
        mode: Literal["overdamped", "inertial"] = "overdamped",
        bounds: tuple[tuple[float, float] | None, ...] | None = None,
        wall_clearance: bool = False,
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
        self.wall_clearance = bool(wall_clearance)

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
        integrator: Literal["solve_ivp", "rk4"] = "solve_ivp",
        dt: float | None = None,
        breakpoints: Sequence[float] | None = None,
        finished: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> TrackResult:
        """Integrate from ``t_span[0]`` to ``t_span[1]``.

        The whole ensemble is one flattened ODE system, so the integrator sees a
        single vectorised right-hand side rather than ``n`` separate
        integrations.

        Parameters
        ----------
        integrator:
            ``"solve_ivp"`` (adaptive, default) or ``"rk4"`` (fixed step).
        dt:
            RK4 step [s]. Required for ``integrator="rk4"``; ignored otherwise.
        breakpoints:
            Times at which the forcing is discontinuous (e.g. a frequency
            switch). Neither integrator will take a step across one.
        finished:
            RK4 + overdamped only. ``f(positions) -> bool per particle``; a
            particle for which it returns ``True`` at a sample time is frozen
            there and dropped from the integration, and the run ends (and the
            output is truncated) once every particle is finished. For a flow
            device, "past the outlet plane" --- the trajectory beyond it cannot
            change any outlet statistic but would otherwise cost as much as the
            part that does. The frozen tail is reported in ``solver_info``.
        """
        t0, t1 = float(t_span[0]), float(t_span[1])
        if t1 <= t0:
            raise ValueError("t_span must be increasing")
        if integrator not in ("solve_ivp", "rk4"):
            raise ValueError("integrator must be 'solve_ivp' or 'rk4'")
        if integrator == "rk4" and (dt is None or dt <= 0):
            raise ValueError("integrator='rk4' needs a positive step dt")
        if check_regime:
            self.check_regime(state, t1 - t0)

        n, dim = state.n, state.dim
        if t_eval is None:
            t_eval = np.linspace(t0, t1, int(n_samples))
        else:
            t_eval = np.clip(np.unique(np.asarray(t_eval, dtype=float)), t0, t1)
        # Every discontinuity strictly inside the window becomes a hard stop.
        bp = np.asarray([] if breakpoints is None else breakpoints, dtype=float).ravel()
        stops = np.unique(np.concatenate([[t0, t1], bp[(bp > t0) & (bp < t1)]]))

        rhs, y0 = self._rhs(state)

        if integrator == "rk4" and finished is not None and self.mode == "overdamped":
            sol: Any = self._rk4_pruned(state, t_eval, stops, float(dt), finished)  # type: ignore[arg-type]
        elif integrator == "rk4":
            ts, ys, nfev = _rk4(rhs, y0, t_eval, stops, float(dt))  # type: ignore[arg-type]
            sol = _Solution(t=ts, y=ys, nfev=nfev)
        else:
            sol = self._solve_piecewise(rhs, y0, t_eval, stops, rtol, atol, method)

        xs = sol.y[: n * dim].reshape(n, dim, -1).transpose(0, 2, 1)  # (n, t, dim)
        xs = self._apply_bounds(xs, state.radius)
        vs = None
        if self.mode == "inertial":
            vs = sol.y[n * dim :].reshape(n, dim, -1).transpose(0, 2, 1)

        result = self._package(state, sol.t, xs, vs, sol)
        result.solver_info["integrator"] = integrator
        if integrator == "rk4":
            result.solver_info["dt"] = float(dt)  # type: ignore[arg-type]
        result.solver_info["n_breakpoints"] = int(len(stops) - 2)
        if getattr(sol, "finished_at", None) is not None:
            result.solver_info["finished_at_s"] = sol.finished_at
        return result

    def _rk4_pruned(
        self,
        state: ParticleState,
        t_eval: np.ndarray,
        stops: np.ndarray,
        dt: float,
        finished: Callable[[np.ndarray], np.ndarray],
    ) -> _Solution:
        """:func:`_rk4` over a shrinking active set (overdamped only).

        Identical steps to :func:`_rk4` for every particle that is still
        active --- the right-hand side is per particle, so removing one changes
        nothing for the others. Checked at sample times only.
        """
        n, dim = state.n, state.dim
        marks = np.unique(np.concatenate([t_eval, stops]))
        is_sample = np.isin(marks, t_eval)  # exact: see _rk4
        x_all = state.x.copy()
        active = np.arange(n)
        rhs, y = self._rhs(_subset(state, active))
        out_t: list[float] = []
        out_y: list[np.ndarray] = []
        finished_at = np.full(n, np.nan)
        if is_sample[0]:
            out_t.append(float(marks[0]))
            out_y.append(x_all.reshape(-1).copy())
        nfev = 0
        for k, (a, b) in enumerate(zip(marks[:-1], marks[1:]), start=1):
            span = float(b - a)
            n_steps = max(1, int(np.ceil(span / dt - 1e-9)))
            h = span / n_steps
            t = float(a)
            f = _inside(rhs, float(a), float(b))
            for _ in range(n_steps):
                k1 = f(t, y)
                k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
                k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
                k4 = f(t + h, y + h * k3)
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += h
                nfev += 4
            x_all[active] = y.reshape(active.size, dim)
            if not is_sample[k]:
                continue
            out_t.append(float(b))
            out_y.append(x_all.reshape(-1).copy())
            done = np.asarray(finished(x_all[active]), dtype=bool)
            if done.any():
                finished_at[active[done]] = float(b)
                active = active[~done]
                if active.size == 0:
                    break
                rhs, y = self._rhs(_subset(state, active))
                y = x_all[active].reshape(-1).copy()
        sol = _Solution(t=np.asarray(out_t), y=np.column_stack(out_y), nfev=nfev)
        sol.finished_at = finished_at
        return sol

    def _rhs(self, state: ParticleState) -> tuple[Callable[[float, np.ndarray], np.ndarray],
                                                  np.ndarray]:
        """The flattened right-hand side and initial vector for *state*."""
        n, dim = state.n, state.dim
        if self.mode == "overdamped":
            y0 = state.x.reshape(-1).copy()
            mobility = 1.0 / (6.0 * np.pi * self.viscosity * state.radius)[:, None]

            def rhs(t: float, y: np.ndarray) -> np.ndarray:
                x = self._clip(y.reshape(n, dim), state.radius)
                st = _with(state, t, x, np.zeros_like(x))
                u = np.asarray(self.fluid_velocity(t, x), dtype=float)
                return (u + self.registry.total(st) * mobility).reshape(-1)

            return rhs, y0

        y0 = np.concatenate([state.x.reshape(-1), state.v.reshape(-1)])
        mass = state.mass[:, None]

        def rhs_inertial(t: float, y: np.ndarray) -> np.ndarray:
            x = self._clip(y[: n * dim].reshape(n, dim), state.radius)
            v = y[n * dim :].reshape(n, dim)
            st = _with(state, t, x, v)
            u = np.asarray(self.fluid_velocity(t, x), dtype=float)
            drag = 6.0 * np.pi * self.viscosity * state.radius[:, None] * (u - v)
            accel = (self.registry.total(st) + drag) / mass
            return np.concatenate([v.reshape(-1), accel.reshape(-1)])

        return rhs_inertial, y0

    @staticmethod
    def _solve_piecewise(
        rhs: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_eval: np.ndarray,
        stops: np.ndarray,
        rtol: float,
        atol: float,
        method: str,
    ) -> Any:
        """``solve_ivp`` restarted at every breakpoint.

        With no breakpoints this is one call and behaves exactly as before.
        Segment ``k`` reports the samples in ``(a, b]`` (``[a, b]`` for the
        first), so a sample sitting on a breakpoint is reported once, as the
        end of the segment that arrives there.
        """
        ts: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        nfev = 0
        y = y0
        for k, (a, b) in enumerate(zip(stops[:-1], stops[1:])):
            lower = (t_eval >= a) if k == 0 else (t_eval > a)
            inside = t_eval[lower & (t_eval <= b)]
            # Always ask for the segment end so the next segment starts from it.
            want = np.unique(np.concatenate([inside, [a, b]]))
            seg_rhs = _inside(rhs, float(a), float(b)) if len(stops) > 2 else rhs
            sol = solve_ivp(seg_rhs, (a, b), y, t_eval=want, rtol=rtol, atol=atol,
                            method=method)
            if not sol.success:  # pragma: no cover - integrator failure path
                raise RuntimeError(f"trajectory integration failed: {sol.message}")
            nfev += int(sol.nfev)
            y = sol.y[:, -1]
            keep = np.isin(sol.t, inside)
            ts.append(sol.t[keep])
            ys.append(sol.y[:, keep])
        t_all = np.concatenate(ts)
        order = np.argsort(t_all, kind="stable")
        return _Solution(t=t_all[order], y=np.concatenate(ys, axis=1)[:, order], nfev=nfev)

    def force_history(self, tracks: TrackResult, state: ParticleState) -> xr.Dataset:
        """Every registered force, and the Stokes drag, along the sampled trajectories.

        Returns ``F(force, particle, time, axis)`` [N] with one entry per enabled
        force plus ``"stokes_drag"`` --- the curves a time-switched device is
        explained with (Zhang et al. 2023, doi:10.3390/ijms24043338, Fig. 8c).

        Drag is ``6 pi mu a (u_f - v_p)`` (Batchelor, doi:10.1017/CBO9780511800955,
        section 4.9). In the overdamped limit the particle velocity *is* the
        force balance, so the drag comes out as exactly minus the sum of the
        other forces; in inertial mode it uses the integrated velocity.
        """
        pos = tracks.trajectories["position"].values  # (n, t, dim)
        times = tracks.trajectories["time"].values
        n, n_t, dim = pos.shape
        names = self.registry.active
        out = np.zeros((len(names) + 1, n, n_t, dim))
        vel = (tracks.trajectories["velocity"].values
               if "velocity" in tracks.trajectories else None)
        for j, t in enumerate(times):
            x = self._clip(pos[:, j, :], state.radius)
            v = vel[:, j, :] if vel is not None else np.zeros_like(x)
            st = _with(state, float(t), x, v)
            forces = self.registry.evaluate(st)
            for i, name in enumerate(names):
                out[i, :, j, :] = forces[name]
            if vel is None:
                out[-1, :, j, :] = -sum(forces.values()) if forces else 0.0
            else:
                u = np.asarray(self.fluid_velocity(float(t), x), dtype=float)
                out[-1, :, j, :] = (
                    6.0 * np.pi * self.viscosity * state.radius[:, None] * (u - v)
                )
        ds = xr.Dataset(
            {"F": (("force", "particle", "time", "axis"), out)},
            coords={
                "force": [*names, "stokes_drag"],
                "particle": np.arange(n),
                "time": times,
                "axis": tracks.trajectories["axis"].values,
                "label": ("particle", tracks.trajectories["label"].values),
            },
        )
        ds["F"].attrs["units"] = "N"
        return ds

    def _limits(self, lo: float, hi: float, radius: np.ndarray | None
                ) -> tuple[Any, Any]:
        """Per-particle ``(lo, hi)`` for one axis, honouring ``wall_clearance``."""
        if not self.wall_clearance or radius is None:
            return lo, hi
        r = np.asarray(radius, dtype=float)
        a = lo + r
        b = np.maximum(hi - r, a)  # a particle wider than the gap sits in the middle
        return a, b

    def _clip(self, x: np.ndarray, radius: np.ndarray | None = None) -> np.ndarray:
        """Clamp positions into the domain before any field is evaluated."""
        if self.bounds is None:
            return x
        out = x.copy()
        for axis, limits in enumerate(self.bounds):
            if limits is None or axis >= out.shape[1]:
                continue
            lo, hi = limits
            if hi > lo:
                a, b = self._limits(lo, hi, radius)
                out[:, axis] = np.clip(out[:, axis], a, b)
        return out

    def _apply_bounds(self, xs: np.ndarray, radius: np.ndarray | None = None) -> np.ndarray:
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
                a, b = self._limits(lo, hi, radius)
                if np.ndim(a):
                    a, b = np.asarray(a)[:, None], np.asarray(b)[:, None]
                out[:, :, axis] = np.clip(out[:, :, axis], a, b)
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


def _inside(
    rhs: Callable[[float, np.ndarray], np.ndarray], a: float, b: float
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Evaluate *rhs* with time held a hair inside ``[a, b]``.

    A time-switched force is ambiguous exactly at a switch: ``t = b`` belongs to
    the next phase by the modulo arithmetic, but it is also the end of the step
    that is integrating the current one. Offsetting the stage time by
    ``1e-9 (b - a)`` resolves that toward the segment being integrated. For a
    force that is smooth in time the shift is far below any tolerance in use.
    """
    delta = 1e-9 * (b - a)
    lo, hi = a + delta, b - delta

    def wrapped(t: float, y: np.ndarray) -> np.ndarray:
        return rhs(min(max(t, lo), hi), y)

    return wrapped


@dataclass
class _Solution:
    """The subset of SciPy's ``OdeResult`` that :meth:`LagrangianTracker._package` reads."""

    t: np.ndarray
    y: np.ndarray
    nfev: int
    finished_at: np.ndarray | None = None


def _subset(state: ParticleState, idx: np.ndarray) -> ParticleState:
    """The particles *idx* of *state*; per-particle ``extra`` arrays are sliced too."""
    n = state.n
    extra = {
        k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
        for k, v in state.extra.items()
    }
    return ParticleState(
        t=state.t, x=state.x[idx], v=state.v[idx], radius=state.radius[idx],
        density=state.density[idx], compressibility=state.compressibility[idx],
        label=state.label[idx], extra=extra,
    )


def _rk4(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_eval: np.ndarray,
    stops: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Classical fourth-order Runge--Kutta on a grid that lands on every stop.

    ``k1 = f(t, y)``, ``k2 = f(t + h/2, y + h k1/2)``, ``k3 = f(t + h/2, y + h k2/2)``,
    ``k4 = f(t + h, y + h k3)``, ``y += h (k1 + 2 k2 + 2 k3 + k4) / 6``
    (Butcher, doi:10.1002/9781119121534, section 23).

    The union of sample times and breakpoints partitions the window; each piece
    is covered by ``ceil(length / dt)`` equal steps, so no step straddles a
    discontinuity and every requested sample is hit exactly rather than
    interpolated. The step is never *longer* than ``dt``.

    Stage times are held strictly inside the piece being stepped (see
    :func:`_inside`), so a forcing that jumps at a breakpoint is always read on
    the correct side of the jump.
    """
    marks = np.unique(np.concatenate([t_eval, stops]))
    # Exact membership: every sample time is copied into `marks` unchanged, so
    # equality is reliable --- rounding is not (np.round and round() disagree).
    is_sample = np.isin(marks, t_eval)
    y = np.asarray(y0, dtype=float).copy()
    out_t: list[float] = []
    out_y: list[np.ndarray] = []
    if is_sample[0]:
        out_t.append(float(marks[0]))
        out_y.append(y.copy())
    nfev = 0
    for k, (a, b) in enumerate(zip(marks[:-1], marks[1:]), start=1):
        span = float(b - a)
        n_steps = max(1, int(np.ceil(span / dt - 1e-9)))
        h = span / n_steps
        t = float(a)
        f = _inside(rhs, float(a), float(b))
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            nfev += 4
        if is_sample[k]:
            out_t.append(float(b))
            out_y.append(y.copy())
    return np.asarray(out_t), np.column_stack(out_y), nfev


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
