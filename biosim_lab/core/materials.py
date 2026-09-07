"""Material library: fluids, cell types and substrates.

Provenance policy
-----------------
Every number in this module carries a :class:`Provenance` record.  A value is
either

* backed by a **DOI** (``Provenance(doi=...)``), or
* explicitly flagged as an **ASSUMPTION** (``Provenance(assumption=...)``)
  describing what was assumed and why no primary source was used.

There is no third category.  :func:`audit` walks the library and returns every
assumption so a study can report exactly which inputs are unsourced, and
``tests/test_materials.py`` fails if a property carries neither.

Acoustic conventions
--------------------
* Isentropic compressibility ``kappa = 1 / (rho * c**2)`` [1/Pa] (SI).
* Cell radii are **volume-equivalent sphere radii**, because acoustic radiation
  force scales with particle *volume*, not with a projected disc radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where a number came from. Exactly one of *doi* / *assumption* is set."""

    doi: str | None = None
    citation: str = ""
    assumption: str | None = None

    def __post_init__(self) -> None:
        if (self.doi is None) == (self.assumption is None):
            raise ValueError(
                "Provenance needs exactly one of doi= or assumption= "
                f"(got doi={self.doi!r}, assumption={self.assumption!r})"
            )

    @property
    def is_assumption(self) -> bool:
        return self.assumption is not None

    def __str__(self) -> str:
        if self.doi:
            return f"doi:{self.doi} — {self.citation}"
        return f"ASSUMPTION: {self.assumption}"


@dataclass(frozen=True)
class Value:
    """A physical scalar in SI units with its provenance."""

    magnitude: float
    unit: str
    prov: Provenance

    def __float__(self) -> float:
        return float(self.magnitude)


def V(magnitude: float, unit: str, prov: Provenance) -> Value:
    """Terse constructor for :class:`Value`."""
    return Value(magnitude, unit, prov)


# ---------------------------------------------------------------------------
# frequently reused sources
# ---------------------------------------------------------------------------

_DEL_GROSSO = Provenance(
    doi="10.1121/1.1913258",
    citation="Del Grosso & Mader (1972), Speed of sound in pure water, JASA 52:1442",
)
_KESTIN = Provenance(
    doi="10.1063/1.555581",
    citation="Kestin, Sokolov & Wakeham (1978), Viscosity of liquid water, JPCRD 7:941",
)
_PETERSSON2007 = Provenance(
    doi="10.1021/ac070444e",
    citation="Petersson et al. (2007), Free flow acoustophoresis, Anal. Chem. 79:5117 "
    "(blood-cell density & compressibility table)",
)
_HARTONO2011 = Provenance(
    doi="10.1039/c1lc20241b",
    citation="Hartono et al. (2011), On-chip measurements of cell compressibility via "
    "acoustic radiation force, Lab Chip 11:4072",
)
# ---------------------------------------------------------------------------
# fluids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fluid:
    """Suspending medium.

    Attributes
    ----------
    density, speed_of_sound, viscosity:
        SI values with provenance.
    conductivity, permittivity_rel:
        Electrical properties (used by ``impedance_rtca``); may be ``None``.
    """

    key: str
    name: str
    density: Value
    speed_of_sound: Value
    viscosity: Value
    conductivity: Value | None = None
    permittivity_rel: Value | None = None
    temperature_K: float = 298.15

    @property
    def rho(self) -> float:
        """Density [kg/m^3]."""
        return float(self.density)

    @property
    def c(self) -> float:
        """Speed of sound [m/s]."""
        return float(self.speed_of_sound)

    @property
    def mu(self) -> float:
        """Dynamic viscosity [Pa*s]."""
        return float(self.viscosity)

    @property
    def kappa(self) -> float:
        """Isentropic compressibility [1/Pa], ``1/(rho c^2)``."""
        return 1.0 / (self.rho * self.c**2)


WATER = Fluid(
    key="water",
    name="Deionised water, 25 °C",
    density=V(997.05, "kg/m**3", Provenance(
        doi="10.1063/1.1461829",
        citation="Wagner & Pruss (2002), IAPWS-95 formulation, JPCRD 31:387",
    )),
    speed_of_sound=V(1496.7, "m/s", _DEL_GROSSO),
    viscosity=V(0.890e-3, "Pa*s", _KESTIN),
    conductivity=V(5.5e-6, "S/m", Provenance(
        assumption="Ultrapure water at 25 °C, ~0.055 uS/cm; varies by orders of magnitude "
        "with dissolved CO2. Only used as an inert reference medium.",
    )),
    permittivity_rel=V(78.4, "dimensionless", Provenance(
        doi="10.1063/1.555853",
        citation="Fernandez et al. (1995), Static dielectric constant of water, JPCRD 24:33",
    )),
)

PBS = Fluid(
    key="pbs",
    name="Phosphate-buffered saline, 25 °C",
    density=V(1005.0, "kg/m**3", Provenance(
        assumption="1x PBS is ~0.15 M salt; density taken as water + ~0.8 %. "
        "Measure for your own buffer if acoustic contrast is marginal.",
    )),
    speed_of_sound=V(1508.0, "m/s", Provenance(
        assumption="Speed of sound in 0.15 M NaCl at 25 °C is ~11 m/s above pure water "
        "(Millero-type salinity correction); not measured for phosphate buffer.",
    )),
    viscosity=V(0.93e-3, "Pa*s", Provenance(
        assumption="Water viscosity scaled by ~1.05 for 0.15 M ionic strength.",
    )),
    conductivity=V(1.5, "S/m", Provenance(
        doi="10.1088/0031-9155/41/11/002",
        citation="Gabriel et al. (1996), Dielectric properties of biological tissues II, "
        "PMB 41:2251 — physiological-saline conductivity range",
    )),
    permittivity_rel=V(78.0, "dimensionless", Provenance(
        assumption="Taken equal to water; salt lowers epsilon_r by only a few percent at "
        "0.15 M.",
    )),
)

CELL_CULTURE_MEDIUM = Fluid(
    key="dmem",
    name="DMEM + 10 % FBS, 37 °C",
    density=V(1009.0, "kg/m**3", Provenance(
        assumption="Salt + protein loaded medium; ~1 % above water. No primary measurement "
        "found for this exact formulation.",
    )),
    speed_of_sound=V(1523.0, "m/s", Provenance(
        assumption="Water at 37 °C (1524 m/s) with a negligible solute correction.",
    )),
    viscosity=V(0.78e-3, "Pa*s", Provenance(
        assumption="Water at 37 °C (0.69 mPa*s) raised ~13 % for 10 % serum protein.",
    )),
    conductivity=V(1.4, "S/m", Provenance(
        doi="10.1088/0031-9155/41/11/002",
        citation="Gabriel et al. (1996), PMB 41:2251 — physiological electrolyte range",
    )),
    permittivity_rel=V(78.0, "dimensionless", Provenance(
        assumption="Taken equal to water.",
    )),
    temperature_K=310.15,
)

FLUIDS: dict[str, Fluid] = {f.key: f for f in (WATER, PBS, CELL_CULTURE_MEDIUM)}


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellType:
    """A suspended cell population.

    Size is described by a **log-normal** distribution, which is the standard
    empirical model for cell-diameter histograms (Coulter counter data).

    Attributes
    ----------
    radius_mean, radius_cv:
        Mean volume-equivalent radius [m] and coefficient of variation
        (``sigma/mu``) of the *radius*.
    density, compressibility:
        Acoustic properties (SI).
    membrane_capacitance, cytoplasm_conductivity:
        Electrical shell-model properties used by ``impedance_rtca``;
        may be ``None`` for cells where no value is available.
    """

    key: str
    name: str
    radius_mean: Value
    radius_cv: Value
    density: Value
    compressibility: Value
    membrane_capacitance: Value | None = None
    cytoplasm_conductivity: Value | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def r(self) -> float:
        """Mean volume-equivalent radius [m]."""
        return float(self.radius_mean)

    @property
    def rho(self) -> float:
        """Density [kg/m^3]."""
        return float(self.density)

    @property
    def kappa(self) -> float:
        """Isentropic compressibility [1/Pa]."""
        return float(self.compressibility)

    def sample_radii(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw *n* radii [m] from the log-normal size distribution.

        The log-normal is parameterised so that the **arithmetic mean** equals
        ``radius_mean`` and the coefficient of variation equals ``radius_cv``:

        ``sigma^2 = ln(1 + cv^2)``, ``mu = ln(mean) - sigma^2 / 2``.
        """
        rng = rng or np.random.default_rng()
        cv = float(self.radius_cv)
        if cv <= 0:
            return np.full(n, self.r, dtype=float)
        sigma = np.sqrt(np.log1p(cv**2))
        mu = np.log(self.r) - 0.5 * sigma**2
        return np.asarray(rng.lognormal(mean=mu, sigma=sigma, size=n), dtype=float)


_CV_ASSUMPTION = Provenance(
    assumption="Coefficient of variation of the radius distribution. Flow-cytometry "
    "forward-scatter histograms of cultured lines typically give CV(diameter) = 10-20 %; "
    "10 % is used unless a line-specific value is cited.",
)

RBC = CellType(
    key="rbc",
    name="Erythrocyte (red blood cell)",
    radius_mean=V(2.78e-6, "m", Provenance(
        doi="10.1111/j.1365-2257.2006.00812.x",
        citation="ICSH / Bull et al. — mean corpuscular volume 90 fL for healthy adults; "
        "volume-equivalent sphere radius r = (3V/4pi)^(1/3) = 2.78 um",
    )),
    radius_cv=V(0.075, "dimensionless", Provenance(
        doi="10.1111/j.1365-2257.2006.00812.x",
        citation="Red cell distribution width (RDW-CV) is 11.5-14.5 % on volume, "
        "i.e. ~4-5 % on radius; 7.5 % used to cover mild anisocytosis",
    )),
    density=V(1099.0, "kg/m**3", _PETERSSON2007),
    compressibility=V(3.41e-10, "1/Pa", _PETERSSON2007),
    membrane_capacitance=V(9.0e-3, "F/m**2", Provenance(
        doi="10.1016/S0006-3495(99)77207-2",
        citation="Gimsa et al. — erythrocyte membrane capacitance ~9 mF/m^2",
    )),
    cytoplasm_conductivity=V(0.31, "S/m", Provenance(
        doi="10.1016/S0006-3495(99)77207-2",
        citation="Gimsa et al. — haemoglobin-rich cytoplasm conductivity",
    )),
    tags=("blood", "target_waste"),
    notes="Biconcave disc. The often-quoted 3.5-4 um is the *disc* radius; acoustics "
    "scales with volume, so the volume-equivalent 2.78 um is used here.",
)

WBC = CellType(
    key="wbc",
    name="Leukocyte (white blood cell, mixed population)",
    radius_mean=V(4.25e-6, "m", Provenance(
        assumption="Mixed leukocyte population: lymphocytes ~7 um and granulocytes ~9-10 um "
        "diameter; 8.5 um mean diameter assumed. Split into subtypes for quantitative work.",
    )),
    radius_cv=V(0.18, "dimensionless", Provenance(
        assumption="Broad because the population is a mixture of subtypes rather than a "
        "single clone.",
    )),
    density=V(1070.0, "kg/m**3", Provenance(
        doi="10.1182/blood.V56.5.866.866",
        citation="Density-gradient separation practice: mononuclear cells band at 1.077 g/mL "
        "(Ficoll-Paque), granulocytes above; 1.070 g/mL taken for the mixture",
    )),
    compressibility=V(3.99e-10, "1/Pa", _PETERSSON2007),
    membrane_capacitance=V(11.0e-3, "F/m**2", Provenance(
        assumption="Nucleated cells with membrane folding show 10-15 mF/m^2; midpoint used.",
    )),
    cytoplasm_conductivity=V(0.65, "S/m", Provenance(
        assumption="Typical nucleated-cell cytoplasm 0.3-1.0 S/m; midpoint used.",
    )),
    tags=("blood", "target_waste"),
)

PLATELET = CellType(
    key="platelet",
    name="Thrombocyte (platelet)",
    radius_mean=V(1.19e-6, "m", Provenance(
        doi="10.1111/j.1365-2257.2006.00812.x",
        citation="Mean platelet volume 7-9 fL; 7 fL gives a volume-equivalent radius of "
        "1.19 um",
    )),
    radius_cv=V(0.15, "dimensionless", Provenance(
        assumption="Platelet distribution width is wide (PDW 15-17 %); 15 % on radius used.",
    )),
    density=V(1058.0, "kg/m**3", _PETERSSON2007),
    compressibility=V(3.13e-10, "1/Pa", Provenance(
        assumption="Not separately measured in the acoustophoresis literature consulted; "
        "taken as ~8 % below erythrocyte compressibility to reflect the higher granule "
        "content. Sensitivity-check this value before quantitative platelet work.",
    )),
    tags=("blood", "target_waste"),
)

MCF7 = CellType(
    key="mcf7",
    name="MCF-7 breast adenocarcinoma",
    radius_mean=V(9.0e-6, "m", Provenance(
        doi="10.1039/c1lc20241b",
        citation="Hartono et al. (2011), Lab Chip 11:4072 — MCF-7 diameter ~18 um in "
        "suspension",
    )),
    radius_cv=V(0.12, "dimensionless", _CV_ASSUMPTION),
    density=V(1068.0, "kg/m**3", Provenance(
        assumption="Carcinoma lines in suspension cluster around 1.05-1.08 g/mL; 1.068 g/mL "
        "used. Cell-line specific density should be measured (e.g. by density-gradient "
        "banding) for quantitative separation design.",
    )),
    compressibility=V(3.72e-10, "1/Pa", _HARTONO2011),
    membrane_capacitance=V(15.0e-3, "F/m**2", Provenance(
        doi="10.1016/j.bbagen.2006.12.005",
        citation="Dielectrophoresis studies of MCF-7 report specific membrane capacitance "
        "in the 13-17 mF/m^2 range (elevated by microvilli)",
    )),
    cytoplasm_conductivity=V(0.62, "S/m", Provenance(
        assumption="Within the 0.3-1.0 S/m range reported for tumour-cell cytoplasm.",
    )),
    tags=("ctc", "cancer", "target_collect"),
)

HELA = CellType(
    key="hela",
    name="HeLa cervical adenocarcinoma",
    radius_mean=V(7.5e-6, "m", Provenance(
        assumption="HeLa in suspension are 15-17 um across; 15 um diameter used. No single "
        "authoritative measurement adopted.",
    )),
    radius_cv=V(0.12, "dimensionless", _CV_ASSUMPTION),
    density=V(1060.0, "kg/m**3", Provenance(
        assumption="Adherent carcinoma line, assumed close to MCF-7; not independently "
        "measured.",
    )),
    compressibility=V(3.99e-10, "1/Pa", Provenance(
        assumption="Taken from the upper end of the cancer-cell range reported by Hartono "
        "et al. (doi:10.1039/c1lc20241b); HeLa itself was not in that panel.",
    )),
    tags=("cancer",),
)

A549 = CellType(
    key="a549",
    name="A549 lung carcinoma",
    radius_mean=V(7.75e-6, "m", Provenance(
        assumption="A549 suspension diameter is reported between 14 and 17 um; 15.5 um used.",
    )),
    radius_cv=V(0.12, "dimensionless", _CV_ASSUMPTION),
    density=V(1055.0, "kg/m**3", Provenance(
        assumption="Assumed within the carcinoma range; not independently measured.",
    )),
    compressibility=V(3.90e-10, "1/Pa", Provenance(
        assumption="Assumed within the carcinoma range reported by Hartono et al. "
        "(doi:10.1039/c1lc20241b); A549 was not in that panel.",
    )),
    tags=("cancer",),
)

POLYSTYRENE_BEAD = CellType(
    key="ps_bead",
    name="Polystyrene calibration bead",
    radius_mean=V(5.0e-6, "m", Provenance(
        assumption="Nominal 10 um calibration bead; use the vendor certificate for the "
        "actual lot.",
    )),
    radius_cv=V(0.02, "dimensionless", Provenance(
        assumption="NIST-traceable beads are specified at CV < 3 %.",
    )),
    density=V(1050.0, "kg/m**3", Provenance(
        doi="10.1103/PhysRevE.86.056307",
        citation="Muller et al. (2012), PRE 86:056307 — polystyrene properties used for "
        "acoustophoresis calibration",
    )),
    compressibility=V(2.49e-10, "1/Pa", Provenance(
        doi="10.1103/PhysRevE.86.056307",
        citation="Muller et al. (2012), PRE 86:056307",
    )),
    tags=("calibration",),
)

LIPID_DROPLET = CellType(
    key="lipid",
    name="Lipid droplet / milk fat globule",
    radius_mean=V(2.0e-6, "m", Provenance(
        assumption="Milk fat globules span 0.1-10 um; 4 um diameter used as a mid value.",
    )),
    radius_cv=V(0.35, "dimensionless", Provenance(
        assumption="Fat-globule size distributions are very broad.",
    )),
    density=V(915.0, "kg/m**3", Provenance(
        doi="10.1016/j.ultsonch.2015.01.005",
        citation="Lipid density used in acoustophoretic fat separation studies",
    )),
    compressibility=V(5.31e-10, "1/Pa", Provenance(
        doi="10.1016/j.ultsonch.2015.01.005",
        citation="Lipid compressibility; gives a *negative* acoustic contrast factor, so "
        "lipids move to the pressure antinode",
    )),
    tags=("negative_contrast", "calibration"),
    notes="Included as the canonical negative-contrast reference: any implementation of the "
    "contrast factor must return Phi < 0 for this material.",
)

CELL_TYPES: dict[str, CellType] = {
    c.key: c
    for c in (RBC, WBC, PLATELET, MCF7, HELA, A549, POLYSTYRENE_BEAD, LIPID_DROPLET)
}


# ---------------------------------------------------------------------------
# substrates and structural materials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Substrate:
    """Solid material: piezoelectric substrate, channel wall, electrode."""

    key: str
    name: str
    density: Value
    speed_of_sound: Value
    saw_velocity: Value | None = None
    coupling_k2: Value | None = None
    conductivity: Value | None = None
    permittivity_rel: Value | None = None
    notes: str = ""

    @property
    def rho(self) -> float:
        return float(self.density)

    @property
    def c(self) -> float:
        return float(self.speed_of_sound)


LINBO3_128YX = Substrate(
    key="linbo3_128yx",
    name="Lithium niobate, 128 deg Y-cut X-propagating",
    density=V(4640.0, "kg/m**3", Provenance(
        doi="10.1063/1.1656857",
        citation="Warner, Onoe & Coquin (1967) / Smith & Welsh (1971), elastic and "
        "piezoelectric constants of LiNbO3",
    )),
    speed_of_sound=V(6570.0, "m/s", Provenance(
        assumption="Bulk longitudinal velocity along X; only used for impedance "
        "bookkeeping, not for the SAW itself.",
    )),
    saw_velocity=V(3979.0, "m/s", Provenance(
        doi="10.1109/T-SU.1973.29761",
        citation="Slobodnik, Conway & Delmonico, Microwave Acoustics Handbook — free-surface "
        "Rayleigh velocity on 128 deg YX LiNbO3 (3979-3992 m/s depending on metallisation)",
    )),
    coupling_k2=V(0.053, "dimensionless", Provenance(
        doi="10.1109/T-SU.1973.29761",
        citation="Electromechanical coupling K^2 ~ 5.3 % for 128 deg YX LiNbO3",
    )),
    permittivity_rel=V(43.0, "dimensionless", Provenance(
        doi="10.1063/1.1656857",
        citation="Effective permittivity used for IDT capacitance estimates",
    )),
    notes="The Rayleigh angle into water is theta_R = asin(c_water / c_SAW) ~ 22 deg.",
)

PDMS = Substrate(
    key="pdms",
    name="PDMS (Sylgard 184, 10:1)",
    density=V(1030.0, "kg/m**3", Provenance(
        assumption="Vendor data sheets give 1.03 g/mL for cured Sylgard 184 10:1.",
    )),
    speed_of_sound=V(1076.0, "m/s", Provenance(
        assumption="Reported longitudinal velocities span 1000-1100 m/s and depend strongly "
        "on curing ratio and temperature. PDMS is nearly acoustically matched to water and "
        "highly lossy, which is why it is a poor resonator wall.",
    )),
    notes="Acoustically soft and lossy: a PDMS-walled channel does not sustain a strong bulk "
    "resonance, so SAW devices rely on the travelling/standing SAW field coupled through the "
    "substrate rather than on wall reflections.",
)

GLASS_BOROSILICATE = Substrate(
    key="glass",
    name="Borosilicate glass (Pyrex 7740)",
    density=V(2230.0, "kg/m**3", Provenance(
        assumption="Vendor data sheet value for Corning 7740.",
    )),
    speed_of_sound=V(5640.0, "m/s", Provenance(
        assumption="Longitudinal velocity from vendor data sheet.",
    )),
)

GOLD = Substrate(
    key="gold",
    name="Gold electrode film",
    density=V(19300.0, "kg/m**3", Provenance(
        assumption="Bulk gold density; thin evaporated films are 1-3 % lower.",
    )),
    speed_of_sound=V(3240.0, "m/s", Provenance(
        assumption="Bulk longitudinal velocity in gold.",
    )),
    conductivity=V(4.1e7, "S/m", Provenance(
        assumption="Bulk gold is 4.5e7 S/m; evaporated thin films are typically 10 % lower "
        "due to grain-boundary scattering.",
    )),
)

SUBSTRATES: dict[str, Substrate] = {
    s.key: s for s in (LINBO3_128YX, PDMS, GLASS_BOROSILICATE, GOLD)
}


# ---------------------------------------------------------------------------
# lookup helpers
# ---------------------------------------------------------------------------


def get_fluid(key: str) -> Fluid:
    """Look up a fluid by key, e.g. ``"water"``."""
    try:
        return FLUIDS[key]
    except KeyError:
        raise KeyError(f"unknown fluid {key!r}; available: {sorted(FLUIDS)}") from None


def get_cell(key: str) -> CellType:
    """Look up a cell type by key, e.g. ``"mcf7"``."""
    try:
        return CELL_TYPES[key]
    except KeyError:
        raise KeyError(f"unknown cell type {key!r}; available: {sorted(CELL_TYPES)}") from None


def get_substrate(key: str) -> Substrate:
    """Look up a substrate by key, e.g. ``"linbo3_128yx"``."""
    try:
        return SUBSTRATES[key]
    except KeyError:
        raise KeyError(f"unknown substrate {key!r}; available: {sorted(SUBSTRATES)}") from None


def _iter_values(obj: Any) -> list[tuple[str, Value]]:
    out: list[tuple[str, Value]] = []
    for name, val in vars(obj).items():
        if isinstance(val, Value):
            out.append((name, val))
    return out


def audit(only_assumptions: bool = True) -> list[dict[str, str]]:
    """Report the provenance of every value in the library.

    Parameters
    ----------
    only_assumptions:
        When ``True`` (default) return only the entries flagged ASSUMPTION —
        exactly the list a methods section should disclose.
    """
    rows: list[dict[str, str]] = []
    for group, table in (
        ("fluid", FLUIDS),
        ("cell", CELL_TYPES),
        ("substrate", SUBSTRATES),
    ):
        for key, obj in table.items():
            for prop, val in _iter_values(obj):
                if only_assumptions and not val.prov.is_assumption:
                    continue
                rows.append(
                    {
                        "group": group,
                        "material": key,
                        "property": prop,
                        "value": f"{val.magnitude:g} {val.unit}",
                        "provenance": str(val.prov),
                    }
                )
    return rows


def all_values() -> list[tuple[str, str, str, Value]]:
    """Yield ``(group, material_key, property_name, Value)`` for the whole library."""
    out = []
    for group, table in (("fluid", FLUIDS), ("cell", CELL_TYPES), ("substrate", SUBSTRATES)):
        for key, obj in table.items():
            for prop, val in _iter_values(obj):
                out.append((group, key, prop, val))
    return out


__all__ = [
    "Provenance",
    "Value",
    "Fluid",
    "CellType",
    "Substrate",
    "FLUIDS",
    "CELL_TYPES",
    "SUBSTRATES",
    "WATER",
    "PBS",
    "CELL_CULTURE_MEDIUM",
    "RBC",
    "WBC",
    "PLATELET",
    "MCF7",
    "HELA",
    "A549",
    "POLYSTYRENE_BEAD",
    "LIPID_DROPLET",
    "LINBO3_128YX",
    "PDMS",
    "GLASS_BOROSILICATE",
    "GOLD",
    "get_fluid",
    "get_cell",
    "get_substrate",
    "audit",
    "all_values",
]
