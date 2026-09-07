"""Experiment configuration: YAML <-> validated objects.

A biosim-lab run is fully described by one YAML file::

    name: ctc_vs_rbc
    instrument: saw_sorter
    seed: 12345
    output_dir: results/
    params:            # instrument-specific block, validated by the plugin
      frequency: 20 MHz
      ...

Unit-bearing scalars are written as strings (``"20 MHz"``, ``"300 um"``) and
converted to bare SI floats at validation time by :class:`Quantity` fields, so
the numerical core never sees a ``pint`` object (ARCHITECTURE.md section 8).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import core_schema

from biosim_lab.core.units import to_si


class _SIQuantity:
    """pydantic annotation that parses a unit string into an SI float.

    Usage::

        Frequency = Annotated[float, SI("Hz")]
        freq: Frequency = 20e6
    """

    def __init__(self, unit: str) -> None:
        self.unit = unit

    def __get_pydantic_core_schema__(self, source_type: Any, handler: Any) -> Any:
        unit = self.unit

        def _validate(value: Any) -> float:
            return to_si(value, unit)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(float),
        )

    def __get_pydantic_json_schema__(self, schema: Any, handler: Any) -> dict[str, Any]:
        return {
            "anyOf": [{"type": "number"}, {"type": "string"}],
            "description": f"quantity convertible to SI base units of '{self.unit}'",
        }


def SI(unit: str) -> _SIQuantity:  # noqa: N802 - reads as a type annotation helper
    """Annotate a float field as a unit-checked quantity (stored in SI)."""
    return _SIQuantity(unit)


# Ready-made annotated aliases for the units used across instruments.
Frequency = Annotated[float, SI("Hz")]
Length = Annotated[float, SI("m")]
Voltage = Annotated[float, SI("V")]
Pressure = Annotated[float, SI("Pa")]
VolumeFlow = Annotated[float, SI("m**3/s")]
Velocity = Annotated[float, SI("m/s")]
Time = Annotated[float, SI("s")]
Density = Annotated[float, SI("kg/m**3")]
Viscosity = Annotated[float, SI("Pa*s")]
Conductivity = Annotated[float, SI("S/m")]
Temperature = Annotated[float, SI("K")]


class BaseConfigModel(BaseModel):
    """Common pydantic settings for every biosim-lab config block."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExperimentConfig(BaseConfigModel):
    """Top-level experiment description.

    ``params`` stays a raw dict here; each :class:`~biosim_lab.core.plugin.Instrument`
    validates it against its own ``ConfigModel`` in ``setup()``.  That keeps the
    core free of any knowledge about specific devices.
    """

    name: str = Field("experiment", description="Human-readable run name")
    instrument: str = Field(..., description="Entry-point name of the instrument plugin")
    seed: int | None = Field(None, description="RNG seed; None means non-reproducible")
    output_dir: Path = Field(Path("results"), description="Where results are written")
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _slug_safe(cls, v: str) -> str:
        if not v or any(ch in v for ch in "/\\"):
            raise ValueError("name must be non-empty and free of path separators")
        return v

    # -- io ---------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate a YAML configuration file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level of a config file must be a mapping")
        cfg = cls.model_validate(data)
        # Resolve a relative output_dir against the config file's own directory,
        # so `biosim run configs/foo.yaml` writes next to the config, not the CWD.
        if not cfg.output_dir.is_absolute():
            cfg = cfg.model_copy(update={"output_dir": path.parent / cfg.output_dir})
        return cfg

    def to_yaml(self, path: str | Path) -> Path:
        """Write this configuration back to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(self.model_dump_json())
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        return path

    @property
    def hash(self) -> str:
        """Stable 12-character content hash, used for provenance in results."""
        blob = self.model_dump_json(exclude={"output_dir"})
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def validated_params(self, model: type[BaseModel] | None) -> BaseModel | dict[str, Any]:
        """Validate ``params`` against an instrument's ``ConfigModel``."""
        if model is None:
            return dict(self.params)
        return model.model_validate(self.params)


__all__ = [
    "SI",
    "BaseConfigModel",
    "ExperimentConfig",
    "Frequency",
    "Length",
    "Voltage",
    "Pressure",
    "VolumeFlow",
    "Velocity",
    "Time",
    "Density",
    "Viscosity",
    "Conductivity",
    "Temperature",
]
