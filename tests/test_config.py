"""Experiment configuration: validation, YAML round-trip, hashing."""

import pydantic
import pytest
import yaml

from biosim_lab.core.config import BaseConfigModel, ExperimentConfig, Frequency, Length


class _Params(BaseConfigModel):
    frequency: Frequency = 20e6
    width: Length = 300e-6


def test_unit_strings_are_validated_into_si():
    p = _Params(frequency="20 MHz", width="300 um")
    assert p.frequency == pytest.approx(20e6)
    assert p.width == pytest.approx(300e-6)


def test_extra_keys_are_rejected():
    with pytest.raises(pydantic.ValidationError):
        _Params(frequency="20 MHz", typo=1)


def test_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig(
        name="demo", instrument="saw_sorter", seed=1, params={"frequency": "20 MHz"}
    )
    path = cfg.to_yaml(tmp_path / "c.yaml")
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.name == cfg.name
    assert loaded.instrument == cfg.instrument
    assert loaded.params["frequency"] == cfg.params["frequency"]


def test_output_dir_is_resolved_against_the_config_file(tmp_path):
    (tmp_path / "sub").mkdir()
    path = tmp_path / "sub" / "c.yaml"
    path.write_text(
        yaml.safe_dump({"instrument": "saw_sorter", "output_dir": "results"}),
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.output_dir == tmp_path / "sub" / "results"


def test_hash_is_stable_and_ignores_output_dir():
    a = ExperimentConfig(instrument="saw_sorter", params={"frequency": 1.0})
    b = a.model_copy(update={"output_dir": "elsewhere"})
    c = a.model_copy(update={"params": {"frequency": 2.0}})
    assert a.hash == b.hash
    assert a.hash != c.hash


def test_name_rejects_path_separators():
    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig(name="a/b", instrument="saw_sorter")
