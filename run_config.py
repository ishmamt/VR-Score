"""
Run-time configuration for the dataset pipeline.

This module is deliberately separate from ``vr_score.config`` which holds
*scoring algorithm* constants.  Here we deal purely with *infrastructure*:
where the dataset lives, where results go, how the run behaves.

Config is stored in a TOML file.  A fully-annotated example is provided
at ``run_config.toml`` in the project root.  Every key has a default so a
minimal config only needs to set ``[dataset] directory``.

TOML parsing uses ``tomllib`` (stdlib in Python ≥ 3.11) or the third-party
``tomli`` backport for earlier versions.

Example
-------
>>> from vr_score.run_config import load_run_config
>>> cfg = load_run_config("run_config.toml")
>>> print(cfg.dataset.directory)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from vr_score.config import (
    VRScoreConfig,
    SamplingConfig,
    WaveletConfig,
    OpticalFlowConfig,
    LatentConfig,
    WeightConfig,
)

# ---------------------------------------------------------------------------
# TOML import shim — tomllib (3.11+) or tomli backport
# ---------------------------------------------------------------------------

try:
    import tomllib  # type: ignore[import]
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[import,no-redef]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "TOML parsing requires Python 3.11+ or the 'tomli' package.\n"
            "Install it with:  pip install tomli"
        ) from exc


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

_DEFAULT_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]


@dataclass(frozen=True)
class DatasetConfig:
    """Describes the on-disk layout of the video dataset."""

    # Root directory whose immediate sub-directories are source-model folders.
    directory: str = ""

    # File extensions (lowercase, with leading dot) that will be treated as
    # video files.  Any file whose suffix is not in this list is ignored.
    video_extensions: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EXTENSIONS)
    )


@dataclass(frozen=True)
class OutputConfig:
    """Controls where and how results are written."""

    # Directory that will receive the output file.  Created if absent.
    directory: str = "results"

    # Base filename without extension.  The correct extension is appended
    # automatically based on ``format``.
    filename: str = "vr_scores"

    # "csv"   → one row per video, appended incrementally.
    # "jsonl" → one JSON object per line, appended incrementally.
    #           Use ``scripts/jsonl_to_json.py`` to convert to a JSON array.
    format: str = "csv"


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration bundling dataset, output, and execution settings."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # PyTorch device string for DINOv2 ("cpu", "cuda", "cuda:1", "mps").
    # Empty string → auto-detect.
    device: str = ""

    # Path to a calibration JSON file produced by ``MetricNormalizer.save()``.
    # Empty string → use heuristic default calibration parameters.
    calibration: str = ""

    # When True, videos whose filename already appears in the output file with
    # status "ok" are skipped.  Videos that previously failed are retried.
    resume: bool = True

    # Python logging level name: "DEBUG", "INFO", "WARNING", "ERROR".
    log_level: str = "INFO"

    # VR-Score algorithm configuration, built from the [scorer.*] TOML sections.
    scorer: VRScoreConfig = field(default_factory=VRScoreConfig)

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    @property
    def output_path(self) -> Path:
        """Absolute path of the output file, extension already appended."""
        ext = ".jsonl" if self.output.format == "jsonl" else ".csv"
        return Path(self.output.directory) / (self.output.filename + ext)

    @property
    def device_or_none(self) -> Optional[str]:
        """Returns the device string, or None if auto-detection is requested."""
        return self.device if self.device else None


# ---------------------------------------------------------------------------
# TOML → dataclass conversion helpers
# ---------------------------------------------------------------------------

def _build_scorer_config(raw: dict) -> VRScoreConfig:
    """
    Construct a ``VRScoreConfig`` from the ``[scorer.*]`` subsections of the
    raw TOML dict.  Any key absent from TOML falls back to the dataclass
    default.
    """
    s = raw.get("scorer", {})

    sampling = SamplingConfig(
        **{k: v for k, v in s.get("sampling", {}).items()
           if k in SamplingConfig.__dataclass_fields__}
    )
    wavelet = WaveletConfig(
        **{k: v for k, v in s.get("wavelet", {}).items()
           if k in WaveletConfig.__dataclass_fields__}
    )
    optical_flow = OpticalFlowConfig(
        **{k: v for k, v in s.get("optical_flow", {}).items()
           if k in OpticalFlowConfig.__dataclass_fields__}
    )
    latent = LatentConfig(
        **{k: v for k, v in s.get("latent", {}).items()
           if k in LatentConfig.__dataclass_fields__}
    )
    weights = WeightConfig(
        **{k: v for k, v in s.get("weights", {}).items()
           if k in WeightConfig.__dataclass_fields__}
    )

    return VRScoreConfig(
        sampling=sampling,
        wavelet=wavelet,
        optical_flow=optical_flow,
        latent=latent,
        weights=weights,
    )


def _build_dataset_config(raw: dict) -> DatasetConfig:
    d = raw.get("dataset", {})
    kwargs = {k: v for k, v in d.items()
              if k in DatasetConfig.__dataclass_fields__}
    return DatasetConfig(**kwargs)


def _build_output_config(raw: dict) -> OutputConfig:
    o = raw.get("output", {})
    fmt = o.get("format", "csv").lower()
    if fmt not in ("csv", "jsonl"):
        raise ValueError(
            f"[output] format must be 'csv' or 'jsonl', got '{fmt}'"
        )
    kwargs = {k: v for k, v in o.items()
              if k in OutputConfig.__dataclass_fields__}
    kwargs["format"] = fmt
    return OutputConfig(**kwargs)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_run_config(toml_path: str) -> RunConfig:
    """
    Parse a TOML config file and return a ``RunConfig``.

    Args:
        toml_path: Path to the TOML configuration file.

    Returns:
        Fully populated ``RunConfig`` with all missing keys defaulted.

    Raises:
        FileNotFoundError: If ``toml_path`` does not exist.
        ValueError:        If any value fails validation.
    """
    path = Path(toml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {toml_path}")

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    run_section = raw.get("run", {})

    return RunConfig(
        dataset=_build_dataset_config(raw),
        output=_build_output_config(raw),
        device=run_section.get("device", ""),
        calibration=run_section.get("calibration", ""),
        resume=run_section.get("resume", True),
        log_level=run_section.get("log_level", "INFO"),
        scorer=_build_scorer_config(raw),
    )
