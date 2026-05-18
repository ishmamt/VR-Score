"""
Per-metric normalisation to a common [0, 1] scale.

Problem
-------
Each raw metric has a different value range and a different *direction*:

  M1 spatial  (diagonal energy ratio): higher raw → more real    (+ve)
  M2 temporal (background jitter):     lower raw  → more real    (−ve, invert)
  M3 noise    (kurtosis):              higher raw → more real    (+ve)
  M4 curvature:                        lower raw  → more real    (−ve, invert)
  M4 dist_var:                         lower raw  → more real    (−ve, invert)

Normalisation strategy
----------------------
We use a logistic (sigmoid) mapping centred on a calibration midpoint:

    normalised = σ( sharpness × (x − midpoint) )
               = 1 / (1 + exp(−sharpness × (x − midpoint)))

For metrics where a *lower* raw value means *more real* the sign is flipped
before the sigmoid so that the output still increases toward 1 as the video
becomes more realistic.

The midpoint is set to the value at which the score should be 0.5 (the
decision boundary between real and AI), and sharpness controls how steeply
the sigmoid transitions around that boundary.

Default values are conservative heuristics for zero-shot use.
Call ``MetricNormalizer.fit()`` with labelled data to learn better parameters.

Persistence
-----------
``MetricNormalizer.save()`` serialises calibration to a JSON file.
``MetricNormalizer.load()`` restores it.  This allows calibration to be
run once on a validation set and then reused at inference time.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MetricCalibration:
    """Logistic normalisation parameters for a single metric."""

    midpoint: float    # Raw value at which normalised output = 0.5
    sharpness: float   # Sigmoid slope; larger → sharper decision boundary
    invert: bool       # If True, the raw value is negated before the sigmoid
                       # so that lower raw values map to higher (more-real) scores

    def normalize(self, raw: float) -> float:
        """Map a single raw value to [0, 1]."""
        signed = -(raw - self.midpoint) if self.invert else (raw - self.midpoint)
        # Clamp the exponent to avoid overflow in exp()
        exponent = np.clip(-self.sharpness * signed, -500, 500)
        return float(1.0 / (1.0 + np.exp(exponent)))


# ---------------------------------------------------------------------------
# Heuristic defaults
# ---------------------------------------------------------------------------
# These are starting-point estimates based on typical value ranges.
# They are intentionally conservative (low sharpness) so the score degrades
# gracefully on unseen content until calibration data is available.
# ---------------------------------------------------------------------------
DEFAULT_CALIBRATIONS: Dict[str, MetricCalibration] = {
    "spatial": MetricCalibration(
        midpoint=0.05,
        sharpness=40.0,
        invert=False,    # higher diagonal energy → more real
    ),
    "temporal": MetricCalibration(
        midpoint=5e-5,
        sharpness=40_000.0,
        invert=True,     # lower jitter → more real
    ),
    "noise": MetricCalibration(
        midpoint=1.5,
        sharpness=1.5,
        invert=False,    # higher kurtosis → more real
    ),
    "latent_curvature": MetricCalibration(
        midpoint=0.25,
        sharpness=12.0,
        invert=True,     # lower curvature → more real
    ),
    "latent_dist_var": MetricCalibration(
        midpoint=0.05,
        sharpness=30.0,
        invert=True,     # lower distance variance → more real
    ),
}

_KNOWN_METRICS = frozenset(DEFAULT_CALIBRATIONS.keys())


class MetricNormalizer:
    """
    Holds calibration parameters and normalises raw metric values to [0, 1].

    Usage (zero-shot, default calibration)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> n = MetricNormalizer()
    >>> n.normalize("spatial", 0.08)
    0.73...

    Usage (with labelled calibration data)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> n = MetricNormalizer()
    >>> n.fit(real_values={"spatial": [...]}, ai_values={"spatial": [...]})
    >>> n.save("calibration.json")

    Usage (loading saved calibration)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> n = MetricNormalizer.load("calibration.json")
    """

    def __init__(
        self,
        calibrations: Optional[Dict[str, MetricCalibration]] = None,
    ) -> None:
        # Start from defaults and overlay any provided calibrations
        self._cal: Dict[str, MetricCalibration] = dict(DEFAULT_CALIBRATIONS)
        if calibrations:
            self._cal.update(calibrations)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, metric_name: str, raw_value: float) -> float:
        """
        Normalise one raw metric value to [0, 1].

        Args:
            metric_name: One of the known metric keys (see ``_KNOWN_METRICS``).
            raw_value:   The raw output from the corresponding metric class.

        Returns:
            Float in [0, 1]; 1.0 means maximally consistent with real video.

        Raises:
            KeyError: If ``metric_name`` is not a recognised metric key.
        """
        if metric_name not in self._cal:
            raise KeyError(
                f"Unknown metric '{metric_name}'.  "
                f"Known metrics: {sorted(self._cal)}"
            )
        return self._cal[metric_name].normalize(raw_value)

    # ------------------------------------------------------------------
    # Calibration fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        real_values: Dict[str, List[float]],
        ai_values: Dict[str, List[float]],
    ) -> None:
        """
        Estimate calibration parameters from labelled metric observations.

        The midpoint is set to the average of the medians of the two classes.
        The sharpness is derived so that the sigmoid spans roughly [0.1, 0.9]
        across ±1 standard deviation of the pooled distribution.

        Args:
            real_values: Mapping from metric name to a list of raw values
                         measured on confirmed-real videos.
            ai_values:   Mapping from metric name to a list of raw values
                         measured on confirmed-AI videos.
        """
        for name, current_cal in self._cal.items():
            reals = real_values.get(name)
            ais   = ai_values.get(name)
            if not reals or not ais:
                continue

            real_arr = np.asarray(reals, dtype=np.float64)
            ai_arr   = np.asarray(ais,   dtype=np.float64)

            midpoint = float(
                (np.median(real_arr) + np.median(ai_arr)) / 2.0
            )
            pooled_std = float(
                np.std(np.concatenate([real_arr, ai_arr]))
            ) + 1e-8
            # σ(k × 1) ≈ 0.73, σ(k × 2) ≈ 0.88  →  k = 2.2 / std gives ~[0.1,0.9]
            sharpness = 2.2 / pooled_std

            self._cal[name] = MetricCalibration(
                midpoint=midpoint,
                sharpness=sharpness,
                invert=current_cal.invert,   # preserve the direction
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise calibration parameters to a JSON file."""
        data = {k: asdict(v) for k, v in self._cal.items()}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "MetricNormalizer":
        """Deserialise calibration parameters from a JSON file."""
        raw = json.loads(Path(path).read_text())
        calibrations = {k: MetricCalibration(**v) for k, v in raw.items()}
        return cls(calibrations)
