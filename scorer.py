"""
VRScorer — the top-level composite pipeline.

This module wires together all four forensic metrics and the normaliser
into a single ``score(video_path)`` call.

Composite formula
-----------------
    VR-Score = w₁·S  +  w₂·T  +  w₃·N  +  w₄·L

where each component is a normalised [0, 1] value:

  S  — spatial realism      (M1, wavelet diagonal energy ratio)
  T  — temporal soundness   (M2, inverted optical-flow background jitter)
  N  — noise fingerprint    (M3, wavelet-residual kurtosis)
  L  — latent trajectory    (M4, average of inverted curvature and
                              inverted distance variance)

Score interpretation
--------------------
  0.0  → strongly AI-generated
  1.0  → strongly consistent with real camera capture

Default weights
---------------
  w₁ = 0.20   w₂ = 0.20   w₃ = 0.20   w₄ = 0.40

M4 carries the most weight because ReStraV achieves the highest reported
accuracy of any single forensics-based metric for AI-video detection
(97.17% accuracy, 98.63% AUROC, NeurIPS 2025).

Weight learning
---------------
Call ``fit_weights(real_metrics, ai_metrics)`` with lists of
``NormalisedMetrics`` from labelled videos to learn weights via
Non-Negative Least Squares (NNLS) with a sum-to-1 constraint.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scipy.optimize import nnls

from vr_score.config import VRScoreConfig, DEFAULT_CONFIG
from vr_score.video_loader import VideoLoader
from vr_score.normalization import MetricNormalizer
from vr_score.metrics.spatial import SpatialMetric
from vr_score.metrics.temporal import TemporalMetric
from vr_score.metrics.noise import NoiseMetric
from vr_score.metrics.latent import LatentMetric


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawMetrics:
    """Un-normalised outputs from each metric.  Useful for diagnostics."""

    spatial: float            # Diagonal wavelet energy ratio (higher = more real)
    temporal: float           # Background jitter variance    (lower  = more real)
    noise: float              # Noise excess kurtosis         (higher = more real)
    latent_curvature: float   # Mean trajectory curvature     (lower  = more real)
    latent_dist_var: float    # Step-distance variance        (lower  = more real)


@dataclass(frozen=True)
class NormalisedMetrics:
    """
    Each component mapped to [0, 1] where 1 = maximally consistent with
    real video.  This is the representation used for weight learning.
    """

    spatial: float
    temporal: float
    noise: float
    latent: float     # Average of the two normalised latent sub-scores


@dataclass(frozen=True)
class VRScoreResult:
    """Full output of a single ``VRScorer.score()`` call."""

    vr_score: float                  # Final composite VR-Score in [0, 1]
    normalised: NormalisedMetrics    # Per-component normalised scores
    raw: RawMetrics                  # Raw un-normalised values
    weights: Dict[str, float]        # Weights used in this computation


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "spatial":  0.20,
    "temporal": 0.20,
    "noise":    0.20,
    "latent":   0.40,
}


class VRScorer:
    """
    Full VR-Score pipeline.

    Basic usage
    -----------
    >>> scorer = VRScorer()
    >>> result = scorer.score("path/to/video.mp4")
    >>> print(f"VR-Score: {result.vr_score:.4f}")

    With custom calibration
    -----------------------
    >>> from vr_score.normalization import MetricNormalizer
    >>> normalizer = MetricNormalizer.load("calibration.json")
    >>> scorer = VRScorer(normalizer=normalizer)

    With GPU acceleration
    ---------------------
    >>> scorer = VRScorer(device="cuda")
    """

    def __init__(
        self,
        config: VRScoreConfig = DEFAULT_CONFIG,
        normalizer: Optional[MetricNormalizer] = None,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            config:     Full pipeline configuration.  Defaults to sensible values.
            normalizer: Pre-fitted normaliser.  Uses heuristic defaults when None.
            weights:    Override composite weights.  Falls back to defaults when None.
            device:     PyTorch device string for DINOv2 ('cpu', 'cuda', 'mps').
                        Auto-detected when None.
        """
        self._cfg        = config
        self._normalizer = normalizer or MetricNormalizer()
        self._weights    = self._merge_weights(weights)

        self._spatial  = SpatialMetric(config.wavelet)
        self._temporal = TemporalMetric(config.optical_flow)
        self._noise    = NoiseMetric(config.wavelet)
        self._latent   = LatentMetric(config.latent, device=device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, video_path: str) -> VRScoreResult:
        """
        Compute the VR-Score for a single video file.

        The video is opened at native resolution; no downsampling is
        performed before analysis.

        Args:
            video_path: Path to a video file supported by OpenCV.

        Returns:
            ``VRScoreResult`` containing the composite score, all
            per-component breakdowns, and the weights used.
        """
        with VideoLoader(video_path) as loader:
            raw = self._compute_raw_metrics(loader)

        normalised = self._normalise(raw)
        composite  = self._composite(normalised)

        return VRScoreResult(
            vr_score=composite,
            normalised=normalised,
            raw=raw,
            weights=dict(self._weights),
        )

    def fit_weights(
        self,
        real_metrics: List[NormalisedMetrics],
        ai_metrics: List[NormalisedMetrics],
    ) -> None:
        """
        Learn composite weights from labelled, normalised metric observations.

        Solves the constrained least-squares problem:
            min_{w ≥ 0} ‖Aw – y‖²   s.t.  Σwᵢ = 1

        via NNLS followed by L1-normalisation of the solution.

        After calling this method, ``score()`` will use the learned weights.

        Args:
            real_metrics: Per-component normalised scores for confirmed-real
                          videos (label = 1.0).
            ai_metrics:   Per-component normalised scores for confirmed-AI
                          videos (label = 0.0).
        """
        if not real_metrics or not ai_metrics:
            return

        X, y = self._build_regression_matrix(real_metrics, ai_metrics)
        w, residual = nnls(X, y)

        total = w.sum()
        if total < 1e-8:
            return   # Degenerate solution; keep current weights

        w_normalised = w / total
        keys = ["spatial", "temporal", "noise", "latent"]
        self._weights = {k: float(v) for k, v in zip(keys, w_normalised)}

    # ------------------------------------------------------------------
    # Private: metric computation
    # ------------------------------------------------------------------

    def _compute_raw_metrics(self, loader: VideoLoader) -> RawMetrics:
        cfg = self._cfg.sampling

        spatial_frames  = loader.sample_uniform(cfg.spatial_frames)
        noise_frames    = loader.sample_uniform(cfg.noise_frames)
        temporal_frames = loader.sample_consecutive(cfg.temporal_frames)
        latent_frames   = loader.sample_window(
            cfg.latent_frames, cfg.latent_window_seconds
        )

        spatial_score  = self._spatial.analyze(spatial_frames)
        temporal_score = self._temporal.analyze(temporal_frames)
        noise_score    = self._noise.analyze(noise_frames)
        latent_stats   = self._latent.analyze(latent_frames)

        return RawMetrics(
            spatial=spatial_score,
            temporal=temporal_score,
            noise=noise_score,
            latent_curvature=latent_stats.mean_curvature,
            latent_dist_var=latent_stats.distance_variance,
        )

    # ------------------------------------------------------------------
    # Private: normalisation and compositing
    # ------------------------------------------------------------------

    def _normalise(self, raw: RawMetrics) -> NormalisedMetrics:
        n = self._normalizer
        s  = n.normalize("spatial",          raw.spatial)
        t  = n.normalize("temporal",         raw.temporal)
        no = n.normalize("noise",            raw.noise)
        lc = n.normalize("latent_curvature", raw.latent_curvature)
        ld = n.normalize("latent_dist_var",  raw.latent_dist_var)

        # Average the two latent sub-scores into a single M4 component
        latent_combined = (lc + ld) / 2.0

        return NormalisedMetrics(
            spatial=s,
            temporal=t,
            noise=no,
            latent=latent_combined,
        )

    def _composite(self, norm: NormalisedMetrics) -> float:
        w = self._weights
        score = (
            w["spatial"]  * norm.spatial  +
            w["temporal"] * norm.temporal +
            w["noise"]    * norm.noise    +
            w["latent"]   * norm.latent
        )
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Private: weight learning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_regression_matrix(
        real_metrics: List[NormalisedMetrics],
        ai_metrics:   List[NormalisedMetrics],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the design matrix X (N, 4) and target vector y (N,) for NNLS.

        Real-video rows get label 1.0; AI-video rows get label 0.0.
        """
        def to_row(m: NormalisedMetrics) -> List[float]:
            return [m.spatial, m.temporal, m.noise, m.latent]

        rows   = [to_row(m) for m in real_metrics] + [to_row(m) for m in ai_metrics]
        labels = [1.0] * len(real_metrics) + [0.0] * len(ai_metrics)
        return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.float64)

    @staticmethod
    def _merge_weights(
        override: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Merge user-supplied weight overrides with defaults."""
        merged = dict(_DEFAULT_WEIGHTS)
        if override:
            merged.update(override)
        return merged
