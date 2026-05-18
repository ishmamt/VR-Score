"""
Central configuration for VR-Score.

All tunable constants live here so that metric classes remain stateless
with respect to hyper-parameters and can be swapped or extended without
touching analysis logic.

Dataclasses are frozen so configuration objects are hashable and cannot
be accidentally mutated after construction.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingConfig:
    """Controls how many frames are drawn from the video for each metric."""

    # Frames spread uniformly across the whole video (M1 spatial, M3 noise).
    spatial_frames: int = 8
    noise_frames: int = 8

    # Consecutive frames starting at the first frame (M2 temporal / optical flow).
    # More frames → more stable jitter estimate; 60 is the minimum recommended.
    temporal_frames: int = 60

    # Frames drawn from a central time window (M4 latent trajectory).
    # ReStraV uses 24 frames over a 2-second window.
    latent_frames: int = 24
    latent_window_seconds: float = 2.0


@dataclass(frozen=True)
class WaveletConfig:
    """
    Wavelet family and decomposition depth used by both the spatial and
    noise metrics.  Daubechies-4 offers a good balance of frequency
    localisation and compact support.
    """

    wavelet: str = "db4"
    decomposition_level: int = 2


@dataclass(frozen=True)
class OpticalFlowConfig:
    """Farneback dense optical flow parameters (see cv2.calcOpticalFlowFarneback)."""

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2

    # Pixels whose flow magnitude falls below this percentile are treated as
    # "background" (ostensibly static).  Their flow variance is the jitter score.
    static_percentile: float = 50.0


@dataclass(frozen=True)
class LatentConfig:
    """DINOv2 model identity and pre-processing constants."""

    # torch.hub repository and model name
    repo: str = "facebookresearch/dinov2"
    model_name: str = "dinov2_vits14"

    # Spatial resolution expected by the ViT patch embedding
    input_size: int = 224

    # CLS token dimension for ViT-S/14
    embedding_dim: int = 384


@dataclass(frozen=True)
class WeightConfig:
    """
    Default heuristic weights for the VR-Score composite.

    M4 (latent trajectory / ReStraV) receives the highest weight because
    it achieves 97.17 % accuracy / 98.63 % AUROC as a standalone metric
    (Internò et al., NeurIPS 2025).  The remaining weight is split evenly
    across the three signal-level metrics.

    These defaults are overridden when `VRScorer.fit_weights()` is called
    with a labelled calibration dataset.
    """

    spatial: float = 0.20
    temporal: float = 0.20
    noise: float = 0.20
    latent: float = 0.40


@dataclass(frozen=True)
class VRScoreConfig:
    """Root configuration object passed to VRScorer."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)


# Module-level singleton so callers can do `from vr_score.config import DEFAULT_CONFIG`
DEFAULT_CONFIG = VRScoreConfig()
