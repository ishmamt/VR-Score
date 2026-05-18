"""
M1 — Spatial Realism via Wavelet Diagonal Band Analysis.

Forensic motivation
-------------------
AI video generators (GANs, diffusion models, DiTs) contain up-sampling
operations — transposed convolutions, pixel-shuffle layers, VAE decoders —
that leave quasi-periodic spectral fingerprints in the frequency domain.
These are visible as anomalous peaks in the Fourier spectrum.

Naive FFT-based high/low energy ratios fail on compressed video because
H.264 / H.265 block-DCT quantisation destroys *horizontal* and *vertical*
high-frequency components.  However, research published at NeurIPS 2025
showed that *diagonal* mid-high frequency components are substantially
more compression-resistant and remain discriminative across diverse
generative architectures.

Implementation
--------------
We apply a 2-level Daubechies-4 discrete wavelet transform (DWT).
The two-dimensional DWT decomposes each frame into four subbands:

  LL  — low-low  (approximation, mostly scene content)
  LH  — low-high  (horizontal edges)
  HL  — high-low  (vertical edges)
  HH  — high-high (diagonal detail) ← our signal

The HH subband at each decomposition level captures diagonal mid-to-high
frequency texture.  We compute the ratio of total HH energy across both
levels to the LL (approximation) energy.

A higher ratio indicates richer real-world diagonal texture.  AI content
typically shows either an unnaturally flat diagonal spectrum (over-smoothed)
or isolated spikes (generator artefacts), both of which separate from the
smooth 1/f decay of natural images.

Aggregation across frames uses the median (rather than the mean) to resist
outlier frames caused by scene cuts or motion blur.

Reference
---------
Corvi R. et al. "Seeing What Matters: Generalizable AI-generated Video
Detection with Forensic-Oriented Augmentation." NeurIPS 2025.
arXiv:2506.16802.
"""

import cv2
import numpy as np
import pywt
from typing import List

from vr_score.config import WaveletConfig


class SpatialMetric:
    """
    Computes the diagonal wavelet energy ratio for a set of video frames.

    The returned value is a raw (un-normalised) score.  Normalization to
    [0, 1] is handled by ``MetricNormalizer``.
    """

    def __init__(self, config: WaveletConfig) -> None:
        self._wavelet = config.wavelet
        self._level = config.decomposition_level

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, frames: List[np.ndarray]) -> float:
        """
        Compute the median diagonal energy ratio across all frames.

        Args:
            frames: BGR frames at native resolution.  Must be non-empty.

        Returns:
            Median ratio of HH-subband energy to LL-subband energy.
            Higher → more natural diagonal texture structure.
        """
        if not frames:
            return 0.0

        per_frame = [self._score_frame(f) for f in frames]
        return float(np.median(per_frame))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_frame(self, frame: np.ndarray) -> float:
        """Return the diagonal energy ratio for a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        coeffs = pywt.wavedec2(gray, self._wavelet, level=self._level)

        # coeffs layout (pywt convention):
        #   coeffs[0]         → cA_N  (LL approximation at deepest level)
        #   coeffs[1]         → (cH_N, cV_N, cD_N)  detail at level N
        #   ...
        #   coeffs[level]     → (cH_1, cV_1, cD_1)  detail at level 1
        approx = coeffs[0]
        approx_energy = float(np.mean(approx ** 2)) + 1e-8

        # Sum HH (diagonal, cD) energy across all decomposition levels.
        diagonal_energy = 0.0
        for cH, cV, cD in coeffs[1:]:  # each element is a 3-tuple of detail bands
            diagonal_energy += float(np.mean(cD ** 2))

        return diagonal_energy / approx_energy
