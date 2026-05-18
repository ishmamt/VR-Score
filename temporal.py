"""
M2 — Temporal Soundness via Optical Flow Background Jitter.

Forensic motivation
-------------------
A real camera captures scenes with very high inter-frame temporal
redundancy: adjacent frames are correlated because the physical world
changes smoothly and the sensor integrates light over a fixed shutter
interval.  Consequently, the optical flow of truly *static* background
regions in authentic footage has a flow magnitude near zero with very low
variance.

AI video generators synthesise frames semi-independently in a learned
latent space.  Even when no object is meant to move, the generator cannot
perfectly reproduce background texture from one frame to the next — the
result is a low-amplitude but persistent "shimmer" or "flicker" that is
invisible to a casual viewer but measurable as elevated variance in the
flow magnitude of ostensibly static pixels.

Implementation
--------------
Dense Farneback optical flow is computed for each consecutive frame pair.
We then isolate the "background" region by thresholding: pixels whose flow
magnitude falls below the 50th percentile are classified as static.  The
variance of flow magnitudes within that mask is our per-pair jitter score.

A minimum of 10 background pixels is required for a valid estimate; pairs
that don't meet this threshold (e.g. high-motion scenes) are discarded.
The final score is the *median* across all valid frame pairs, which makes
the estimate robust to occasional high-motion windows.

Lower jitter → the video behaves like a real camera capture.
Higher jitter → consistent with AI temporal incoherence.

Reference
---------
"Exposing AI-generated Videos: A Benchmark Dataset and Detection Framework
Based on Local and Global Temporal Defects." IEEE Access, 2024.
arXiv:2405.04133.
"""

import cv2
import numpy as np
from typing import List, Optional

from vr_score.config import OpticalFlowConfig


class TemporalMetric:
    """
    Measures background motion variance via dense optical flow.

    Returns a raw jitter score; *lower* values indicate more temporal
    realism.  The ``MetricNormalizer`` inverts this before compositing.
    """

    _MIN_STATIC_PIXELS = 10

    def __init__(self, config: OpticalFlowConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, frames: List[np.ndarray]) -> float:
        """
        Compute the median background jitter across all consecutive frame pairs.

        Args:
            frames: Consecutive BGR frames at native resolution.
                    Requires at least 2 frames; 48–60 recommended for a
                    stable estimate.

        Returns:
            Median background flow variance.
            Lower → more temporally realistic.
        """
        if len(frames) < 2:
            return 0.0

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        jitter_scores: List[float] = []

        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = self._pair_jitter(prev_gray, gray)
            if score is not None:
                jitter_scores.append(score)
            prev_gray = gray

        return float(np.median(jitter_scores)) if jitter_scores else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pair_jitter(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> Optional[float]:
        """
        Compute the background jitter for a single consecutive frame pair.

        Returns None when there are too few static pixels for a reliable
        estimate (e.g. a high-motion pan or fast cut).
        """
        cfg = self._cfg
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            cfg.pyr_scale,
            cfg.levels,
            cfg.winsize,
            cfg.iterations,
            cfg.poly_n,
            cfg.poly_sigma,
            flags=0,
        )

        # Flow magnitude at every pixel
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Pixels below the static_percentile threshold are "background"
        threshold = np.percentile(mag, cfg.static_percentile)
        static_mask = mag < threshold

        if np.sum(static_mask) < self._MIN_STATIC_PIXELS:
            return None

        # High variance here means the "static" background is flickering
        return float(np.var(mag[static_mask]))
