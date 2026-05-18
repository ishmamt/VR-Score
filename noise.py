"""
M3 — Camera Noise Fingerprint via Wavelet-Residual Kurtosis.

Forensic motivation
-------------------
Every digital camera sensor imprints a characteristic, spatially-fixed
noise signature on each frame it captures: Photo Response Non-Uniformity
(PRNU).  This arises from pixel-level manufacturing imperfections and the
full camera pipeline (demosaicing, gamma correction, JPEG / codec
compression).  The resulting noise residual has a *heavy-tailed*
(high-kurtosis) distribution — a consequence of the non-linear, spatially
correlated nature of real imaging.

AI video generators are optimised for perceptual plausibility, not for
replicating camera physics.  Their output noise fields are closer to
Gaussian or uniform distributions (excess kurtosis ≈ 0), lacking the
heavy-tailed character of genuine sensor noise.

Reference-based PRNU (comparing an estimated camera fingerprint to a
query frame) is the gold standard but requires 10–20 frames from the same
known-real camera.  For blind, single-video analysis we use a
*reference-free* approach: estimate the noise residual directly and
measure its statistical character.

Implementation
--------------
For each sampled frame we compute:

  noise_residual = frame_gray – denoise_wavelet(frame_gray)

Wavelet denoising (BayesShrink, soft-thresholding) is far superior to
Gaussian-blur subtraction for isolating sensor noise because it operates
in frequency subbands and preserves edges — avoiding the scene-content
bleed-through that Gaussian subtraction introduces.

We then compute the excess kurtosis (Fisher's definition, kurtosis = 0
for a Gaussian) of the flattened residual.

Higher excess kurtosis → heavier-tailed noise → more consistent with a
real imaging pipeline.

Aggregation uses the median across frames for outlier robustness.

References
----------
Lukas J. et al. "Digital Camera Identification from Sensor Pattern Noise."
IEEE Transactions on Information Forensics and Security, 2006.

Mandelli S. et al. "PRaNA: PRNU-based Technique to Tell Real and Deepfake
Videos Apart." ICIP, 2022.

Vamshi M.V. et al. "A dual-stream model based on PRNU and quaternion RGB
for detecting fake faces." PLOS ONE, 2025.
"""

import cv2
import numpy as np
from scipy.stats import kurtosis
from skimage.restoration import denoise_wavelet
from typing import List

from vr_score.config import WaveletConfig


class NoiseMetric:
    """
    Estimates camera-noise kurtosis via wavelet-denoising residuals.

    Returns a raw excess kurtosis score; *higher* values indicate a noise
    distribution more consistent with a real imaging sensor.
    """

    def __init__(self, config: WaveletConfig) -> None:
        self._wavelet = config.wavelet

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, frames: List[np.ndarray]) -> float:
        """
        Compute the median noise kurtosis across all frames.

        Args:
            frames: BGR frames at native resolution (uniformly sampled).

        Returns:
            Median excess kurtosis of the wavelet noise residuals.
            Higher → noise distribution more consistent with real sensors.
        """
        if not frames:
            return 0.0

        per_frame = [self._kurtosis_for_frame(f) for f in frames]
        return float(np.median(per_frame))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kurtosis_for_frame(self, frame: np.ndarray) -> float:
        """Compute excess kurtosis of the wavelet noise residual for one frame."""
        # Work in float [0, 1] for numerical stability in the wavelet domain
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

        denoised = denoise_wavelet(
            gray,
            method="BayesShrink",
            mode="soft",
            wavelet=self._wavelet,
            wavelet_levels=None,   # auto-select based on image size
            channel_axis=None,     # single-channel (grayscale) input
            rescale_sigma=True,    # rescale noise estimate to input range
        )

        # Noise residual: what the denoiser removed
        residual = gray - denoised

        # Fisher's excess kurtosis: 0 for Gaussian, positive for heavy tails
        return float(kurtosis(residual.flatten(), fisher=True))
