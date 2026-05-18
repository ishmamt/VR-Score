"""
M4 — Latent Trajectory Curvature via DINOv2 (ReStraV).

Forensic motivation
-------------------
The "perceptual straightening" hypothesis (Hénaff et al., 2019; 2021)
states that biological visual systems represent natural video sequences as
near-linear trajectories in neural representation space.  The underlying
intuition is that the physical world evolves predictably: consecutive
real frames are causally related by physics, so their embeddings trace
smooth, geometrically straight paths.

AI-generated videos lack genuine physical causality between frames.  Each
frame is partly "hallucinated" by the generative model in its learned
latent space — the resulting trajectories in a neural feature space are
*curved* and *erratic*, even when the generated content appears visually
plausible.

ReStraV (Internò et al., NeurIPS 2025) applies this insight to deepfake
detection using DINOv2 ViT-S/14 as the feature encoder.  On the VidProM
benchmark a lightweight MLP classifier on top of these geometric features
achieves 97.17 % accuracy and 98.63 % AUROC in approximately 48 ms
end-to-end, outperforming all prior video-based methods.

Implementation
--------------
For a video we:

  1. Sample 24 frames from a 2-second window centred on the video midpoint.
  2. Resize each frame to 224 × 224 and normalise with ImageNet statistics.
  3. Run a forward pass through DINOv2 ViT-S/14 (lazy-loaded on first use)
     and extract the CLS token embedding  z_i ∈ ℝ^384.
  4. Compute trajectory statistics over the sequence z_0, …, z_{N-1}:

       Displacement vectors:  Δ_i = z_{i+1} – z_i
       Unit vectors:          û_i = Δ_i / ‖Δ_i‖

       Angular curvature at step i:
           θ_i = arccos( clamp(û_i · û_{i+1}, −1, 1) )   [radians]

       mean_curvature  = mean(θ_i),    i = 0 … N-3
       distance_variance = var(‖Δ_i‖), i = 0 … N-2

Lower curvature + lower distance variance → straighter, more uniform
trajectory → consistent with real camera footage.

The two sub-scores are returned as a ``LatentTrajectoryStats`` dataclass
and averaged into a single M4 component inside the scorer after
normalisation.

DINOv2 is lazy-loaded on the first call to ``analyze()`` to avoid
importing PyTorch at module import time (which can be slow).

Reference
---------
Internò C. et al. "AI-Generated Video Detection via Perceptual
Straightening." NeurIPS 2025. arXiv:2507.00583.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from dataclasses import dataclass
from typing import List, Optional

from vr_score.config import LatentConfig


@dataclass(frozen=True)
class LatentTrajectoryStats:
    """Geometric statistics of a video's trajectory in DINOv2 embedding space."""

    mean_curvature: float     # Mean angular deviation between consecutive steps (rad).
                              # Lower values → straighter → more real.
    distance_variance: float  # Variance of step-wise L2 distances.
                              # Lower values → more uniform → more real.
    n_frames: int             # Number of frames that were actually embedded.


class LatentMetric:
    """
    Computes DINOv2 latent-trajectory curvature and distance variance.

    The model is lazily initialised on the first call to ``analyze()``.
    """

    # ImageNet normalisation constants expected by DINOv2
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        config: LatentConfig,
        device: Optional[str] = None,
    ) -> None:
        self._cfg = config
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: Optional[torch.nn.Module] = None
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config.input_size, config.input_size)),
            T.ToTensor(),
            T.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, frames: List[np.ndarray]) -> LatentTrajectoryStats:
        """
        Compute latent trajectory statistics for a sequence of frames.

        Args:
            frames: BGR frames from a short, temporally contiguous window.
                    At least 3 frames are needed to compute curvature
                    (which requires two consecutive displacement vectors).

        Returns:
            ``LatentTrajectoryStats`` with mean_curvature and
            distance_variance.  Both are lower for real video.
        """
        self._ensure_model_loaded()

        if len(frames) < 3:
            return LatentTrajectoryStats(
                mean_curvature=0.0,
                distance_variance=0.0,
                n_frames=len(frames),
            )

        embeddings = self._embed_frames(frames)   # (N, D)

        if embeddings.shape[0] < 3:
            return LatentTrajectoryStats(
                mean_curvature=0.0,
                distance_variance=0.0,
                n_frames=embeddings.shape[0],
            )

        return self._compute_trajectory_stats(embeddings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load DINOv2 on first use to avoid startup overhead."""
        if self._model is not None:
            return

        self._model = torch.hub.load(
            self._cfg.repo,
            self._cfg.model_name,
            pretrained=True,
        )
        self._model.eval()
        self._model.to(self._device)

    def _embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Convert a list of BGR frames to a batch of DINOv2 CLS embeddings.

        Returns:
            Float32 array of shape (N, embedding_dim).
        """
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self._device)   # (N, 3, H, W)

        with torch.no_grad():
            # DINOv2 forward() returns the CLS token: (N, embedding_dim)
            embeddings = self._model(batch)

        return embeddings.cpu().float().numpy()

    @staticmethod
    def _compute_trajectory_stats(z: np.ndarray) -> LatentTrajectoryStats:
        """
        Compute curvature and distance variance from an (N, D) embedding array.

        Displacement vectors between consecutive embeddings:
            Δ_i = z[i+1] – z[i]     for i in 0 … N-2

        Unit displacement vectors:
            û_i = Δ_i / ‖Δ_i‖

        Angular curvature at step i:
            θ_i = arccos( clamp( û_i · û_{i+1} ) )    for i in 0 … N-3

        mean_curvature   = mean(θ_i)
        distance_variance = var(‖Δ_i‖)
        """
        # Step displacements: shape (N-1, D)
        deltas = z[1:] - z[:-1]

        # L2 norms of each step: shape (N-1,)
        norms = np.linalg.norm(deltas, axis=1)
        norms_safe = np.maximum(norms, 1e-8)

        # Unit displacement vectors: shape (N-1, D)
        unit_deltas = deltas / norms_safe[:, np.newaxis]

        # Dot products between consecutive unit displacements: shape (N-2,)
        dot_products = np.einsum(
            "id,id->i", unit_deltas[:-1], unit_deltas[1:]
        )
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # Angular curvature in radians: shape (N-2,)
        angles = np.arccos(dot_products)

        return LatentTrajectoryStats(
            mean_curvature=float(np.mean(angles)),
            distance_variance=float(np.var(norms)),
            n_frames=z.shape[0],
        )
