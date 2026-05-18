"""
Native-resolution video loading and frame sampling.

Design principle
----------------
Every frame returned by this module is at the video's *original* resolution.
Down-sampling before analysis would destroy high-frequency forensic cues —
the very signals that distinguish AI-generated from real camera content
(Mandelli et al., 2022; native-scale processing study in arXiv:2604.04634).

The VideoLoader is a context manager so the underlying cv2.VideoCapture is
always released, even when an exception propagates out of the caller.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable snapshot of a video file's technical properties."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


class VideoLoader:
    """
    Opens a video file and exposes three frame-sampling strategies:

    * ``sample_uniform``      — frames spread evenly across the whole clip.
    * ``sample_consecutive``  — a run of adjacent frames (optical flow use).
    * ``sample_window``       — uniformly spaced frames inside a short central
                                 window (latent trajectory use).

    All three return frames in BGR format at native resolution.
    """

    def __init__(self, video_path: str) -> None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        self.metadata = self._read_metadata(str(path))

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def sample_uniform(self, n: int) -> List[np.ndarray]:
        """
        Return *n* frames sampled at equal temporal intervals across the
        full video.  Used by the spatial (M1) and noise (M3) metrics which
        need a representative picture of static per-frame properties.

        Args:
            n: Number of frames to return.  Clamped to ``frame_count``.

        Returns:
            List of BGR frames in chronological order.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        n = min(n, self.metadata.frame_count)
        indices = np.linspace(0, self.metadata.frame_count - 1, n, dtype=int)
        return self._read_at_indices(indices.tolist())

    def sample_consecutive(
        self, n: int, start_frame: int = 0
    ) -> List[np.ndarray]:
        """
        Return *n* consecutive frames starting at *start_frame*.

        Consecutive frames are mandatory for dense optical flow because
        Farneback flow requires adjacent pairs with no temporal gaps.

        Args:
            n:           Number of frames.
            start_frame: Index of the first frame (0-based).

        Returns:
            List of BGR frames in chronological order.
        """
        end = min(start_frame + n, self.metadata.frame_count)
        indices = list(range(start_frame, end))
        return self._read_at_indices(indices)

    def sample_window(
        self, n: int, window_seconds: float
    ) -> List[np.ndarray]:
        """
        Sample *n* uniformly-spaced frames from a ``window_seconds``-long
        segment centred on the temporal midpoint of the video.

        This mirrors the ReStraV protocol: 24 frames from a 2-second
        window near the middle of the clip, avoiding opening/closing
        transitions that are atypical of the video's main content.

        Args:
            n:              Number of frames to return.
            window_seconds: Duration of the sampling window.

        Returns:
            List of BGR frames in chronological order.
        """
        fps = self.metadata.fps if self.metadata.fps > 0 else 25.0
        mid = self.metadata.frame_count // 2
        half_span = int(window_seconds * fps / 2)

        start = max(0, mid - half_span)
        end = min(self.metadata.frame_count - 1, mid + half_span)

        n = min(n, end - start + 1)
        indices = np.linspace(start, end, n, dtype=int)
        return self._read_at_indices(indices.tolist())

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_at_indices(self, indices: List[int]) -> List[np.ndarray]:
        """Seek to and read each frame index.  Skips unreadable frames."""
        frames: List[np.ndarray] = []
        for idx in indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = self._cap.read()
            if ret and frame is not None:
                frames.append(frame)
        return frames

    def _read_metadata(self, path: str) -> VideoMetadata:
        cap = self._cap
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        return VideoMetadata(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
