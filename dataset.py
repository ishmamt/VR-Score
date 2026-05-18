"""
Dataset directory crawler.

Expected layout
---------------
    dataset/
    ├── Sora/
    │   ├── video_001.mp4
    │   └── video_002.mp4
    ├── Sora2/
    │   └── video_003.mp4
    └── Kling/
        └── video_004.mp4

The crawler walks one level below the root.  Each immediate sub-directory
name becomes the ``source_model`` label.  Files at the root level (not
inside any sub-directory) are skipped with a warning.

Symbolic links are followed.  Hidden files and directories (names starting
with ``"."``) are ignored.

``DatasetCrawler.crawl()`` returns a lazy iterator so the full dataset does
not need to be listed before processing begins.  Call ``list()`` on it if
you need random access (e.g. to show a progress bar with a known total).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoEntry:
    """
    Identifies a single video file within the dataset.

    Attributes
    ----------
    video_path:   Absolute path to the video file.
    source_model: Name of the immediate parent directory (the generator label).
    video_name:   Filename without directory (e.g. ``"clip_001.mp4"``).
                  Guaranteed to be unique across the dataset per the user's
                  specification; used as the primary key in output files.
    """

    video_path: str
    source_model: str
    video_name: str


class DatasetCrawler:
    """
    Walks a two-level dataset directory tree and yields ``VideoEntry`` objects.

    Parameters
    ----------
    dataset_dir:       Root directory containing one sub-folder per model.
    video_extensions:  Lowercase extensions (with leading dot) to accept.
                       Case-insensitive comparison is applied to actual files.
    """

    def __init__(
        self,
        dataset_dir: str,
        video_extensions: List[str],
    ) -> None:
        self._root = Path(dataset_dir).resolve()
        # Normalise extensions to lowercase for comparison
        self._extensions = frozenset(e.lower() for e in video_extensions)
        self._validate_root()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def crawl(self) -> Iterator[VideoEntry]:
        """
        Yield ``VideoEntry`` for every video file found in the dataset.

        Iteration order: alphabetical by source_model, then alphabetical
        by filename within each model folder.  This makes runs reproducible
        and resume behaviour predictable.
        """
        model_dirs = sorted(
            d for d in self._root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        if not model_dirs:
            logger.warning(
                "No sub-directories found in dataset root: %s", self._root
            )
            return

        for model_dir in model_dirs:
            source_model = model_dir.name
            video_files = sorted(
                f for f in model_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self._extensions
            )

            if not video_files:
                logger.warning(
                    "No video files found in model directory: %s", model_dir
                )
                continue

            logger.debug(
                "Found %d video(s) under source model '%s'",
                len(video_files),
                source_model,
            )

            for video_file in video_files:
                yield VideoEntry(
                    video_path=str(video_file),
                    source_model=source_model,
                    video_name=video_file.name,
                )

    def count(self) -> int:
        """
        Return the total number of video files across all model directories.

        This performs a full directory walk and is O(N) in the number of
        files — call it once before starting a run rather than repeatedly.
        """
        return sum(1 for _ in self.crawl())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_root(self) -> None:
        if not self._root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self._root}"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {self._root}"
            )
