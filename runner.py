"""
DatasetRunner — orchestrates VR-Score analysis over an entire dataset.

Responsibilities
----------------
1. Discover all video files via ``DatasetCrawler``.
2. Determine which videos to skip based on resume state from any existing
   output file (only videos with ``status == "ok"`` are skipped — failed
   videos are always retried).
3. Run ``VRScorer.score()`` on each video, catching and logging any
   per-video exceptions so a single bad file never aborts the whole run.
4. Write each result *immediately* to the output file so that a crash or
   keyboard interrupt never loses already-computed results.
5. Report a summary once the run completes.

Output formats
--------------
CSV (default)
    One row per video.  Header written on first creation; appended
    to on subsequent writes.  Readable in Excel/pandas with zero conversion.

JSONL (JSON Lines)
    One JSON object per line.  Appendable without loading the whole file
    into memory.  Convert to a standard JSON array with the helper at
    ``scripts/jsonl_to_json.py``.

Progress
--------
If ``tqdm`` is installed it renders a live progress bar with ETA.  If it
is not installed the runner falls back to plain logging lines (one per video).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from vr_score.dataset import DatasetCrawler, VideoEntry
from vr_score.run_config import RunConfig
from vr_score.scorer import VRScorer, VRScoreResult
from vr_score.normalization import MetricNormalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import tqdm; fall back gracefully if absent
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result record — everything written to the output file for one video
# ---------------------------------------------------------------------------

# Ordered list of CSV column names.  Used both for the header row and for
# dict-to-row serialisation so columns never silently drift out of sync.
_CSV_COLUMNS = [
    "video_name",
    "source_model",
    "video_path",
    # composite
    "vr_score",
    # normalised components
    "norm_spatial",
    "norm_temporal",
    "norm_noise",
    "norm_latent",
    # raw values
    "raw_spatial",
    "raw_temporal",
    "raw_noise",
    "raw_latent_curvature",
    "raw_latent_dist_var",
    # weights actually used
    "w_spatial",
    "w_temporal",
    "w_noise",
    "w_latent",
    # bookkeeping
    "status",           # "ok" | "error"
    "error_message",    # empty string on success
    "duration_seconds",
]


@dataclass
class RunRecord:
    """All data associated with one processed video."""

    # Identity
    video_name: str
    source_model: str
    video_path: str

    # Scores (None when status == "error")
    vr_score: Optional[float]
    norm_spatial: Optional[float]
    norm_temporal: Optional[float]
    norm_noise: Optional[float]
    norm_latent: Optional[float]
    raw_spatial: Optional[float]
    raw_temporal: Optional[float]
    raw_noise: Optional[float]
    raw_latent_curvature: Optional[float]
    raw_latent_dist_var: Optional[float]
    w_spatial: Optional[float]
    w_temporal: Optional[float]
    w_noise: Optional[float]
    w_latent: Optional[float]

    # Bookkeeping
    status: str              # "ok" | "error"
    error_message: str       # "" on success
    duration_seconds: float

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_success(
        cls,
        entry: VideoEntry,
        result: VRScoreResult,
        duration: float,
    ) -> "RunRecord":
        return cls(
            video_name=entry.video_name,
            source_model=entry.source_model,
            video_path=entry.video_path,
            vr_score=result.vr_score,
            norm_spatial=result.normalised.spatial,
            norm_temporal=result.normalised.temporal,
            norm_noise=result.normalised.noise,
            norm_latent=result.normalised.latent,
            raw_spatial=result.raw.spatial,
            raw_temporal=result.raw.temporal,
            raw_noise=result.raw.noise,
            raw_latent_curvature=result.raw.latent_curvature,
            raw_latent_dist_var=result.raw.latent_dist_var,
            w_spatial=result.weights["spatial"],
            w_temporal=result.weights["temporal"],
            w_noise=result.weights["noise"],
            w_latent=result.weights["latent"],
            status="ok",
            error_message="",
            duration_seconds=round(duration, 3),
        )

    @classmethod
    def from_error(
        cls,
        entry: VideoEntry,
        error: Exception,
        duration: float,
    ) -> "RunRecord":
        return cls(
            video_name=entry.video_name,
            source_model=entry.source_model,
            video_path=entry.video_path,
            vr_score=None,
            norm_spatial=None,
            norm_temporal=None,
            norm_noise=None,
            norm_latent=None,
            raw_spatial=None,
            raw_temporal=None,
            raw_noise=None,
            raw_latent_curvature=None,
            raw_latent_dist_var=None,
            w_spatial=None,
            w_temporal=None,
            w_noise=None,
            w_latent=None,
            status="error",
            error_message=str(error),
            duration_seconds=round(duration, 3),
        )

    def to_dict(self) -> dict:
        """Return an ordered dict suitable for CSV / JSON serialisation."""
        d = {col: getattr(self, col) for col in _CSV_COLUMNS}
        return d


# ---------------------------------------------------------------------------
# Incremental results writer
# ---------------------------------------------------------------------------

class ResultsWriter:
    """
    Writes ``RunRecord`` objects to an output file immediately upon receipt.

    Supports CSV (default) and JSONL formats.  The file is created if it
    does not exist; existing files are appended to (resume mode).

    Use as a context manager to ensure the file handle is always closed.
    """

    def __init__(self, output_path: Path, fmt: str) -> None:
        self._path = output_path
        self._fmt  = fmt   # "csv" or "jsonl"
        self._fh   = None
        self._csv_writer = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._path.exists() and self._path.stat().st_size > 0

        self._fh = open(self._path, "a", newline="", encoding="utf-8")

        if self._fmt == "csv":
            self._csv_writer = csv.DictWriter(
                self._fh,
                fieldnames=_CSV_COLUMNS,
                extrasaction="ignore",
            )
            # Only write the header when creating the file for the first time
            if not file_exists:
                self._csv_writer.writeheader()
                self._fh.flush()

    def write(self, record: RunRecord) -> None:
        """Write one record immediately and flush to disk."""
        row = record.to_dict()

        if self._fmt == "csv":
            self._csv_writer.writerow(row)
        else:
            # JSONL: replace None with null naturally via json.dumps
            self._fh.write(json.dumps(row) + "\n")

        self._fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "ResultsWriter":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Resume helpers (class-level, no open file handle needed)
    # ------------------------------------------------------------------

    @classmethod
    def load_completed_names(cls, output_path: Path, fmt: str) -> Set[str]:
        """
        Return the set of ``video_name`` values that completed successfully
        in a previous run.

        Only records with ``status == "ok"`` count as completed; failed
        records are excluded so they will be retried.
        """
        if not output_path.exists() or output_path.stat().st_size == 0:
            return set()

        completed: Set[str] = set()
        try:
            if fmt == "csv":
                with open(output_path, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if row.get("status") == "ok":
                            name = row.get("video_name", "").strip()
                            if name:
                                completed.add(name)
            else:
                with open(output_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if obj.get("status") == "ok":
                                name = obj.get("video_name", "").strip()
                                if name:
                                    completed.add(name)
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.warning("Could not read existing output file: %s", exc)

        return completed


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    total_discovered: int
    total_skipped: int
    total_processed: int
    total_ok: int
    total_errors: int
    elapsed_seconds: float

    def log(self) -> None:
        logger.info("=" * 60)
        logger.info("Run complete")
        logger.info("  Discovered : %d", self.total_discovered)
        logger.info("  Skipped    : %d  (already done)", self.total_skipped)
        logger.info("  Processed  : %d", self.total_processed)
        logger.info("  Success    : %d", self.total_ok)
        logger.info("  Errors     : %d", self.total_errors)
        logger.info("  Elapsed    : %.1f s", self.elapsed_seconds)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# DatasetRunner
# ---------------------------------------------------------------------------

class DatasetRunner:
    """
    Runs VR-Score analysis over a full dataset directory.

    Parameters
    ----------
    run_config: Fully populated ``RunConfig`` (loaded from TOML).
    """

    def __init__(self, run_config: RunConfig) -> None:
        self._cfg = run_config
        self._scorer = self._build_scorer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> RunSummary:
        """
        Execute the full pipeline and return a summary.

        This method:
        1. Discovers all videos via ``DatasetCrawler``.
        2. Filters out already-completed videos if ``resume=True``.
        3. Scores each remaining video, writing results incrementally.
        4. Returns a ``RunSummary``.
        """
        run_start = time.perf_counter()

        crawler = DatasetCrawler(
            dataset_dir=self._cfg.dataset.directory,
            video_extensions=self._cfg.dataset.video_extensions,
        )

        logger.info("Scanning dataset: %s", self._cfg.dataset.directory)
        all_entries = list(crawler.crawl())
        total_discovered = len(all_entries)
        logger.info("Found %d video(s) across all models", total_discovered)

        # Determine which videos to skip
        completed: Set[str] = set()
        if self._cfg.resume:
            completed = ResultsWriter.load_completed_names(
                self._cfg.output_path, self._cfg.output.format
            )
            if completed:
                logger.info(
                    "Resuming: %d video(s) already completed, will be skipped",
                    len(completed),
                )

        to_process = [e for e in all_entries if e.video_name not in completed]
        total_skipped   = total_discovered - len(to_process)
        total_processed = 0
        total_ok        = 0
        total_errors    = 0

        logger.info(
            "Processing %d video(s) → output: %s",
            len(to_process),
            self._cfg.output_path,
        )

        with ResultsWriter(self._cfg.output_path, self._cfg.output.format) as writer:
            for entry in self._progress(to_process):
                record = self._process_one(entry)
                writer.write(record)
                total_processed += 1

                if record.status == "ok":
                    total_ok += 1
                else:
                    total_errors += 1

        elapsed = time.perf_counter() - run_start
        summary = RunSummary(
            total_discovered=total_discovered,
            total_skipped=total_skipped,
            total_processed=total_processed,
            total_ok=total_ok,
            total_errors=total_errors,
            elapsed_seconds=elapsed,
        )
        summary.log()
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_one(self, entry: VideoEntry) -> RunRecord:
        """
        Score a single video.  Catches all exceptions so one bad file
        cannot abort the entire run.
        """
        t0 = time.perf_counter()
        try:
            result = self._scorer.score(entry.video_path)
            duration = time.perf_counter() - t0
            logger.debug(
                "[ok]    %-40s  vr=%.4f  (%.1fs)",
                entry.video_name, result.vr_score, duration,
            )
            return RunRecord.from_success(entry, result, duration)

        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - t0
            logger.warning(
                "[error] %-40s  %s  (%.1fs)",
                entry.video_name, exc, duration,
            )
            return RunRecord.from_error(entry, exc, duration)

    def _build_scorer(self) -> VRScorer:
        normalizer = None
        if self._cfg.calibration:
            cal_path = Path(self._cfg.calibration)
            if not cal_path.exists():
                raise FileNotFoundError(
                    f"Calibration file not found: {self._cfg.calibration}"
                )
            normalizer = MetricNormalizer.load(str(cal_path))
            logger.info("Loaded calibration from: %s", cal_path)

        return VRScorer(
            config=self._cfg.scorer,
            normalizer=normalizer,
            device=self._cfg.device_or_none,
        )

    @staticmethod
    def _progress(entries: List[VideoEntry]) -> Iterator[VideoEntry]:
        """
        Wrap the entry list in a tqdm progress bar if available,
        otherwise yield entries while logging each one.
        """
        if _TQDM_AVAILABLE:
            yield from _tqdm(
                entries,
                desc="VR-Score",
                unit="video",
                dynamic_ncols=True,
            )
        else:
            total = len(entries)
            for i, entry in enumerate(entries, start=1):
                logger.info(
                    "[%d/%d] %s / %s",
                    i, total, entry.source_model, entry.video_name,
                )
                yield entry
