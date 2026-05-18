"""
vr_score — forensics-based metric for AI-generated video detection.

Single-video API
----------------
>>> from vr_score import VRScorer
>>> scorer = VRScorer()
>>> result = scorer.score("video.mp4")
>>> print(result.vr_score)   # float in [0, 1]; 1.0 = more real

Dataset API
-----------
>>> from vr_score import DatasetRunner
>>> from vr_score.run_config import load_run_config
>>> cfg    = load_run_config("run_config.toml")
>>> runner = DatasetRunner(cfg)
>>> runner.run()
"""

from vr_score.scorer import VRScorer, VRScoreResult, RawMetrics, NormalisedMetrics
from vr_score.normalization import MetricNormalizer
from vr_score.config import VRScoreConfig, DEFAULT_CONFIG
from vr_score.dataset import DatasetCrawler, VideoEntry
from vr_score.runner import DatasetRunner, RunRecord, RunSummary
from vr_score.run_config import RunConfig, load_run_config

__all__ = [
    # Single-video scoring
    "VRScorer",
    "VRScoreResult",
    "RawMetrics",
    "NormalisedMetrics",
    "MetricNormalizer",
    "VRScoreConfig",
    "DEFAULT_CONFIG",
    # Dataset pipeline
    "DatasetCrawler",
    "VideoEntry",
    "DatasetRunner",
    "RunRecord",
    "RunSummary",
    "RunConfig",
    "load_run_config",
]
