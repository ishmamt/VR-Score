"""
run_dataset.py — run VR-Score analysis over an entire dataset directory.

Usage
-----
    # Use the default config file (run_config.toml in the current directory)
    python run_dataset.py

    # Specify a config file explicitly
    python run_dataset.py --config /path/to/my_config.toml

    # Override individual settings from the command line
    python run_dataset.py --dataset /data/videos --output-dir results/
    python run_dataset.py --device cuda --no-resume
    python run_dataset.py --format jsonl

    # Dry-run: list discovered videos without scoring them
    python run_dataset.py --dry-run

Output
------
Results are written incrementally to the output file (CSV or JSONL) so that
a crash or keyboard interrupt never loses already-computed scores.

Re-running with resume=true (the default) skips videos that previously
completed successfully and retries any that errored.

Exit codes
----------
    0  — all videos processed successfully
    1  — one or more videos failed (see output file for details)
    2  — configuration error (bad path, missing required key, etc.)
"""

import argparse
import logging
import sys
from pathlib import Path

from vr_score.run_config import load_run_config, RunConfig, DatasetConfig, OutputConfig
from vr_score.dataset import DatasetCrawler
from vr_score.runner import DatasetRunner


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_dataset",
        description=(
            "Compute VR-Scores for every video in a structured dataset.\n\n"
            "The dataset must be organised as:\n"
            "  dataset/\n"
            "  ├── ModelA/\n"
            "  │   └── clip.mp4\n"
            "  └── ModelB/\n"
            "      └── clip.mp4\n\n"
            "Results are written to a CSV or JSONL file incrementally."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--config",
        default="run_config.toml",
        metavar="FILE",
        help=(
            "Path to a TOML configuration file.  "
            "Defaults to run_config.toml in the current directory."
        ),
    )

    # CLI overrides — these take precedence over values in the TOML file
    p.add_argument(
        "--dataset",
        default=None,
        metavar="DIR",
        help="Override [dataset] directory from the config file.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Override [output] directory from the config file.",
    )
    p.add_argument(
        "--output-name",
        default=None,
        metavar="NAME",
        help="Override [output] filename (without extension).",
    )
    p.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default=None,
        help="Override [output] format: 'csv' (default) or 'jsonl'.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Override [run] device for DINOv2 ('cpu', 'cuda', 'mps', ...).",
    )
    p.add_argument(
        "--calibration",
        default=None,
        metavar="FILE",
        help="Override [run] calibration path.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing output file and process all videos from scratch.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover and list all videos that would be processed "
            "without actually running the scorer."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Config assembly (TOML + CLI overrides)
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    """
    Return a new ``RunConfig`` with CLI overrides applied on top of the
    values from the TOML file.

    Only keys explicitly provided on the command line are overridden.
    We rebuild only the sub-dataclasses that need to change; everything
    else is passed through unchanged.
    """
    dataset = cfg.dataset
    if args.dataset:
        dataset = DatasetConfig(
            directory=args.dataset,
            video_extensions=cfg.dataset.video_extensions,
        )

    output = cfg.output
    if args.output_dir or args.output_name or args.format:
        output = OutputConfig(
            directory=args.output_dir or cfg.output.directory,
            filename=args.output_name or cfg.output.filename,
            format=args.format or cfg.output.format,
        )

    return RunConfig(
        dataset=dataset,
        output=output,
        device=args.device if args.device is not None else cfg.device,
        calibration=args.calibration if args.calibration is not None else cfg.calibration,
        resume=(not args.no_resume) if args.no_resume else cfg.resume,
        log_level=cfg.log_level,
        scorer=cfg.scorer,
    )


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _dry_run(cfg: RunConfig) -> None:
    """List all videos that would be processed without scoring them."""
    crawler = DatasetCrawler(
        dataset_dir=cfg.dataset.directory,
        video_extensions=cfg.dataset.video_extensions,
    )
    entries = list(crawler.crawl())

    print(f"\nDataset directory : {cfg.dataset.directory}")
    print(f"Total videos found: {len(entries)}\n")

    current_model = None
    for entry in entries:
        if entry.source_model != current_model:
            current_model = entry.source_model
            print(f"  [{current_model}]")
        print(f"    {entry.video_name}")

    print(f"\nOutput would be written to: {cfg.output_path}")
    print("(Dry-run complete — no videos were scored.)\n")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Load config file (must exist unless the user provides --dataset on CLI)
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = load_run_config(str(config_path))
        except Exception as exc:
            print(f"Error reading config file '{config_path}': {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        if args.dataset is None:
            print(
                f"Config file '{config_path}' not found and --dataset not provided.\n"
                "Either create run_config.toml or pass --dataset /path/to/dataset",
                file=sys.stderr,
            )
            sys.exit(2)
        # No TOML file; start from all defaults
        cfg = RunConfig()

    # Apply any CLI overrides
    cfg = _apply_overrides(cfg, args)

    _configure_logging(cfg.log_level)
    logger = logging.getLogger(__name__)

    # Validate that dataset directory was actually set
    if not cfg.dataset.directory:
        logger.error(
            "Dataset directory is not set.  "
            "Add 'directory = ...' under [dataset] in run_config.toml "
            "or pass --dataset on the command line."
        )
        sys.exit(2)

    logger.info("Config file    : %s", config_path if config_path.exists() else "(defaults)")
    logger.info("Dataset dir    : %s", cfg.dataset.directory)
    logger.info("Output path    : %s", cfg.output_path)
    logger.info("Output format  : %s", cfg.output.format)
    logger.info("Device         : %s", cfg.device or "auto")
    logger.info("Resume         : %s", cfg.resume)

    if args.dry_run:
        try:
            _dry_run(cfg)
        except Exception as exc:
            logger.error("Dry-run failed: %s", exc)
            sys.exit(2)
        return

    # Run the pipeline
    try:
        runner  = DatasetRunner(cfg)
        summary = runner.run()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Unexpected error during run: %s", exc)
        sys.exit(2)

    sys.exit(0 if summary.total_errors == 0 else 1)


if __name__ == "__main__":
    main()
