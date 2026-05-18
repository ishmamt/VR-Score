"""
run_vr_score.py — command-line entry point for the VR-Score pipeline.

Usage
-----
    # Basic analysis
    python run_vr_score.py path/to/video.mp4

    # With pre-fitted calibration
    python run_vr_score.py path/to/video.mp4 --calibration calibration.json

    # Force GPU / CPU
    python run_vr_score.py path/to/video.mp4 --device cuda
    python run_vr_score.py path/to/video.mp4 --device cpu

    # Machine-readable JSON output
    python run_vr_score.py path/to/video.mp4 --json

Exit codes
----------
    0  — analysis completed successfully
    1  — input file not found or could not be opened
    2  — calibration file not found
"""

import argparse
import json
import sys
from pathlib import Path

from vr_score import VRScorer
from vr_score.normalization import MetricNormalizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_vr_score",
        description=(
            "Compute the VR-Score for a video file.\n\n"
            "VR-Score is a forensics-based composite metric that estimates\n"
            "how consistent a video is with real camera capture (1.0) versus\n"
            "AI generation (0.0).  It combines four signal-level metrics:\n"
            "  M1 Spatial  — wavelet diagonal energy ratio\n"
            "  M2 Temporal — optical-flow background jitter\n"
            "  M3 Noise    — wavelet-residual kurtosis (PRNU proxy)\n"
            "  M4 Latent   — DINOv2 trajectory curvature (ReStraV)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "video",
        help="Path to the input video file (any codec supported by OpenCV).",
    )
    p.add_argument(
        "--calibration",
        default=None,
        metavar="FILE",
        help=(
            "Path to a calibration JSON file produced by "
            "MetricNormalizer.save().  When omitted, heuristic default "
            "calibration parameters are used."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "PyTorch device for DINOv2 inference "
            "('cpu', 'cuda', 'cuda:1', 'mps').  "
            "Auto-detected when omitted."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit a JSON object instead of human-readable text.",
    )
    return p


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_human(result) -> None:
    bar_width = 40
    filled = int(result.vr_score * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    print()
    print("┌─────────────────────────────────────────────────────┐")
    print(f"│  VR-Score  [{bar}]  {result.vr_score:.4f}  │")
    print("│             0 = AI-generated        1 = Real camera │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("  Normalised components  (0 = AI, 1 = Real)")
    print("  ─────────────────────────────────────────")
    components = [
        ("M1  Spatial  (wavelet diagonal)", result.normalised.spatial),
        ("M2  Temporal (optical-flow jitter)", result.normalised.temporal),
        ("M3  Noise    (PRNU kurtosis)", result.normalised.noise),
        ("M4  Latent   (DINOv2 trajectory)", result.normalised.latent),
    ]
    for label, value in components:
        mini_bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {label:<38}  [{mini_bar}]  {value:.4f}")

    print()
    print("  Raw values")
    print("  ─────────────────────────────────────────")
    print(f"  Diagonal wavelet energy ratio  : {result.raw.spatial:.6f}")
    print(f"  Background jitter variance     : {result.raw.temporal:.2e}")
    print(f"  Noise excess kurtosis          : {result.raw.noise:.4f}")
    print(f"  Trajectory curvature (rad)     : {result.raw.latent_curvature:.4f}")
    print(f"  Step-distance variance         : {result.raw.latent_dist_var:.6f}")

    print()
    print("  Weights used in composite")
    print("  ─────────────────────────────────────────")
    for name, w in result.weights.items():
        print(f"  {name:<10}: {w:.2f}")
    print()


def _print_json(result) -> None:
    output = {
        "vr_score": result.vr_score,
        "normalised": {
            "spatial":  result.normalised.spatial,
            "temporal": result.normalised.temporal,
            "noise":    result.normalised.noise,
            "latent":   result.normalised.latent,
        },
        "raw": {
            "spatial":          result.raw.spatial,
            "temporal":         result.raw.temporal,
            "noise":            result.raw.noise,
            "latent_curvature": result.raw.latent_curvature,
            "latent_dist_var":  result.raw.latent_dist_var,
        },
        "weights": result.weights,
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found — {args.video}", file=sys.stderr)
        sys.exit(1)

    normalizer = None
    if args.calibration:
        cal_path = Path(args.calibration)
        if not cal_path.exists():
            print(
                f"Error: calibration file not found — {args.calibration}",
                file=sys.stderr,
            )
            sys.exit(2)
        normalizer = MetricNormalizer.load(str(cal_path))

    if not args.json_output:
        print(f"Analysing: {video_path}")

    scorer = VRScorer(normalizer=normalizer, device=args.device)
    result = scorer.score(str(video_path))

    if args.json_output:
        _print_json(result)
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
