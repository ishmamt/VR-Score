# VR-Score

A forensics-based composite metric for AI-generated video detection.

VR-Score estimates how consistent a video is with real camera capture, scoring from **0.0** (strongly AI-generated) to **1.0** (strongly real). It combines four signal-level metrics derived from peer-reviewed forensics research, and is designed to complement deep-learning detector ensembles rather than replace them.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Running on a dataset](#running-on-a-dataset)
- [Running on a single video](#running-on-a-single-video)
- [Python API](#python-api)
- [Calibration](#calibration)
- [Weight learning](#weight-learning)
- [Output format](#output-format)
- [References](#references)

---

## How it works

VR-Score computes four independent forensic metrics and combines them into a weighted composite score.

| ID | Metric | Signal measured | Interpretation | Reference |
|----|--------|-----------------|----------------|-----------|
| M1 | Spatial | Wavelet diagonal (HH) energy ratio | AI generators leave quasi-periodic spectral artefacts from their up-sampling layers; real video has smooth 1/f diagonal decay | Corvi et al., NeurIPS 2025 |
| M2 | Temporal | Optical-flow background jitter | Real camera backgrounds have near-zero inter-frame motion variance; AI video backgrounds shimmer even when "static" | arXiv:2405.04133 |
| M3 | Noise | Wavelet-residual excess kurtosis | Real sensors produce heavy-tailed (high kurtosis) PRNU noise; AI generators produce near-Gaussian noise | Lukas et al. 2006; PRaNA 2022 |
| M4 | Latent | DINOv2 trajectory curvature + distance variance | Real video traces straight paths in DINOv2 representation space; AI video traces curved, erratic paths (ReStraV) | Internò et al., NeurIPS 2025 |

**Composite formula:**

```
VR-Score = 0.20 · S  +  0.20 · T  +  0.20 · N  +  0.40 · L
```

where each component is normalised to [0, 1]. M4 carries the highest default weight because ReStraV achieves 97.17% accuracy / 98.63% AUROC as a standalone metric. All weights are tunable and can be learned from labelled data.

**Important:** Analysis is always performed at the video's native resolution. Downsampling before forensic analysis destroys the high-frequency cues that differentiate AI-generated from real content.

---

## Requirements

- Python 3.9 or later
- A CUDA-capable GPU is strongly recommended for M4 (DINOv2). CPU inference works but is significantly slower on large datasets.
- ~350 MB of disk space for the DINOv2 ViT-S/14 model weights, downloaded automatically on first use.

---

## Installation

**1. Clone the repository**

```bash
git clone <repo-url>
cd vr-score
```

**2. Create and activate a virtual environment** (recommended)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

On Python 3.10 or earlier, this also installs `tomli` as a TOML parser backport. Python 3.11+ uses the built-in `tomllib`.

**4. Verify the installation**

```bash
python -c "from vr_score import VRScorer; print('OK')"
```

**PyTorch with CUDA** — if the above installs a CPU-only PyTorch build, install the CUDA build manually first:

```bash
# Example for CUDA 12.1 — check https://pytorch.org for your exact command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Project structure

```
vr-score/
│
├── run_config.toml          ← Edit this to point at your dataset
├── run_dataset.py           ← Entry point: score an entire dataset
├── run_vr_score.py          ← Entry point: score a single video
├── requirements.txt
│
├── vr_score/                ← Package
│   ├── __init__.py          Public API surface
│   │
│   ├── config.py            Scoring algorithm hyper-parameters (frozen dataclasses)
│   ├── video_loader.py      Native-resolution frame sampling (context manager)
│   ├── normalization.py     Logistic calibration: raw → [0,1], with save/load
│   ├── scorer.py            VRScorer — wires all metrics into one score() call
│   │
│   ├── run_config.py        Dataset/run infrastructure config + TOML loader
│   ├── dataset.py           Directory crawler → VideoEntry objects
│   ├── runner.py            DatasetRunner — orchestration, resume, incremental writes
│   │
│   └── metrics/
│       ├── spatial.py       M1 — wavelet diagonal energy ratio
│       ├── temporal.py      M2 — optical-flow background jitter
│       ├── noise.py         M3 — wavelet-residual kurtosis (PRNU proxy)
│       └── latent.py        M4 — DINOv2 trajectory curvature (ReStraV)
│
└── scripts/
    └── jsonl_to_json.py     Converts JSONL output to a standard JSON array
```

---

## Running on a dataset

### 1. Set up the config file

Open `run_config.toml` and set the `directory` field under `[dataset]`:

```toml
[dataset]
directory = "/path/to/your/dataset"
```

The dataset must be organised with one sub-folder per source model. Each sub-folder name becomes the `source_model` label in the output. Video filenames must be unique across the entire dataset.

```
dataset/
├── Sora/
│   ├── video_001.mp4
│   └── video_002.mp4
├── Sora2/
│   └── video_003.mp4
└── Kling/
    └── video_004.mp4
```

Everything else in `run_config.toml` has a sensible default and does not need to change for a first run.

### 2. Verify discovery before scoring

Use the dry-run flag to confirm the crawler finds the expected videos without running the scorer:

```bash
python run_dataset.py --dry-run
```

This prints every video grouped by model and shows the output path. No scoring is performed and no files are written.

### 3. Run the full dataset

```bash
python run_dataset.py
```

Results are written to `results/vr_scores.csv` incrementally — one row is flushed to disk immediately after each video completes. If the process is interrupted, re-running the same command resumes from where it left off (resume is on by default).

### Common command-line options

```bash
# Use a different config file
python run_dataset.py --config /path/to/other_config.toml

# Override the dataset directory without editing the config file
python run_dataset.py --dataset /path/to/dataset

# Force GPU or CPU for DINOv2
python run_dataset.py --device cuda
python run_dataset.py --device cpu

# Start from scratch, ignoring any existing output file
python run_dataset.py --no-resume

# Save as JSON Lines instead of CSV
python run_dataset.py --format jsonl

# Combine overrides freely — they take precedence over run_config.toml
python run_dataset.py --dataset /data/videos --device cuda --output-dir /results/run1
```

### Full `run_config.toml` reference

```toml
[dataset]
directory         = "/path/to/dataset"
video_extensions  = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]

[output]
directory  = "results"       # Created automatically if absent
filename   = "vr_scores"     # Extension appended based on format
format     = "csv"           # "csv" or "jsonl"

[run]
device      = ""             # "cuda", "cpu", "mps", or "" (auto-detect)
calibration = ""             # Path to calibration.json, or "" for defaults
resume      = true           # Skip already-completed videos on re-run
log_level   = "INFO"         # "DEBUG", "INFO", "WARNING", "ERROR"

[scorer.sampling]
spatial_frames        = 8    # Frames for M1 and M3 (uniform across video)
noise_frames          = 8
temporal_frames       = 60   # Consecutive frames for M2 optical flow
latent_frames         = 24   # Frames for M4 DINOv2 trajectory
latent_window_seconds = 2.0  # Duration of the central sampling window (M4)

[scorer.weights]
spatial  = 0.20
temporal = 0.20
noise    = 0.20
latent   = 0.40
```

---

## Running on a single video

```bash
# Human-readable output
python run_vr_score.py path/to/video.mp4

# Machine-readable JSON (useful for scripting)
python run_vr_score.py path/to/video.mp4 --json

# With GPU acceleration
python run_vr_score.py path/to/video.mp4 --device cuda

# With a fitted calibration file
python run_vr_score.py path/to/video.mp4 --calibration calibration.json
```

Example output:

```
┌─────────────────────────────────────────────────────┐
│  VR-Score  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.2143  │
│             0 = AI-generated        1 = Real camera │
└─────────────────────────────────────────────────────┘

  Normalised components  (0 = AI, 1 = Real)
  ─────────────────────────────────────────
  M1  Spatial  (wavelet diagonal)       [████████░░░░░░░░░░░░]  0.3812
  M2  Temporal (optical-flow jitter)    [██████░░░░░░░░░░░░░░]  0.2954
  M3  Noise    (PRNU kurtosis)          [████░░░░░░░░░░░░░░░░]  0.1873
  M4  Latent   (DINOv2 trajectory)      [████░░░░░░░░░░░░░░░░]  0.1441
```

---

## Python API

### Score a single video

```python
from vr_score import VRScorer

scorer = VRScorer()
result = scorer.score("path/to/video.mp4")

print(f"VR-Score : {result.vr_score:.4f}")

# Per-component normalised scores (all in [0, 1], higher = more real)
print(f"M1 Spatial  : {result.normalised.spatial:.4f}")
print(f"M2 Temporal : {result.normalised.temporal:.4f}")
print(f"M3 Noise    : {result.normalised.noise:.4f}")
print(f"M4 Latent   : {result.normalised.latent:.4f}")

# Raw un-normalised values (for diagnostics)
print(f"Raw spatial          : {result.raw.spatial:.6f}")
print(f"Raw temporal jitter  : {result.raw.temporal:.2e}")
print(f"Raw noise kurtosis   : {result.raw.noise:.4f}")
print(f"Raw latent curvature : {result.raw.latent_curvature:.4f}")
print(f"Raw latent dist var  : {result.raw.latent_dist_var:.6f}")
```

### Run the dataset pipeline programmatically

```python
from vr_score import DatasetRunner, load_run_config

cfg     = load_run_config("run_config.toml")
runner  = DatasetRunner(cfg)
summary = runner.run()

print(f"Processed : {summary.total_processed}")
print(f"Success   : {summary.total_ok}")
print(f"Errors    : {summary.total_errors}")
```

### Inspect the crawler without running the scorer

```python
from vr_score import DatasetCrawler

crawler = DatasetCrawler(
    dataset_dir="/path/to/dataset",
    video_extensions=[".mp4", ".mov"],
)

for entry in crawler.crawl():
    print(entry.source_model, entry.video_name, entry.video_path)

print(f"Total: {crawler.count()} videos")
```

### Specify a device

```python
scorer = VRScorer(device="cuda")    # single GPU
scorer = VRScorer(device="cuda:1")  # second GPU
scorer = VRScorer(device="mps")     # Apple Silicon
scorer = VRScorer(device="cpu")     # force CPU
```

---

## Calibration

The default normalisation parameters are conservative heuristics. Once you have a set of labelled real and AI videos, fitting calibration to your data will improve score accuracy.

### Collect raw metrics from labelled videos

```python
from vr_score import VRScorer, MetricNormalizer
from vr_score.video_loader import VideoLoader

scorer = VRScorer()

keys      = ["spatial", "temporal", "noise", "latent_curvature", "latent_dist_var"]
real_raws = {k: [] for k in keys}
ai_raws   = {k: [] for k in keys}

def collect(paths, bucket):
    for path in paths:
        with VideoLoader(path) as loader:
            raw = scorer._compute_raw_metrics(loader)
        bucket["spatial"].append(raw.spatial)
        bucket["temporal"].append(raw.temporal)
        bucket["noise"].append(raw.noise)
        bucket["latent_curvature"].append(raw.latent_curvature)
        bucket["latent_dist_var"].append(raw.latent_dist_var)

collect(real_video_paths, real_raws)
collect(ai_video_paths,   ai_raws)
```

### Fit and save

```python
normalizer = MetricNormalizer()
normalizer.fit(real_raws, ai_raws)
normalizer.save("calibration.json")
```

### Use the saved calibration

```bash
# Single video CLI
python run_vr_score.py video.mp4 --calibration calibration.json

# Dataset CLI
python run_dataset.py --calibration calibration.json

# Or set it once in run_config.toml
# calibration = "calibration.json"
```

```python
# Python API
normalizer = MetricNormalizer.load("calibration.json")
scorer     = VRScorer(normalizer=normalizer)
```

---

## Weight learning

After calibration, you can learn the composite weights from labelled data using Non-Negative Least Squares, replacing the default `[0.20, 0.20, 0.20, 0.40]` with weights optimised for your dataset.

```python
from vr_score import VRScorer, MetricNormalizer

normalizer = MetricNormalizer.load("calibration.json")
scorer     = VRScorer(normalizer=normalizer)

real_normed = [scorer.score(p).normalised for p in real_video_paths]
ai_normed   = [scorer.score(p).normalised for p in ai_video_paths]

# Modifies the scorer in place; score() uses the new weights immediately
scorer.fit_weights(real_normed, ai_normed)

print(scorer._weights)
```

---

## Output format

The dataset runner writes one record per video. The columns are:

| Column | Description |
|--------|-------------|
| `video_name` | Filename — the primary key (e.g. `clip_001.mp4`) |
| `source_model` | Parent folder name (e.g. `Sora`) |
| `video_path` | Full absolute path to the video file |
| `vr_score` | Final composite VR-Score in [0, 1] |
| `norm_spatial` | Normalised M1 score |
| `norm_temporal` | Normalised M2 score |
| `norm_noise` | Normalised M3 score |
| `norm_latent` | Normalised M4 score |
| `raw_spatial` | Raw diagonal wavelet energy ratio |
| `raw_temporal` | Raw background jitter variance |
| `raw_noise` | Raw noise excess kurtosis |
| `raw_latent_curvature` | Raw DINOv2 trajectory curvature (radians) |
| `raw_latent_dist_var` | Raw step-distance variance in embedding space |
| `w_spatial` / `w_temporal` / `w_noise` / `w_latent` | Weights used in this run |
| `status` | `ok` or `error` |
| `error_message` | Exception text if `status == error`, empty otherwise |
| `duration_seconds` | Wall-clock time for this video |

### Analysing results in pandas

```python
import pandas as pd

df = pd.read_csv("results/vr_scores.csv")

# Per-model summary statistics
print(df.groupby("source_model")["vr_score"].describe())

# Only successfully scored rows
ok = df[df["status"] == "ok"]

# Videos that errored — inspect and retry
errors = df[df["status"] == "error"][["video_name", "source_model", "error_message"]]
print(errors)
```

### Converting JSONL to a JSON array

If you used `format = "jsonl"`, convert the output with:

```bash
python scripts/jsonl_to_json.py results/vr_scores.jsonl
# writes results/vr_scores.json

python scripts/jsonl_to_json.py results/vr_scores.jsonl --pretty
# pretty-printed

python scripts/jsonl_to_json.py results/vr_scores.jsonl --out /tmp/out.json
# explicit output path
```

---

## References

Corvi R., Cozzolino D., Prashnani E., De Mello S., Nagano K., Verdoliva L. "Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation." NeurIPS 2025. arXiv:2506.16802.

Internò C., Geirhos R., Olhofer M., Liu S., Hammer B., Klindt D. "AI-Generated Video Detection via Perceptual Straightening." NeurIPS 2025. arXiv:2507.00583.

Lukas J., Fridrich J., Goljan M. "Digital Camera Identification from Sensor Pattern Noise." IEEE Transactions on Information Forensics and Security, 2006.

Mandelli S., Bestagini P., Tubaro S. "PRaNA: PRNU-based Technique to Tell Real and Deepfake Videos Apart." ICIP, 2022.

"Exposing AI-generated Videos: A Benchmark Dataset and Detection Framework Based on Local and Global Temporal Defects." IEEE Access, 2024. arXiv:2405.04133.