# VR-Score

A forensics-based composite metric for AI-generated video detection.

VR-Score estimates how consistent a video is with real camera capture on
a scale from **0.0** (strongly AI-generated) to **1.0** (strongly real).
It is designed as a complement to deep-learning detector ensembles, not a
replacement.

---

## Metrics

| ID | Name | Signal | Direction | Reference |
|----|------|--------|-----------|-----------|
| M1 | Spatial | Wavelet diagonal (HH) energy ratio | ↑ real | Corvi et al., NeurIPS 2025 |
| M2 | Temporal | Optical-flow background jitter | ↓ real | arXiv:2405.04133 |
| M3 | Noise | Wavelet-residual excess kurtosis (PRNU proxy) | ↑ real | Lukas et al. 2006; PRaNA 2022 |
| M4 | Latent | DINOv2 trajectory curvature + distance variance | ↓ real | Internò et al., NeurIPS 2025 |

Default weights: `[0.20, 0.20, 0.20, 0.40]`.  M4 carries the most weight
because ReStraV (the basis for M4) achieves 97.17 % accuracy / 98.63 %
AUROC as a standalone metric.

---

## Installation

```bash
pip install -r requirements.txt
```

DINOv2 (M4) is downloaded automatically from `torch.hub` on first use
(~350 MB for ViT-S/14).

---

## Usage

### Command line

```bash
# Human-readable output
python run_vr_score.py path/to/video.mp4

# JSON output (for scripting)
python run_vr_score.py path/to/video.mp4 --json

# With GPU acceleration
python run_vr_score.py path/to/video.mp4 --device cuda

# With calibration file
python run_vr_score.py path/to/video.mp4 --calibration calibration.json
```

### Python API

```python
from vr_score import VRScorer

scorer = VRScorer()
result = scorer.score("path/to/video.mp4")

print(f"VR-Score : {result.vr_score:.4f}")
print(f"Spatial  : {result.normalised.spatial:.4f}")
print(f"Temporal : {result.normalised.temporal:.4f}")
print(f"Noise    : {result.normalised.noise:.4f}")
print(f"Latent   : {result.normalised.latent:.4f}")
```

---

## Calibration (recommended)

The default normalisation parameters are conservative heuristics.
For better results, fit the normaliser on a labelled dataset:

```python
from vr_score import VRScorer, MetricNormalizer

# Collect raw metric values for known-real and known-AI videos
scorer = VRScorer()

real_raws = {"spatial": [], "temporal": [], "noise": [],
             "latent_curvature": [], "latent_dist_var": []}
ai_raws   = {k: [] for k in real_raws}

for path in real_video_paths:
    with VideoLoader(path) as loader:
        raw = scorer._compute_raw_metrics(loader)   # internal; for calibration only
    real_raws["spatial"].append(raw.spatial)
    # ... repeat for other keys

# Fit and save
normalizer = MetricNormalizer()
normalizer.fit(real_raws, ai_raws)
normalizer.save("calibration.json")

# Use fitted calibration
calibrated_scorer = VRScorer(normalizer=normalizer)
```

---

## Weight learning (optional)

After calibration, learn composite weights from labelled videos:

```python
real_normed = [calibrated_scorer._normalise(raw) for raw in real_raws_list]
ai_normed   = [calibrated_scorer._normalise(raw) for raw in ai_raws_list]
calibrated_scorer.fit_weights(real_normed, ai_normed)
```

---

## File structure

```
vr_score/
├── __init__.py          Public API surface
├── config.py            All tunable constants
├── video_loader.py      Native-resolution frame sampling
├── normalization.py     Logistic calibration + persistence
├── scorer.py            Composite pipeline (VRScorer)
└── metrics/
    ├── __init__.py
    ├── spatial.py       M1 — wavelet diagonal energy ratio
    ├── temporal.py      M2 — optical-flow background jitter
    ├── noise.py         M3 — wavelet-residual kurtosis
    └── latent.py        M4 — DINOv2 trajectory curvature
run_vr_score.py          CLI entry point
requirements.txt
```
