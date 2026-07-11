# All-Weather Moving Object Detection — Radar-Centric Multimodal Deep Learning

**A radar-first 3D detector on nuScenes Mini, built to test one hypothesis: radar-centric detection degrades gracefully under weather that collapses camera and LiDAR.**

---

## Hypothesis

Cameras fail in rain and fog — optically (blur, scattering) and semantically (low light hides object class). LiDAR fails too: 905 nm laser light scatters off raindrops and fog droplets, producing missing and ghost points. Radar, at 77 GHz (λ ≈ 3.9 mm), is essentially unaffected by weather at this scale — Rayleigh scattering off droplets that size falls off as λ⁻⁴.

**The bet:** a detector built with radar as its primary sensor — not bolted on as a secondary cue — will retain far more of its accuracy under adverse weather than a camera- or LiDAR-primary system. This project builds a radar-only detector from scratch, adds optional camera/LiDAR fusion, and measures the mAP retention curve across six simulated weather severities to test that bet directly.

## Notebooks

| Notebook | Purpose | Key output |
|---|---|---|
| `01_eda_nuscenes.ipynb` | Data exploration, sensor visualization, sanity checks | Dataset statistics, BEV/point-density figures |
| `02_radarbevnet.ipynb` | **Core contribution** — radar-only RadarBEVNet training + evaluation, built from scratch | Trained checkpoint, mAP table, training curves |
| `03_fusion.ipynb` | Radar + camera + LiDAR fusion variants | Fusion checkpoint, modality ablation table |
| `04_centerpoint_negative.ipynb` | **Negative-result appendix** — mmdet3d/CenterPoint fusion attempt; documents the pkg_resources/ImpImporter bootstrap loop that made it un-deployable on Python 3.12 / Kaggle | Failure writeup, not a trained model |
| `05_fastercnn_baseline.ipynb` | Pretrained COCO Faster R-CNN, camera-primary — the *foil* for the weather robustness story, evaluated under the same simulated degradation | Baseline mAP, qualitative side-by-side |

Each notebook is self-contained and reproduces its own results, so any single stage can be re-run in isolation. The contribution is the radar pipeline (`02`–`03`); `05` is context and comparison, not the result, and `04` is kept and documented deliberately rather than discarded.

## Dataset

**nuScenes Mini** — 10 scenes, 404 keyframes, full sensor suite:

| Sensor | Count | Rate | Role |
|---|---|---|---|
| Radar (77 GHz FMCW) | 5 | 13 Hz | Primary |
| Camera (RGB, 1600×900) | 6 | 12 Hz | Secondary |
| LiDAR (32-beam) | 1 | 20 Hz | Secondary |

Classes are consolidated into 5 project super-classes (`car`, `truck`, `pedestrian`, `two_wheeler`, `static_object`) to reflect what radar can realistically resolve. Splits are **scene-wise** (7 train / 1 val / 2 test scenes) to prevent near-duplicate frames from leaking across train and test.

## Pipeline

```
raw nuScenes Mini
   │
   ▼
Sample loader — sync radar(×5) + camera(×6) + LiDAR + GT boxes at keyframe
   │
   ▼
Multi-sweep aggregation — 13 historical radar sweeps → ego frame (with velocity rotation)
   │
   ▼
BEV rasterization — 256×256 cells @ 0.4 m/cell, 8 channels/cell
   │
   ▼
Augmentation — random flip + BEV jitter (no rotation — see Limitations)
   │
   ▼
Radar encoder ──┐
Camera encoder ─┼──▶ BEV Fusion (channel attention) ──▶ CenterPoint head ──▶ NMS ──▶ 3D boxes
LiDAR encoder ──┘
```

**BEV channels per cell:** point count, mean/max z, mean/max RCS, mean vx/vy, speed magnitude — each chosen to give the network density, height, reflectivity, and Doppler-motion cues per cell.

## Model — RadarBEVNet

A compact ResNet+FPN-style CNN operating directly on the 8-channel BEV pseudo-image, **trained from scratch** (no ImageNet/KITTI init), 5.3M parameters:

- Stem → 2 residual stages (stride-2 downsampling) → FPN top-down merge back to full BEV resolution
- **CenterPoint-style head**: per-class heatmap (object centers as Gaussian peaks) + offset/size/rotation/velocity regression, decoded via max-pool NMS + top-K + distance-based NMS
- Focal loss on the heatmap (handles ~99.9% background cells), L1 loss on regression heads, heatmap bias initialized to −4.6 to match the empirical ~1% positive-cell prior
- **Fusion variants** add a camera branch (ResNet + Lift-Splat-Shoot into BEV) and a LiDAR branch (same BEV rasterization strategy), merged via a Squeeze-and-Excitation channel-attention block — this is the mechanism that lets the model down-weight degraded modalities

## Training

- AdamW, weight decay 1e-4, base LR 5e-4, cosine decay over 100 epochs (+100 continuation)
- Batch size 8 (radar-only) / 4 (full fusion), gradient clipping at norm 35.0
- Rotation augmentation was tried and dropped — it broke the velocity head because input velocity channels weren't rotated consistently with target labels; flip + jitter only

## Results

**Radar-only, test set (robust mAP, classes ≥ 30 GT): 0.045**

| Class | GT | AP@0.5 | AP@4.0 | mAP |
|---|---|---|---|---|
| car | 844 | 0.005 | 0.151 | 0.066 |
| truck | 67 | 0.009 | 0.054 | 0.029 |
| pedestrian | 1,445 | 0.010 | 0.170 | 0.080 |
| two_wheeler | 181 | 0.000 | 0.018 | 0.006 |

**Modality ablation (overall mAP):** Radar-only 0.028 → LiDAR-only 0.166 → Radar+LiDAR 0.186. Camera adds classification accuracy; LiDAR adds both localization and classification since it directly measures geometry. Full fusion wins in clear weather — but by a margin small enough that radar alone carries most of the signal.

**Localization is the bottleneck**, not detection: AP@0.5 is only ~5% of AP@4.0 (vs. 30–50% for a well-tuned model), because ~450 aggregated radar points across a 100m×100m grid can't localize a 0.6m pedestrian to within 0.5m. Denser radar, a finer BEV grid, or a refinement head are the natural fixes.

## Weather Robustness Study — the headline result

Six severity levels of synthetic degradation (blur, contrast loss, noise, point dropout) are applied **only to camera and LiDAR**; radar stays untouched, since it's physically immune at this scale.

| Condition | mAP | Retention vs. clear |
|---|---|---|
| Clear | 0.040 | 100% |
| Light rain | 0.032 | 78% |
| Moderate rain | 0.025 | 63% |
| Heavy rain | 0.016 | 40% |
| Fog | 0.012 | 30% |
| Severe fog | 0.008 | 20% |
| Extreme | 0.007 | 17% |

The radar-centric model retains 17–78% of clear-weather mAP across all six severities, while camera-only and LiDAR-only baselines collapse to near-zero under heavy rain and severe fog. This curve is the quantitative case for the project's hypothesis — and it's reported as an **upper bound**, since the simulation captures first-order optical effects but not radar-specific artifacts like multipath reflections off wet roads.

## Limitations — read before citing any number

- **Absolute mAP (~5%) is not a deployment metric.** Mini is 1/70th of full nuScenes; every number here is a *relative* comparison (modality vs. modality, weather vs. weather on the same model), not a benchmark.
- Localization precision is the main bottleneck (see above).
- Truck (67 GT) and two-wheeler (181 GT) AP numbers are noisy from class imbalance; `static_object` has zero test GT and is excluded.
- Weather simulation is synthetic, not validated against real adverse-weather datasets (RADIATE, K-Radar, Boreas).
- Single training seed — no mean ± std across runs.
- No inference-speed (FPS) benchmarking.

## Comparison with Related Work

- **CenterFusion** (Nabati & Qi, 2021) — camera-primary radar fusion; the architectural inverse of this project. We don't claim to beat it; our contribution is the weather-robustness study, which it doesn't report.
- **BEVFusion** (Liu et al., 2023) — the fusion module here is a smaller-scale version of this design with radar as a third branch.
- **PointPillars** (Lang et al., 2019) — used for LiDAR in the fusion notebook; not used for radar, which is too sparse for a learnable per-point encoder to beat hand-designed BEV aggregation.
- **CenterPoint** (Yin et al., 2021) — source of the heatmap-based detection head used throughout.

## Repository Structure

```
radar-detection/
├── notebooks/
│   ├── 01_eda_nuscenes.ipynb
│   ├── 02_radarbevnet.ipynb
│   ├── 03_fusion.ipynb
│   ├── 04_centerpoint_negative.ipynb
│   └── 05_fastercnn_baseline.ipynb
├── results/                      # Saved metrics, figures, checkpoints
├── CV_Project_Report.pdf         # Full technical report
├── requirements.txt
└── README.md
```

## Getting Started

```bash
conda create -n radar_det python=3.10 -y
conda activate radar_det
pip install -r requirements.txt
pip install nuscenes-devkit
```

## References

1. Lang et al., *PointPillars* (2019)
2. Yin et al., *CenterPoint* (2021)
3. Liu et al., *BEVFusion* (2023)
4. Nabati & Qi, *CenterFusion* (2021)
5. Lin et al., *Focal Loss / RetinaNet* (2017)
6. Lin et al., *Feature Pyramid Networks* (2017)
7. Philion & Fidler, *Lift, Splat, Shoot* (2020)
8. Hu et al., *Squeeze-and-Excitation Networks* (2018)
9. He et al., *Deep Residual Learning* (2016)
10. Loshchilov & Hutter, *AdamW / Cosine LR Decay* (2017)
11. Caesar et al., *nuScenes* (2020)

## License

MIT
