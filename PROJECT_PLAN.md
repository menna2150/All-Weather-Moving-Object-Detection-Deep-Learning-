# All-Weather Moving Object Detection Using Radar-Centric Multimodal Deep Learning

## Project Plan & Scope

---

## 1. Project Overview

**Goal**: Build a deep learning system that detects moving objects (vehicles, pedestrians, cyclists, etc.) using **radar as the primary sensor**, optionally fused with camera and LiDAR data. The system must be robust under adverse weather conditions where cameras and LiDAR degrade.

**Dataset**: nuScenes Mini (from Kaggle)
- 10 scenes, ~400 annotated keyframes
- **5 radar sensors**: front, front-left, front-right, back-left, back-right
- **6 cameras**: full 360° surround view
- **1 LiDAR**: 32-beam, 360° point cloud
- **Annotations**: 3D bounding boxes with 10 detection categories
- Categories: car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier

---

## 2. Architecture Design (What Will Impress)

### 2.1 Overall Pipeline: BEV (Bird's Eye View) Fusion Network

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Radar       │────▶│  Radar Encoder   │────▶│                 │
│  Point Cloud │     │  (PointPillars)  │     │                 │
└─────────────┘     └──────────────────┘     │                 │
                                              │  BEV Fusion     │     ┌──────────────┐
┌─────────────┐     ┌──────────────────┐     │  Module         │────▶│  Detection   │
│  Camera      │────▶│  Image Backbone  │────▶│  (Attention-    │     │  Head        │
│  Images      │     │  (ResNet-50)     │     │   based)        │     │  (CenterPoint│
└─────────────┘     └──────────────────┘     │                 │     │   style)     │
                                              │                 │     └──────────────┘
┌─────────────┐     ┌──────────────────┐     │                 │
│  LiDAR       │────▶│  LiDAR Encoder   │────▶│                 │
│  Point Cloud │     │  (PointPillars)  │     │                 │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

### 2.2 Key Components

#### A. Radar Encoder (Primary - PointPillars-based)
- Convert radar point cloud (x, y, z, velocity, RCS) into pillar representation
- Encode pillars using a simplified PointNet
- Scatter to pseudo-image (BEV grid)
- 2D CNN backbone to extract BEV features
- **Why radar-first**: Radar provides velocity (Doppler) and works in rain/fog/snow

#### B. Camera Encoder (Secondary)
- ResNet-50 or EfficientNet backbone for image feature extraction
- LSS (Lift-Splat-Shoot) to project 2D image features into BEV space
- This bridges the camera view to the shared BEV representation

#### C. LiDAR Encoder (Secondary)
- PointPillars to convert LiDAR point cloud to BEV features
- Higher resolution than radar but degrades in weather

#### D. BEV Fusion Module (The Impressive Part)
- **Adaptive Attention Fusion**: Learn which sensor to trust per-region
- Channel-wise attention weights for each modality
- This is where you show the model can "down-weight" degraded sensors
- Cross-attention between radar BEV and camera/LiDAR BEV features

#### E. Detection Head (CenterPoint-style)
- Predict object centers as heatmap peaks
- Regress 3D box attributes: size (w, h, l), rotation, velocity
- Class-specific heads for each detection category
- NMS (Non-Maximum Suppression) for final detections

---

## 3. Use Cases & Scenarios

### Use Case 1: Radar-Only Detection (Baseline)
- **Input**: 5 radar point clouds merged into unified BEV
- **Output**: 3D bounding boxes with class labels
- **Purpose**: Establish baseline; show radar alone can detect objects

### Use Case 2: Radar + Camera Fusion
- **Input**: Radar BEV + 6 camera images projected to BEV
- **Output**: Improved 3D detections with better classification
- **Purpose**: Show cameras improve class discrimination

### Use Case 3: Radar + LiDAR Fusion
- **Input**: Radar BEV + LiDAR BEV
- **Output**: High-accuracy 3D detections
- **Purpose**: Show geometric complementarity

### Use Case 4: Full Fusion (Radar + Camera + LiDAR)
- **Input**: All three modalities
- **Output**: Best-case detections
- **Purpose**: State-of-the-art comparison

### Use Case 5: Weather Robustness Simulation (The "Wow" Factor)
- **Simulate sensor degradation**:
  - Add Gaussian noise to LiDAR points (simulates rain/fog scattering)
  - Apply image corruption to cameras (blur, contrast reduction, noise)
  - Keep radar clean (radar is weather-robust)
- **Show**: Detection accuracy degrades for camera/LiDAR-only but radar-centric fusion remains stable
- **This is what will really impress the professor**

---

## 4. Implementation Milestones

### Phase 1: Data Pipeline & Exploration (Week 1-2)
- [ ] Download and extract nuScenes mini from Kaggle
- [ ] Install nuscenes-devkit (`pip install nuscenes-devkit`)
- [ ] Write data exploration notebook:
  - Visualize radar point clouds in BEV
  - Visualize camera images with projected 3D boxes
  - Visualize LiDAR point clouds
  - Understand annotation format and coordinate systems
- [ ] Build PyTorch Dataset class:
  - Load synchronized radar, camera, LiDAR data per sample
  - Parse 3D bounding box annotations
  - Generate BEV ground truth heatmaps and regression targets
- [ ] Implement data augmentation:
  - Random flip, rotation, scaling in BEV
  - GT-paste (paste ground truth objects from other scenes)

### Phase 2: Radar-Only Model (Week 3-4)
- [ ] Implement PointPillars encoder for radar
  - Pillarize radar points (x, y, z, vx, vy, RCS)
  - PointNet feature extraction per pillar
  - Scatter to BEV pseudo-image
- [ ] Implement 2D BEV backbone (ResNet-style or simple FPN)
- [ ] Implement CenterPoint detection head
  - Heatmap head (Gaussian targets at object centers)
  - Regression heads (offset, size, rotation, velocity)
- [ ] Implement loss functions:
  - Focal loss for heatmap
  - L1 loss for regression
- [ ] Train radar-only model
- [ ] Evaluate with nuScenes metrics (mAP, NDS)
- [ ] Visualize radar-only detections in BEV

### Phase 3: Multimodal Fusion (Week 5-6)
- [ ] Implement camera encoder:
  - ResNet-50 backbone (pretrained ImageNet)
  - LSS view transformation (image → BEV)
- [ ] Implement LiDAR encoder:
  - PointPillars (same architecture, different input features)
- [ ] Implement BEV Fusion module:
  - Concatenation baseline
  - Channel attention fusion (SE-Net style)
  - Cross-attention fusion (transformer-based)
- [ ] Train Radar+Camera model
- [ ] Train Radar+LiDAR model
- [ ] Train Radar+Camera+LiDAR model

### Phase 4: Weather Robustness & Ablations (Week 7-8)
- [ ] Implement weather simulation:
  - **Rain/Fog on LiDAR**: Random point dropout + Gaussian noise on z/intensity
  - **Rain/Fog on Camera**: Gaussian blur, reduced contrast, additive noise, synthetic fog overlay
  - **Radar**: Keep clean (it's weather-invariant)
- [ ] Run experiments at multiple degradation levels (mild, moderate, severe)
- [ ] Build comparison table: performance vs. weather severity per modality
- [ ] Ablation studies:
  - Effect of radar velocity features
  - Effect of fusion strategy (concat vs. attention)
  - Per-class detection performance
  - Detection range analysis (near vs. far)

### Phase 5: Visualization & Report (Week 9-10)
- [ ] Create impressive visualizations:
  - BEV detection plots with ground truth vs. predictions
  - 3D bounding boxes overlaid on camera images
  - Radar point cloud colored by velocity
  - Attention weight heatmaps (which sensor the model trusts where)
  - Performance degradation curves under simulated weather
- [ ] Write final report:
  - Introduction & motivation
  - Related work (cite BEVFusion, CenterFusion, RadarNet, CenterPoint)
  - Methodology (architecture diagrams)
  - Experiments & results (tables + plots)
  - Weather robustness analysis
  - Conclusion & future work
- [ ] Prepare demo/presentation

---

## 5. Tech Stack

| Component        | Tool/Library                          |
|------------------|---------------------------------------|
| Language         | Python 3.10+                          |
| Deep Learning    | PyTorch 2.x + torchvision            |
| Dataset Toolkit  | nuscenes-devkit                       |
| Point Cloud      | Open3D, PyTorch3D                     |
| Visualization    | matplotlib, Open3D, cv2              |
| Metrics          | nuscenes-devkit eval tools            |
| Experiment Track | TensorBoard or Weights & Biases      |
| Notebooks        | Jupyter (for exploration & demos)     |
| GPU              | CUDA-compatible (Google Colab if needed) |

---

## 6. Evaluation Metrics

| Metric              | Description                                          |
|----------------------|------------------------------------------------------|
| **mAP**             | Mean Average Precision across classes (primary)      |
| **NDS**             | nuScenes Detection Score (composite metric)          |
| **mATE**            | Mean Average Translation Error                       |
| **mASE**            | Mean Average Scale Error                             |
| **mAOE**            | Mean Average Orientation Error                       |
| **mAVE**            | Mean Average Velocity Error                          |
| **Precision/Recall**| Per-class precision and recall curves                |

---

## 7. What Will Impress the Professor

### 7.1 Technical Depth
- **Attention-based fusion** instead of simple concatenation — shows you understand modern deep learning
- **PointPillars from scratch** (not just using someone's pretrained model) — shows engineering skill
- **Proper evaluation** using official nuScenes metrics, not just accuracy

### 7.2 Scientific Rigor
- **Ablation studies**: Systematically show what each component contributes
- **Modality comparison**: Clear table showing Radar-only vs. R+C vs. R+L vs. R+C+L
- **Weather robustness**: Quantitative proof that radar-centric fusion is weather-robust
- **Statistical significance**: Run multiple seeds, report mean ± std

### 7.3 Presentation Quality
- **Architecture diagrams**: Clean, publication-quality figures
- **BEV visualizations**: Colorful, clear detection plots
- **Attention heatmaps**: Show the model learning to trust radar in degraded conditions
- **Performance curves**: mAP vs. weather severity — the money plot

### 7.4 Going Beyond Requirements
- **Temporal modeling**: Use radar velocity + multi-frame aggregation for tracking
- **Real-time inference speed**: Report FPS, show it's practical
- **Error analysis**: Where does the model fail? Why? What would fix it?
- **Comparison with published results**: Even on mini, compare trends with papers

---

## 8. Project Structure

```
project/
├── PROJECT_PLAN.md              # This file
├── configs/                     # Training configs (YAML)
│   ├── radar_only.yaml
│   ├── radar_camera.yaml
│   ├── radar_lidar.yaml
│   └── full_fusion.yaml
├── data/                        # nuScenes mini data (gitignored)
│   └── nuscenes_mini/
├── datasets/                    # PyTorch dataset classes
│   ├── __init__.py
│   ├── nuscenes_dataset.py
│   └── transforms.py
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── encoders/
│   │   ├── radar_encoder.py     # PointPillars for radar
│   │   ├── camera_encoder.py    # ResNet + LSS
│   │   └── lidar_encoder.py     # PointPillars for LiDAR
│   ├── fusion/
│   │   ├── bev_fusion.py        # BEV fusion module
│   │   └── attention.py         # Attention mechanisms
│   └── heads/
│       └── centerpoint_head.py  # Detection head
├── utils/                       # Helpers
│   ├── losses.py
│   ├── metrics.py
│   ├── visualization.py
│   └── weather_simulation.py    # Weather degradation simulator
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_results_visualization.ipynb
├── scripts/                     # Training & eval scripts
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── results/                     # Saved results & plots
├── checkpoints/                 # Saved model weights
├── requirements.txt
└── README.md
```

---

## 9. Key References to Cite

1. **PointPillars** (Lang et al., 2019) — pillar-based point cloud encoding
2. **CenterPoint** (Yin et al., 2021) — center-based 3D detection
3. **BEVFusion** (Liu et al., 2023) — multi-modal BEV fusion
4. **CenterFusion** (Nabati & Qi, 2021) — radar-camera fusion
5. **RadarNet** (Yang et al., 2020) — radar-centric detection
6. **LSS - Lift, Splat, Shoot** (Philion & Fidler, 2020) — camera to BEV projection
7. **nuScenes** (Caesar et al., 2020) — dataset paper

---

## 10. Risk Mitigation

| Risk                                  | Mitigation                                          |
|---------------------------------------|-----------------------------------------------------|
| nuScenes mini is too small (10 scenes)| Use aggressive augmentation; focus on methodology   |
| Radar is very sparse (~35 points/frame)| Use multi-sweep aggregation (stack 10 sweeps)      |
| Camera→BEV (LSS) is complex          | Start simple; can use ground-plane projection first |
| Training time on limited GPU          | Use Google Colab Pro; small BEV grid (200x200)     |
| Low mAP numbers on mini              | Emphasize relative comparisons, not absolute mAP   |
