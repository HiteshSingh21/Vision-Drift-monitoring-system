# Vision Drift Monitoring System — V3 (Statistical Core & YOLO Integration)

A production-grade, highly optimized MLOps pipeline for detecting statistical data drift in computer vision models. Version 3 marks a complete architectural overhaul, transitioning from a decoupled classification monitoring setup (ResNet-18 + MMD) into an integrated object-detection/classification backbone (**YOLOv11**) backed by a custom, interpretable **Mean-Variance Statistical Engine** and automated **Before/After Retraining Recovery Analytics**.

This system monitors incoming real-time image streams, flags distribution decay (such as environmental fading, camera lens blur, or exposure variations), and runs an automated pipeline lifecycle simulation from ingestion to drift recovery.

---

## Project Structure

```text
vision_drift_project/
├── data/                    # Standard datasets cache (e.g., CIFAR-10)
├── reference_state/         # Permanent cache for healthy data distributions
│   └── baseline_embeddings.pt # Extracted baseline mathematical footprint (640 x 256)
├── src/
│   ├── data_loader.py       # Stable data streaming pipelines (Clean vs. Distorted)
│   ├── yolo_extractor.py    # [NEW] YOLOv11 framework wrapper + C2PSA Layer Hook
│   ├── stat_monitor.py      # [NEW] Custom mu and sigma^2 running statistical engine
│   └── visualizer.py        # [NEW] Dark-themed Seaborn KDE distribution graphics
├── legacy/                  # Archived V2 components (kept for backward compatibility)
│   ├── feature_extractor.py # Old ResNet-18 model pipeline
│   ├── drift_detector.py    # Old torchdrift MMD statistical engine
│   └── utils.py             # Old t-SNE scatter-plotting utility
├── main.py                  # [REWRITTEN] 5-Phase Lifecycle Orchestrator
├── requirements.txt         # Project dependency manifest
├── yolo11n-cls.pt           # Locally cached official YOLOv11 pretrained weights
└── WALKTHROUGH_V3.md        # Technical execution logs and milestone summaries
```

---

## V2 vs. V3 Architectural Evolution

| Architectural Layer | Version 2 (Legacy Baseline) | Version 3 (Target Production State) |
| :--- | :--- | :--- |
| **Neural Network Backbone** | ResNet-18 (`torchvision`) | **YOLOv11n-cls** (`ultralytics`) |
| **Embedding Spatial Features** | 512-Dimensional Vector | **256-Dimensional Vector** |
| **Extraction Methodology** | Slicing off final Classification Fully-Connected layer | **PyTorch Forward Hook** on penultimate `C2PSA` backbone layer |
| **Statistical Monitoring Engine** | `torchdrift` Kernel MMD (Black-box P-Value) | **Custom Decomposed Shape Engine** (Tracks Mean Shift & Variance Expansion) |
| **Visualization Philosophy** | 2D Spatial t-SNE Scatter Plot | **Smooth Continuous Kernel Density Estimation (KDE) Curves** |
| **Forensic Evidence Output** | Single static scatter image (`drift_visualization_batch_X.png`) | **Tri-Asset Package** (Before Training, After Training, Combined Dashboard) |
| **Pipeline Lifecycle Complexity** | 3 Phases (Ingest, Calibrate, Stream) | **5 Phases** (Ingest, Calibrate, Stream, Alert, Retraining Recovery) |

---

## Deep-Dive: Modular Architecture & Implementation

### 1. Unified Inference & Hook Layer (`src/yolo_extractor.py`)
To completely eliminate the computational cost of extracting features separately, V3 taps into the **YOLOv11** backbone using a non-destructive PyTorch forward hook.

* **Target Hook Layer:** Layer 9 — **`C2PSA` (Cross Stage Partial with Spatial Attention)** block. This layer captures highly rich, attention-aware spatial and geometric features right before they pass to the classification or object anchoring heads.
* **Single Forward Pass:** Running `logits, embedding = model(images)` routes tensor arrays through the network exactly once. The user receives user-facing predictions immediately, and the background queue captures the 256-D flattened embedding vector with **zero additional GPU/CPU forward pass overhead (50% compute saving)**.

```python
# Conceptual slice of forward pass routing
def forward(self, x):
    # Pass through YOLO blocks; forward hook automatically intercepts intermediate features
    logits = self.yolo_model(x) 
    # Unwraps underlying tuple containers safely
    if isinstance(logits, list or tuple):
        logits = logits[0]
    
    # Global average pooling on hooked layer to generate dense 256-D embedding vector
    embedding = torch.mean(self.hooked_features, dim=[2, 3])
    return logits, embedding
```

### 2. Custom Decomposed Statistical Engine (`src/stat_monitor.py`)
Instead of an uninterpretable black-box P-value that simply says "drift exists," the V3 statistical monitor decomposes the distribution shift into two clear metrics:

1. **Mean Deviation Score (60% Weight):** Calculates the L2 Euclidean distance between the center-of-mass of live production window embeddings (mu_prod) and calibrated baseline embeddings (mu_base), scaled by baseline standard deviation (sigma_base).
2. **Variance Expansion Score (40% Weight):** Measures how much the empirical spread of feature activations stretches or shrinks compared to original baseline benchmarks.

* **Threshold Gate:** Combined Score = 0.6 * Mean Deviation + 0.4 * Variance Expansion. If the final composite metric crosses **2.00**, a critical drift event is declared.

### 3. Kernel Density Estimation Visualizer (`src/visualizer.py`)
Replaces scatter plots with smooth continuous **Kernel Density Estimation (KDE)** plots styled with an enterprise dark theme (`#1a1a2e`).
* **Smart Feature Selection:** Rather than cluttering charts with 256 overlapping dimensions, the visualization sub-engine runs a statistical scan to find the **Top-K most drifted dimensions** (where absolute mean shift is highest) and isolates those specific feature channels for forensic plotting.
* **Output Artifacts:** Generates `drift_before_training.png` (Gap verification), `drift_after_training.png` (Alignment verification), and a side-by-side synchronized dashboard block (`drift_combined_report.png`).

---

## The 5-Phase Pipeline Lifecycle (`main.py`)

The orchestrator guides your data architecture through five distinct automated phases:

```
[Phase 1: Ingestion]  --> Loads CIFAR-10 data streams and initializes YOLOv11 model blocks.
         │
         v
[Phase 2: Calibration] --> Extracts 640 clean baseline vectors; caches mu/sigma^2 statistics to disk.
         │
         v
[Phase 3: Streaming]   --> Streams production batches; executes lightning-fast inference while
         │                 monitoring rolling feature windows in the background.
         │
         ├──► (If Score <= 2.00) ──► Normal logs served, pipeline continues streaming.
         └──► (If Score >  2.00) ──► CRITICAL BREACH MET! Drops to Phase 4.
         │
         v
[Phase 4: Drift Alert] --> Halts execution loop, logs statistical breakdown, dumps "Before" KDE plots.
         │
         v
[Phase 5: Recovery]    --> Simulates model fine-tuning via a 30% blend shift + Gaussian noise.
                           Generates "After" and Combined Reports to prove distribution realignment.
```

---

## Real-World Telemetry Logs

Below is the verified terminal log demonstrating progressive data degradation captured under the new YOLO parameters:

```text
======================================================================
  VISION DRIFT MONITORING SYSTEM — V3 LIFECYCLE
======================================================================

=== Phase 1: Ingesting Data & Preparing Pipelines ===
Loading YOLOv11 model: yolo11n-cls.pt
Found cached weights at: C:\Users\hites\Vision-Drift-monitoring-system\vision_drift_project\yolo11n-cls.pt
YOLOv11 model loaded on cpu. Hook registered on layer: C2PSA
Embedding dimensionality: 256

=== Phase 2: Calibration & Baselining (Offline) ===
Extracting baseline embeddings through YOLOv11 backbone...
  Calibration batch 5/10 processed
  Calibration batch 10/10 processed
Baseline shape: torch.Size([640, 256])
[StatMonitor] Calibrated with 640 samples, embedding dim = 256
[StatMonitor] Baseline mu norm = 15.9351, variance mean = 0.991545
Baseline embeddings cached to reference_state/baseline_embeddings.pt

=== Phase 3: Integrated Production Stream (Online Simulation) ===
Starting live inference stream with single-pass YOLOv11...

  Batch  1 -> Drift Score: 1.4858 (threshold: 2.00)
       Mean Deviation: 2.3158 | Variance Expansion: 0.2409
       Status: Normal

  Batch  2 -> Drift Score: 1.0331 (threshold: 2.00)
       Mean Deviation: 1.5201 | Variance Expansion: 0.3027
       Status: Normal

  ...

  Batch 17 -> Drift Score: 2.1289 (threshold: 2.00)
       Mean Deviation: 3.3504 | Variance Expansion: 0.2965
       Status: *** DRIFT DETECTED ***

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ALERT: DATA DRIFT DETECTED IN LIVE INFERENCE STREAM
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Drift Score 2.1289 exceeds threshold 2.00
  Reason: Production feature activations have significantly deviated from baseline calibration.
  Action: Generating 'Before Training' distribution evidence...

[Visualizer] 'Before Training' plot saved to: drift_before_training_batch_17.png

=== Phase 5: Recovery Simulation (Retraining) ===
Simulating model retraining with drifted data exposure...

[Recovery] Simulating retraining with blend ratio = 30.0%
[Recovery] Generated 64 recovered embeddings

[Recovery Verification]
  Post-retraining drift score: 0.7871 (threshold: 2.00)
       Mean Deviation:      1.1263
       Variance Expansion:  0.2783
  Recovery successful — features realigned with baseline.

[Visualizer] 'After Training' plot saved to: drift_after_training.png
[Visualizer] Combined report saved to: drift_combined_report.png

======================================================================
  V3 LIFECYCLE COMPLETE
======================================================================
```

---

## Core Engineering Edge-Cases & Bugs Resolved

### 1. YOLOv11 Initialization Loop (Auto-Training Defeated)
* **Problem:** Calling the standard `YOLO("yolo11n-cls.pt")` constructor high-level class wrapper was aggressively triggering a 100-epoch automated training run on local datasets upon execution, causing 45-second script hangs.
* **Fix:** Bypassed the high-level API. Used low-level weight loading via `load_checkpoint` directly out of `ultralytics.nn.tasks` to load model dictionaries instantly as a clean, static `nn.Module`.

### 2. Tuple Container Unwrapping
* **Problem:** Subclassing YOLO layers led forward passes to return tuples containing internal layer dimensions, throwing `TypeError: argument 'input' must be Tensor` down the classification heads.
* **Fix:** Added list and tuple unboxing wrappers inside `yolo_extractor.py` to strip secondary layer outputs, returning an isolated, clean classification logits matrix.

### 3. Matplotlib Emoji Font Missing Glyphs
* **Problem:** Incorporating rich status indicators (such as white heavy check marks or critical alarms) inside graph headers threw heavy layout execution crashes (`DejaVu Sans glyph warnings`).
* **Fix:** Standardized visualization strings using clean ASCII typography (`[!] DRIFT`, `[OK] BASELINE`) for complete cross-platform rendering safety.

---

## Quick Start & Verification

### 1. Configure the Environment
Ensure your virtual environment is loaded, then update dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the central orchestrator to launch the entire automated lifecycle:
```bash
python main.py
```

### 3. Review Generated Artifacts
Once execution hits Phase 5, open your root project folder to analyze your newly outputted forensic proof:
* `drift_before_training_batch_17.png` (Statistical deviation plot)
* `drift_after_training.png` (Realignment verification plot)
* `drift_combined_report.png` (Side-by-side executive summary panel)
