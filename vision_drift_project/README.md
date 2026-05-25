# Vision Drift Monitoring System (v2-development)

A production-ready, highly efficient pipeline for detecting statistical data drift in computer vision models. This system monitors incoming live image streams and mathematically compares them against a known "healthy" baseline to ensure your model's predictions remain reliable in the real world.

## Version 2: Integrated Efficiency Architecture

In this version (v2), the pipeline has been heavily optimized to reduce computational overhead in production.

### How v2 is Different from the Previous Approach (v1)

*   **Elimination of the "Double-Pass":** The v1 architecture was inefficient. To serve a user, it ran an image through ResNet-18 for a classification prediction, and then ran it *again* through a stripped-down feature extractor to get embeddings for drift monitoring. You paid the mathematical cost of the neural network twice.
*   **Integrated Single Forward Pass:** In v2, the `IntegratedVisionModel` processes both tasks simultaneously. A single `forward(x)` call yields both the classification logits and the intermediate 512-D embeddings.
*   **Massive Compute Savings:** By sharing the ResNet-18 backbone between the classifier and the drift monitor, GPU/CPU compute overhead is drastically reduced, enabling higher throughput for live streaming.
*   **Real-time Streaming Buffer:** Predictions (`logits`) are served instantly to the end-user, while the `embeddings` are silently dropped into an accumulation buffer. Once enough samples are gathered, bulk statistical drift validation happens in the background.
*   **Offline Weights Fallback:** Enhanced reliability by checking for local offline weights (`models/resnet18_weights.pth`) for air-gapped environments, gracefully falling back to downloading official PyTorch weights if the local file is missing.

---

## Project Structure

```text
vision_drift_project/
├── data/                  # Dataset cache (e.g., CIFAR-10)
├── models/                # Local weights storage (.pth files)
├── src/
│   ├── feature_extractor.py # Contains IntegratedVisionModel
│   ├── drift_detector.py    # MMD monitor using torchdrift
│   ├── data_loader.py       # Data streaming pipelines
│   └── utils.py             # t-SNE plot generation
├── reference_state/       # Cached baseline embeddings
├── main.py                # Main simulation loop
└── requirements.txt       # Project dependencies
```

---

## How It Works

The system decouples drift monitoring from standard model inference across three distinct phases:

### Phase 1: Ingestion & Model Loading
We initialize the `IntegratedVisionModel` (based on ResNet-18). The model logic splits the architecture into a feature backbone and a classification head, allowing dual-outputs from a single pass.

### Phase 2: Offline Calibration (The Baseline)
During calibration, we pass a gold-standard "healthy" dataset through the model. The classification outputs are ignored, but the intermediate embeddings are collected and saved to `reference_state/baseline_embeddings.pt`. This serves as the system's absolute memory of what healthy data looks like.

### Phase 3: Online Streaming & Drift Detection (The Live Stream)
Incoming stream data is collected into batches.
1. The model performs a single forward pass, generating both predictions for the user and embeddings for the monitor.
2. The `MMDMonitor` uses the **Kernel MMD** statistical test to compare the new batch's embeddings against the cached baseline.
3. Permutation Testing is performed. If the calculated **P-Value falls below 0.05**, the structural integrity of the data stream has failed.
4. The system triggers a critical alert and automatically generates a **t-SNE** visual scatter plot so you can visually verify the separation between the healthy reference cluster and the degraded production cluster.

---

## Getting Started

### 1. Install Requirements
Navigate to the project folder, activate your virtual environment, and install dependencies:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the main script to watch the simulation run:
```bash
python main.py
```

### 3. Understanding the Output
The pipeline will simulate incoming production data. 
* As batches process, you will see `Served predictions for batch X` followed by a background drift evaluation.
* If the data is clean, the system logs `Status Normal: Data stream is consistent with model expectations`. 
* Once degraded/corrupted images accumulate enough to cross the statistical threshold (P-Value < 0.05), the system throws a `🚨 ALERT: Data drift identified...` and saves a `drift_visualization_batch_X.png` plot to your directory.
