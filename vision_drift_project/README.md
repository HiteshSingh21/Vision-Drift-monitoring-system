# Vision Drift Monitoring System

A production-ready pipeline for detecting statistical data drift in computer vision models. This system monitors incoming image streams and mathematically compares them against a known "healthy" baseline to ensure your model's predictions remain reliable in the real world. 

## Project Structure

```text
vision_drift_project/
├── data/
├── models/
├── src/
│   ├── feature_extractor.py
│   ├── drift_detector.py
│   ├── data_loader.py
│   └── utils.py
├── reference_state/
├── main.py
└── requirements.txt
```

## The Modular Workflow

The system is designed to completely decouple the drift monitoring logic from standard model inference. It operates in three distinct phases:

### Phase 1: Ingestion & Feature Extraction
We use a pre-trained **ResNet-18** model as our feature extractor (`src/feature_extractor.py`). Instead of classifying images, we strip away the final classification layer to produce dense 512-dimensional embeddings. These embeddings act as a compressed mathematical footprint of the image. For true production readiness, the model loads its weights entirely offline from the `models/` directory.

### Phase 2: Offline Calibration (The Baseline)
During the offline phase, we pass a gold-standard dataset (clean CIFAR-10 images) through the ResNet-18 model. We save these baseline embeddings to the `reference_state/` folder. This acts as the absolute "Memory" of what healthy data looks like. 

### Phase 3: Online Streaming & Drift Detection
In the simulated production phase, incoming stream data is collected into batches (e.g., 64 images). Using `torchdrift`, we compare the batch's embeddings against our cached baseline. If the data starts degrading (e.g., simulated lens blur or lighting changes), the mathematical distance between the datasets increases.

---

## Mathematical Methods Applied

### 1. Maximum Mean Discrepancy (MMD)
To compare two high-dimensional datasets without relying on labels, we use the **Kernel MMD** statistical test. MMD measures the distance between the mean embeddings of two probability distributions in a reproducing kernel Hilbert space (RKHS). A larger MMD distance implies the production data has "drifted" away from the training distribution.

### 2. Permutation Testing (P-Values)
To determine if the calculated MMD distance is statistically significant (and not just random noise), the engine uses Permutation Testing. It shuffles the reference and production data together 1,000 times to see if the observed distance could happen by pure random chance. 
* If the calculated **P-Value falls below 0.05**, the system triggers a critical drift alert. 
* **Extreme Drift:** (e.g., heavy blur) will trigger a hard `0.000` P-Value.
* **Subtle Drift:** (e.g., slight brightness shifts) will cause the P-Value to fluctuate safely above 0.05 until enough degraded images aggregate in a single batch to cross the threshold.

### 3. t-SNE Visualization
When a critical drift alert is triggered, the system automatically calls `src/utils.py` to generate a 2D scatter plot using **t-SNE** (t-Distributed Stochastic Neighbor Embedding). This maps the 512-dimensional embeddings down to two dimensions so you can visually verify the separation between the healthy reference cluster (blue dots) and the drifted production cluster (red triangles).

---

## Getting Started

### 1. Install Requirements
Make sure your virtual environment is activated, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the main script to watch the simulation run:
```bash
python main.py
```

### 3. Understanding the Output
The pipeline will simulate incoming production data. 
* If the data is clean (or the drift is too subtle in a given batch), the system logs `Status Normal`. 
* Once a batch crosses the statistical threshold (P-Value < 0.05), the system halts, throws a `CRITICAL DATA DRIFT DETECTED` alert, and saves a `drift_visualization_batch_X.png` chart to your folder.
