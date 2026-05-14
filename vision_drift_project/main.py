import os
import torch
from src.data_loader import get_loaders
from src.feature_extractor import VisionEncoder
from src.drift_detector import MMDMonitor
from src.utils import plot_embeddings_tsne
def main():
    print("=== Phase 1: Ingesting Data & Preparing Pipelines ===")
    os.makedirs('reference_state', exist_ok=True)
    ref_loader, prod_loader = get_loaders(batch_size=64)
    encoder = VisionEncoder()
    print("\n=== Phase 2: Calibration & Baselining (Offline) ===")
    ref_embeddings = encoder.extract_dataset_embeddings(ref_loader, max_batches=10)
    monitor = MMDMonitor(p_val_threshold=0.05)
    monitor.fit(ref_embeddings)
    torch.save(ref_embeddings, 'reference_state/baseline_embeddings.pt')
    print("Baseline cached successfully to disk.")
    print("\n=== Phase 3: Live Stream Monitoring (Online Simulation) ===")
    for batch_idx, (images, _) in enumerate(prod_loader):
        print(f"\nProcessing Production Streaming Batch #{batch_idx + 1}...")
        prod_embeddings = encoder.predict_single_batch(images)
        is_drifted, p_value = monitor.check_drift(prod_embeddings)
        print(f"Calculated Batch Matrix P-Value: {p_value:.6f}")
        if is_drifted:
            print("ALERT: CRITICAL DATA DRIFT DETECTED!")
            print("Reason: Input features have significantly deviated from training context.")
            print("Action Recommended: Route payload to annotation queue & trigger retraining pipeline.")
            print("Generating Visual Evidence...")
            plot_embeddings_tsne(ref_embeddings, prod_embeddings, save_path=f'drift_visualization_batch_{batch_idx + 1}.png')
            break
        else:
            print("Status Normal: Data stream is consistent with model expectations.")
if __name__ == "__main__":
    main()
