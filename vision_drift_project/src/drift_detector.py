import torch
import torchdrift
class MMDMonitor:
    def __init__(self, p_val_threshold=0.05):
        self.detector = torchdrift.detectors.KernelMMDDriftDetector()
        self.p_val_threshold = p_val_threshold
    def fit(self, reference_embeddings):
        print(f"Calibrating monitor with {reference_embeddings.shape[0]} reference samples...")
        self.detector.fit(reference_embeddings)
    def check_drift(self, current_batch_embeddings):
        with torch.no_grad():
            score = self.detector(current_batch_embeddings)
            p_val = self.detector.compute_p_value(current_batch_embeddings)
        is_drifted = p_val.item() < self.p_val_threshold
        return is_drifted, p_val.item()
