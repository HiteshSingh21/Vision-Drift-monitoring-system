import os
import torch
from src.data_loader import get_loaders
from src.feature_extractor import IntegratedVisionModel
from src.drift_detector import MMDMonitor
from src.utils import plot_embeddings_tsne

def main():
    print("=== Phase 1: Ingesting Data & Preparing Pipelines ===")
    os.makedirs('reference_state', exist_ok=True)
    ref_loader, prod_loader = get_loaders(batch_size=64)
    model = IntegratedVisionModel()
    monitor = MMDMonitor(p_val_threshold=0.05)
    
    print("\n=== Phase 2: Calibration & Baselining (Offline) ===")
    print("Calibrating baseline...")
    ref_embeddings = []
    with torch.no_grad():
        for i, (images, _) in enumerate(ref_loader):
            if i >= 10: break
            _, embedding = model(images)
            ref_embeddings.append(embedding.cpu())
            
    ref_embeddings_tensor = torch.cat(ref_embeddings, dim=0)
    monitor.fit(ref_embeddings_tensor)
    torch.save(ref_embeddings_tensor, 'reference_state/baseline_embeddings.pt')
    print("Baseline cached successfully to disk.")
    
    print("\n=== Phase 3: Integrated Production Stream (Online Simulation) ===")
    print("Starting live inference stream...")
    
    production_buffer = []
    BUFFER_SIZE = 64
    
    for batch_idx, (images, _) in enumerate(prod_loader):
        print(f"\nProcessing Production Streaming Batch #{batch_idx + 1}...")
        
        with torch.no_grad():
            logits, embedding = model(images)
            predictions = torch.argmax(logits, dim=1)
            
        print(f"Served predictions for batch {batch_idx+1}") 
        
        production_buffer.append(embedding.cpu())
        
        if len(production_buffer) >= (BUFFER_SIZE // 64):
            print("Evaluating accumulated buffer embeddings for structural drift...")
            current_stream = torch.cat(production_buffer, dim=0)
            production_buffer = []
            
            is_drifted, p_value = monitor.check_drift(current_stream)
            print(f"Streaming Window Matrix P-Value: {p_value:.6f}")
            
            if is_drifted:
                print("ALERT: Data drift identified inside live inference stream workflow!")
                print("Reason: Input features have significantly deviated from training context.")
                print("Action Recommended: Route payload to annotation queue & trigger retraining pipeline.")
                print("Generating Visual Evidence...")
                plot_embeddings_tsne(ref_embeddings_tensor, current_stream, save_path=f'drift_visualization_batch_{batch_idx + 1}.png')
                break
            else:
                print("Status Normal: Data stream is consistent with model expectations.")

if __name__ == "__main__":
    main()
