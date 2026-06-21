"""
Vision Drift Monitoring System - V3 Lifecycle

Runs the full pipeline: data ingestion, baseline calibration,
production streaming with drift detection, and recovery simulation.
"""

import os
import torch
import numpy as np
from src.data_loader import get_loaders
from src.yolo_extractor import IntegratedYOLO11Model
from src.stat_monitor import StatisticalDriftMonitor
from src.visualizer import (
    plot_drift_distributions,
    plot_recovery_distributions,
    plot_combined_report,
)


def simulate_retraining(model, baseline_embeddings, drifted_embeddings, blend_ratio=0.3):
    """
    Approximate a retraining event by shifting drifted embeddings back
    toward the baseline mean. Adds slight noise for realism.
    """
    print(f"\n[Recovery] Simulating retraining (blend ratio={blend_ratio:.1%})")

    baseline_mean = baseline_embeddings.mean(dim=0)
    drifted_mean = drifted_embeddings.mean(dim=0)

    # Shift drifted features back toward baseline center
    recovery_shift = (baseline_mean - drifted_mean) * (1.0 - blend_ratio)
    recovered = drifted_embeddings + recovery_shift.unsqueeze(0)

    # Small noise to simulate imperfect recovery
    noise_scale = baseline_embeddings.std(dim=0).mean() * 0.05
    recovered = recovered + torch.randn_like(recovered) * noise_scale

    print(f"[Recovery] Generated {recovered.shape[0]} recovered embeddings")
    return recovered


def main():
    # -- Phase 1: Data Ingestion --
    print("=" * 70)
    print("  VISION DRIFT MONITORING SYSTEM - V3")
    print("=" * 70)

    print("\n=== Phase 1: Data Ingestion ===")
    os.makedirs('reference_state', exist_ok=True)
    ref_loader, prod_loader = get_loaders(batch_size=64)

    model = IntegratedYOLO11Model()
    emb_dim = model.get_embedding_dim()
    print(f"Embedding dim: {emb_dim}")

    monitor = StatisticalDriftMonitor(threshold=2.0, alpha=0.6)

    # -- Phase 2: Baseline Calibration --
    print("\n=== Phase 2: Baseline Calibration ===")
    print("Extracting baseline embeddings...")

    ref_embeddings = []
    with torch.no_grad():
        for i, (images, _) in enumerate(ref_loader):
            if i >= 10:
                break
            _, embedding = model(images)
            ref_embeddings.append(embedding.cpu())
            if (i + 1) % 5 == 0:
                print(f"  Batch {i + 1}/10 done")

    ref_embeddings_tensor = torch.cat(ref_embeddings, dim=0)
    print(f"Baseline shape: {ref_embeddings_tensor.shape}")

    monitor.fit(ref_embeddings_tensor)

    torch.save(ref_embeddings_tensor, 'reference_state/baseline_embeddings.pt')
    print("Baseline cached to reference_state/baseline_embeddings.pt")

    # -- Phase 3: Production Streaming --
    print("\n=== Phase 3: Production Streaming ===")
    print("Running inference on production data...")

    production_buffer = []
    BUFFER_SIZE = 64
    drifted_embeddings = None

    for batch_idx, (images, _) in enumerate(prod_loader):
        print(f"\nBatch #{batch_idx + 1}...")

        with torch.no_grad():
            logits, embedding = model(images)
            predictions = torch.argmax(logits, dim=1)

        class_counts = torch.bincount(predictions, minlength=10)[:5].tolist()
        print(f"  Predictions served (top classes: {class_counts})")

        production_buffer.append(embedding.cpu())

        if len(production_buffer) >= (BUFFER_SIZE // 64):
            print("  Checking for drift...")
            current_stream = torch.cat(production_buffer, dim=0)
            production_buffer = []

            is_drifted, drift_score, details = monitor.check_drift(current_stream)

            print(f"  Score: {drift_score:.4f} (threshold: {details['threshold']:.2f})")
            print(f"    Mean deviation:     {details['mean_deviation']:.4f}")
            print(f"    Variance expansion: {details['variance_expansion']:.4f}")

            if is_drifted:
                # -- Phase 4: Drift detected --
                print("\n" + "=" * 70)
                print("  DRIFT DETECTED")
                print("=" * 70)
                print(f"  Score {drift_score:.4f} exceeds threshold "
                      f"{details['threshold']:.2f}")
                print("  Generating before-training distribution plot...")

                drifted_embeddings = current_stream
                plot_drift_distributions(
                    baseline_emb=ref_embeddings_tensor,
                    drifted_emb=drifted_embeddings,
                    save_path=f'drift_before_training_batch_{batch_idx + 1}.png'
                )
                break
            else:
                print("  Status: Normal")

    # -- Phase 5: Recovery Simulation --
    if drifted_embeddings is not None:
        print("\n=== Phase 5: Recovery Simulation ===")

        recovered_embeddings = simulate_retraining(
            model=model,
            baseline_embeddings=ref_embeddings_tensor,
            drifted_embeddings=drifted_embeddings,
            blend_ratio=0.3
        )

        is_still_drifted, recovery_score, recovery_details = monitor.check_drift(recovered_embeddings)

        print(f"\n  Post-recovery score: {recovery_score:.4f} "
              f"(threshold: {recovery_details['threshold']:.2f})")
        print(f"    Mean deviation:     {recovery_details['mean_deviation']:.4f}")
        print(f"    Variance expansion: {recovery_details['variance_expansion']:.4f}")

        if not is_still_drifted:
            print("  Recovery successful - features realigned with baseline.")
        else:
            print("  Partial recovery - some residual drift remains.")

        plot_recovery_distributions(
            baseline_emb=ref_embeddings_tensor,
            recovered_emb=recovered_embeddings,
            save_path='drift_after_training.png'
        )

        plot_combined_report(
            baseline_emb=ref_embeddings_tensor,
            drifted_emb=drifted_embeddings,
            recovered_emb=recovered_embeddings,
            save_path='drift_combined_report.png'
        )

        print("\n" + "=" * 70)
        print("  LIFECYCLE COMPLETE")
        print("=" * 70)
        print("  Output files:")
        print("    drift_before_training_batch_X.png")
        print("    drift_after_training.png")
        print("    drift_combined_report.png")
        print("=" * 70)
    else:
        print("\n=== Done ===")
        print("No drift detected during this run.")


if __name__ == "__main__":
    main()
