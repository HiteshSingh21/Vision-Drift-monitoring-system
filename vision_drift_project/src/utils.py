import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
def plot_embeddings_tsne(ref_embeddings, prod_embeddings, save_path='drift_visualization.png'):
    print("Generating t-SNE visualization... (this may take a moment)")
    ref_np = ref_embeddings.cpu().numpy()
    prod_np = prod_embeddings.cpu().numpy()
    labels = np.concatenate([np.zeros(len(ref_np)), np.ones(len(prod_np))])
    combined_data = np.vstack([ref_np, prod_np])
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(combined_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[labels == 0, 0], reduced_embeddings[labels == 0, 1], 
                c='blue', label='Reference (Baseline)', alpha=0.5, edgecolors='w', s=50)
    plt.scatter(reduced_embeddings[labels == 1, 0], reduced_embeddings[labels == 1, 1], 
                c='red', label='Production (Stream)', alpha=0.5, edgecolors='w', marker='^', s=50)
    plt.title('t-SNE Visualization: Reference vs Production Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")
