"""
YOLOv11 Inference Layer

Wraps a YOLOv11 classification backbone to extract both classification
logits and intermediate feature embeddings in a single forward pass.
A forward hook on the penultimate backbone layer (C2PSA) captures spatial
features for downstream drift analysis.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _download_yolo_weights(model_variant):
    """Download pretrained weights if not already cached locally."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    local_path = os.path.join(models_dir, model_variant)
    if os.path.isfile(local_path):
        return local_path

    cwd_path = os.path.join(os.getcwd(), model_variant)
    if os.path.isfile(cwd_path):
        return cwd_path

    # Download from ultralytics asset repo
    print(f"Downloading {model_variant}...")
    from ultralytics.utils.downloads import attempt_download_asset
    downloaded_path = attempt_download_asset(model_variant)
    return str(downloaded_path)


class IntegratedYOLO11Model(nn.Module):
    """
    YOLOv11 wrapper producing (logits, embeddings) from a single forward pass.

    Hooks into the backbone's penultimate layer to capture spatial features,
    then pools them into a fixed-length embedding vector for drift monitoring.
    Weights are loaded directly via load_checkpoint to avoid the YOLO class
    auto-training behavior.
    """

    def __init__(self, model_variant='yolo11n-cls.pt', device=None):
        super(IntegratedYOLO11Model, self).__init__()
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading YOLOv11 model: {model_variant}")
        weights_path = _download_yolo_weights(model_variant)

        # Load weights directly, bypassing YOLO() to avoid auto-training
        from ultralytics.nn.tasks import load_checkpoint
        self._torch_model, _ckpt = load_checkpoint(
            weights_path, device=str(self.device)
        )
        self._torch_model.eval()

        # Freeze all params for inference only
        for param in self._torch_model.parameters():
            param.requires_grad = False

        self._hooked_features = None

        # Hook the layer right before the classification head
        backbone_layers = list(self._torch_model.model.children())
        self._hook_target = backbone_layers[-2]
        self._hook_handle = self._hook_target.register_forward_hook(self._feature_hook)

        self.eval()
        print(f"Model loaded on {self.device}, "
              f"hook on: {self._hook_target.__class__.__name__}")

    def _feature_hook(self, module, input, output):
        """Captures intermediate features from the hooked layer."""
        self._hooked_features = output

    def forward(self, x):
        """
        Returns (logits, embedding) from a single inference pass.
        logits:    (B, num_classes)
        embedding: (B, D) pooled spatial features
        """
        x = x.to(self.device)
        raw_output = self._torch_model(x)

        # Normalize output (model may return tuple or tensor)
        if isinstance(raw_output, (tuple, list)):
            logits = raw_output[0] if isinstance(raw_output[0], torch.Tensor) else raw_output[-1]
        else:
            logits = raw_output

        if logits.dim() > 2:
            logits = logits.flatten(1)

        # Pool hooked features to fixed-length vector
        features = self._hooked_features
        if features.dim() == 4:
            embedding = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif features.dim() == 3:
            embedding = features.mean(dim=1)
        else:
            embedding = features.flatten(1)

        self._hooked_features = None
        return logits, embedding

    def get_embedding_dim(self):
        """Run a dummy forward pass to determine embedding size."""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _, emb = self.forward(dummy)
        return emb.shape[1]

    def __del__(self):
        if hasattr(self, '_hook_handle'):
            self._hook_handle.remove()
