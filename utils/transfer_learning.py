"""
This file defines utility functions for transfer learning.
"""
import numpy as np
import torch
from torch import nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """
    Feature extractor for the model.
    """

    def __init__(self, model, topk=1):
        super().__init__()
        self.base_model = model
        self.topk = topk
        self._features = {}

        self.register_hook(topk)

    def register_hook(self, topk):
        chosen_layers = get_topk_layers(self.base_model, topk)
        for name, module in self.base_model.named_children():
            if module in chosen_layers:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output

        return hook

    def forward(self, x):
        self._features = {}
        self.base_model(x)
        feats_cat = []
        for k, feat in self._features.items():
            feats_cat.append(feat.flatten(1))
        feats_cat = torch.cat(feats_cat, dim=1) if feats_cat else None

        return feats_cat


class WrappedModel(nn.Module):
    def __init__(self, feature_extractor, fc):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = fc

    def forward(self, x):
        feats_cat = self.feature_extractor(x)
        out = self.fc(feats_cat)
        return out


def get_topk_layers(model, topk):
    """
    Get the top-k layers of the model excluding the classification head.
    """
    # Get the top-level children
    children = list(model.children())

    # Exclude the last one if it's the classification head:
    if isinstance(children[-1], nn.Linear):
        children = children[:-1]

    # Now just take the last `topk` from that shortened list
    if topk > len(children):
        topk = len(children)
    return children[-topk:]


def extract_features(feature_extractor, dataloader):
    """
    Extract features from the model.
    """
    feature_extractor.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            if isinstance(targets, list) and isinstance(targets[0], torch.Tensor):  # Hack for handling celeb_a
                targets = torch.stack(targets, dim=0)  # shape: (40, batch_size)
                targets = targets.T
                targets = targets.float()  # ensure float if needed

            inputs = inputs.to(device)
            feature = feature_extractor(inputs)
            features.append(feature.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels
