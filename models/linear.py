import timm
import torch
from torch import nn as nn


class LinClsModel(nn.Module):
    """
    Generic CNN classifier built on top of a timm backbone.

    This model wraps a pretrained CNN from the `timm` library,
    removes its original classification head, and replaces it
    with a task-specific linear classifier.

    Architecture Overview
    ---------------------
    1. timm backbone (feature extractor)
    2. Global adaptive average pooling
    3. Dropout
    4. Linear classification head

    Parameters
    ----------
    cfg : object
       Configuration object containing:
           - cfg.frontend.in_chans
           - cfg.network.model_name
           - cfg.network.droppath_rate
           - cfg.network.dropout_rate
           - cfg.train.num_classes

    Input
    -----
    x : torch.Tensor
       Shape (B, C, H, W)
       B = batch size
       C = input channels
       H, W = spatial dimensions (e.g., spectrogram height/width)

    Returns
    -------
    torch.Tensor
       Logits of shape (B, num_classes)

    Notes
    -----
    - Uses pretrained ImageNet weights by default.
    - The original classifier of the backbone is removed.
    - Suitable for spectrogram-based classification tasks.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Input channels
        self.in_chans = self.cfg.frontend.in_chans

        # --------------------------------------------------
        # Backbone (timm model)
        # --------------------------------------------------
        timm_params = {"model_name": cfg.network.model_name,
                       "pretrained": True,
                       "in_chans": self.in_chans,
                       "drop_path_rate": self.cfg.network.droppath_rate}

        self.backbone = timm.create_model(**timm_params)


        backbone_out = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0, '')

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.cfg.network.dropout_rate)

        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, cfg.train.num_classes)



    def forward(self, x, center_5s=False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
           Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
           Logits of shape (B, num_classes)
        """

        # Extract spatial feature maps from backbone
        features = self.backbone(x)

        # Ensure backbone outputs 4D feature maps
        assert len(features.shape) == 4

        # Global pooling to obtain feature vector
        if not isinstance(self.pooling, nn.Identity):
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        # Apply dropout and final classification layer
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits

