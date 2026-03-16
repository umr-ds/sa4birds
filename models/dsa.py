import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block import MultiHeadSABlock



class DSA(nn.Module):
    """
    Dual-branch Spectrogram Attention (DSA) Model.

    This model combines local and global feature representations
    extracted from a CNN backbone and applies class-wise spatial
    attention for weakly supervised multi-label classification.

    Architecture Overview
    ----------------------
    1. Backbone (timm model):
      Extracts time-frequency feature maps.

    2. Dual Feature Branches:
      - Local branch: operates on full-resolution features.
      - Global branch: operates on spatially downsampled features
        via average pooling.

    3. Projection:
      Each branch applies LayerNorm, 1x1 projection, and residual
      refinement (local branch).

    4. Multi-Head Spatial Attention:
      Produces class-wise attention-weighted predictions.

    5. Learnable Fusion:
      Final prediction is a weighted combination of local and
      global branch outputs.

    Parameters
    ----------
    cfg : object
       Configuration object containing at least:
           - cfg.frontend.in_chans
           - cfg.network.model_name
           - cfg.network.droppath_rate
           - cfg.network.temperature
           - cfg.network.dropout_rate
           - cfg.train.num_classes

    Input
    -----
    x : torch.Tensor
       Shape (B, C, F, T)
       B = batch size
       C = input channels
       F = frequency bins
       T = time frames

    Returns
    -------
    torch.Tensor
       Weak (clip-level) class predictions of shape
       (B, num_classes).

    Notes
    -----
    - Designed for spectrogram-like inputs.
    - Uses timm backbone for feature extraction.
    - Supports optional center cropping for 5-second inference.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Input channels
        self.in_chans = self.cfg.frontend.in_chans

        # Expected pixel length for 5-second segment
        # (depends on FFT hop size etc.)
        self.px_per_5s = 501

        # Backbone downsampling factor
        # (depends on architecture)
        self.downsample_factor = 64

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
        # Attention block
        # --------------------------------------------------
        self.attention = MultiHeadSABlock(backbone_out,
                                           self.cfg.train.num_classes,
                                            heads=1,
                                            activation='sigmoid',
                                            temperature=self.cfg.network.temperature)

        # --------------------------------------------------
        # Local and Global projection layers
        # --------------------------------------------------
        self.l_proj = nn.Conv2d(backbone_out, backbone_out, kernel_size=(1,1), bias=True)
        self.g_proj = nn.Conv2d(backbone_out, backbone_out, kernel_size=(1,1), bias=True)


        self.dropout = cfg.network.dropout_rate

        # Feature normalization
        self.norm = nn.LayerNorm(normalized_shape=backbone_out)

        # Learnable fusion weight between branches
        self.fuse_weight = nn.Parameter(torch.tensor(0.9))

        # Spatial pooling for global branch
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))


    def forward(self, x, center_5s=False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, F, T).
        center_5s : bool, optional
            If True, only the center 5-second region of the
            feature map is used for prediction.

        Returns
        -------
        torch.Tensor
            Weak (clip-level) predictions of shape
            (B, num_classes).
        """
        def normalize_and_project(feat, proj_layer):
            """
                Apply LayerNorm and 1x1 projection with ReLU.
            """
            feat = self.norm(feat.transpose(1, -1)).transpose(1, -1)
            feat = F.relu(proj_layer(feat))
            return feat

        t = x.shape[-1]
        # Backbone feature extraction
        x = self.backbone(x)  # -> (b, feats, f, t)

        # --------------------------------------------------
        # Branch 1: Local features
        # --------------------------------------------------
        l_x = F.dropout(x, p=self.dropout, training=self.training)
        residual = l_x
        l_x = normalize_and_project(l_x, self.l_proj)
        l_x = residual + l_x
        l_x = F.dropout(l_x, p=self.dropout, training=self.training)

        # --------------------------------------------------
        # Branch 2: Global features
        # --------------------------------------------------
        g_x = self.avg_pool(x)
        g_x = F.dropout(g_x, p=self.dropout, training=self.training)
        g_x = normalize_and_project(g_x, self.g_proj)
        g_x = F.dropout(g_x, p=self.dropout, training=self.training)

        # --------------------------------------------------
        # Optional center cropping
        # --------------------------------------------------
        if center_5s:
            crop = int(round((t - self.px_per_5s) / self.downsample_factor))
        else:
            crop =  0

        l_weak, l_att, _ = self.attention(l_x, crop)
        g_weak, g_att, _ = self.attention(g_x, crop // 2)

        # --------------------------------------------------
        # Fuse local and global predictions
        # --------------------------------------------------
        fuse_weight = torch.clamp(self.fuse_weight, min=0, max=1)
        weak = l_weak * fuse_weight + g_weak * (1 - fuse_weight)

        if self.training:
            return weak, l_att, g_att
        else:
            return weak
