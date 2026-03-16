import torch
import torch.nn.functional as F
import torch.nn as nn
import timm



def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)



class AttBlock(nn.Module):
    """
   1D Attention Block for temporal aggregation.

   This module applies class-wise temporal attention over
   1D feature sequences. It consists of:

       - Attention branch (Conv1d + softmax)
       - Classification branch (Conv1d + activation)
       - Weighted temporal aggregation

   The attention weights are normalized over the temporal
   dimension and used to compute a weighted sum of
   classification features.

   Parameters
   ----------
   in_features : int
       Number of input feature channels.
   out_features : int
       Number of output classes.
   activation : str, optional
       Activation applied to classification logits.
       Supported: {"linear", "sigmoid"}.
       Default: "linear".
   temperature : float, optional
       Temperature scaling factor (stored but not currently
       applied in forward pass).
       Default: 2.0.

   Input
   -----
   x : torch.Tensor
       Shape (B, C, T)
       B = batch size
       C = feature channels
       T = temporal dimension

   Returns
   -------
   tuple:
       - x : torch.Tensor
           Aggregated class predictions (B, out_features)
       - norm_att : torch.Tensor
           Normalized attention weights (B, out_features, T)
       - cla : torch.Tensor
           Classification logits before aggregation
           (B, out_features, T)

   Notes
   -----
   - Attention logits are clamped to [-10, 10] before softmax
     for numerical stability.
   - Uses 1x1 convolutions to compute attention and class
     predictions.
   """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=2.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x, crop):
        if crop > 0:
            x = x[:, :, crop: -crop]

        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class TimeAttModel(nn.Module):
    """
    Time attention model with temporal attention.

    This model extracts time-frequency features using a timm backbone,
    collapses the frequency dimension, and applies a temporal
    attention block to obtain weak (clip-level) predictions.

    Architecture Overview
    ---------------------
    1. timm backbone (feature extractor)
    2. Frequency pooling (mean over mel bins)
    3. Temporal projection (Linear + ReLU)
    4. Temporal attention block (AttBlock)
    5. Weak prediction via attention-weighted aggregation

    Parameters
    ----------
    cfg : object
        Configuration object containing:
            - cfg.frontend.in_chans
            - cfg.frontend.n_mels
            - cfg.frontend.target_length
            - cfg.network.model_name
            - cfg.network.droppath_rate
            - cfg.network.dropout_rate
            - cfg.train.num_classes

    Input
    -----
    x : torch.Tensor
        Shape (B, C, F, T)
        B = batch size
        C = input channels
        F = frequency bins (e.g., mel bins)
        T = time frames

    Returns
    -------
    torch.Tensor
        Weak (clip-level) predictions of shape (B, num_classes)

    Notes
    -----
    - Uses pretrained timm backbone.
    - If model_name contains "deit", img_size must be specified.
    - If model_name contains "efficientnet", classifier removal
      is handled differently.
    - Temporal cropping is applied during evaluation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.in_chans = self.cfg.frontend.in_chans

        timm_params = {"model_name": cfg.network.model_name,
                       "pretrained": False,
                       "in_chans": self.in_chans,
                       "drop_path_rate": self.cfg.network.droppath_rate}

        self.backbone = timm.create_model(**timm_params)
        backbone_out = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0, '')

        # Expected pixel length for 5-second segment
        # (depends on FFT hop size etc.)
        self.px_per_5s = 501

        # Backbone downsampling factor
        # (depends on architecture)
        self.downsample_factor = 64

        self.attention = AttBlock(backbone_out,  self.cfg.train.num_classes, activation='sigmoid')
        self.proj = nn.Linear(backbone_out, backbone_out, bias=True)
        self.dropout = self.cfg.network.dropout_rate


    def forward(self, x, center_5s=False):
        t = x.shape[-1]
        # x: (b, c, f, t)
        x = self.backbone(x)  # -> (b, feats, f, t)

        x = torch.mean(x, dim=2)  # -> (b, feats, t)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2) # -> (b, t, feats)
        x = F.relu_(self.proj(x))
        x = x.transpose(1, 2) # -> (b, feats, t)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if center_5s:
            crop = int(round((t - self.px_per_5s) / self.downsample_factor))
        else:
            crop = 0

        # -> weak: (b, n_class)
        # -> norm_att: (b, n_class, t)
        # -> strong: (b, n_class, t)
        weak, norm_att, strong = self.attention(x, crop)

        return weak