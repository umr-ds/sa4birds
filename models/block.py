import torch
from torch import nn



class MultiHeadSABlock(nn.Module):
    """
    Multi-Head Spectrogram Attention Block.

    This module applies class-wise multi-head attention over
    2D feature maps (e.g., time-frequency representations).
    It learns separate attention maps and classification
    features per head and aggregates them into final
    class-level predictions.

    Parameters
    ----------
    in_features : int
       Number of input feature channels.
    num_classes : int
       Number of output classes.
    heads : int, optional
       Number of attention heads. Default: 1.
    activation : str, optional
       Activation function applied to classification features.
       Supported: {"linear", "sigmoid", "tanh"}.
       Default: "linear".
    temperature : float, optional
       Temperature scaling factor applied to attention logits
       before softmax. Higher values produce softer attention.
       Default: 2.0.

    Input
    -----
    x : torch.Tensor
       Shape (B, C, F, T)
       B = batch size
       C = input channels
       F = frequency dimension
       T = time dimension

    Returns
    -------
    tuple:
       final_features : torch.Tensor
           Shape (B, num_classes)
           Aggregated class predictions.
       norm_att : torch.Tensor
           Normalized attention maps.
           Shape (B, num_classes, heads, F, T)
       cla_feat : torch.Tensor
           Activated class-wise features before aggregation.
           Shape (B, num_classes, heads, F, T)
   """
    def __init__(self, in_features, num_classes, heads=1, activation="linear", temperature=2.0):
        super().__init__()
        self.heads = heads
        self.activation = activation
        self.temperature = torch.tensor(float(temperature))

        # Attention map generator (1x1 conv)
        self.att = nn.Conv2d(in_features, num_classes * heads, kernel_size=(1, 1), bias=True)

        # Class feature generator (1x1 conv)
        self.cla = nn.Conv2d(in_features, num_classes * heads, kernel_size=1, bias=True)

        # Learnable head importance weights
        self.head_weights = nn.Parameter(torch.ones(self.heads))

    def nonlinearity(self, x):
        """
            Apply selected activation function.
        """
        activations = {
            "linear": lambda t: t,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }
        if self.activation not in activations:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return activations[self.activation](x)

    def forward(self, x, crop):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, F, T).
        crop : int
            Number of frames to crop from both temporal sides.

        Returns
        -------
        tuple:
            (final_features, norm_att, cla_feat)
        """

        # Optional temporal cropping
        if crop > 0:
            x = x[:, :, :, crop: -crop]

        B, C, Fq, T = x.size()

        # --------------------------------------------------
        # Attention branch
        # --------------------------------------------------
        att_map = self.att(x) / self.temperature
        att_map = att_map.view(B, -1, self.heads, Fq, T)

        # Softmax over spatial dimensions (F*T)
        norm_att = torch.softmax(att_map.view(B, -1, self.heads, Fq * T), dim=-1)
        norm_att = norm_att.view(B, -1, self.heads, Fq, T)

        # --------------------------------------------------
        # Classification branch
        # --------------------------------------------------
        cla_feat = self.nonlinearity(self.cla(x))
        cla_feat = cla_feat.view(B, -1, self.heads, Fq, T)

        # Apply attention weighting
        weighted = norm_att * cla_feat

        # Aggregate spatial dimensions
        weighted = weighted.sum(dim=-1).sum(dim=-1)

        # Weighted aggregation across heads
        final_features = (weighted * self.head_weights.view(1, 1, -1)).sum(dim=-1) / self.head_weights.sum()

        return final_features, norm_att, cla_feat



class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism.

    Applies multiplicative gating between two linear projections
    to compute class-wise attention weights.

    Commonly used in multiple instance learning (MIL) settings.

    Parameters
    ----------
    in_dim : int
       Input feature dimension.
    n_classes : int
       Number of output classes.

    Input
    -----
    x : torch.Tensor
       Shape (B, T, D)
       B = batch size
       T = sequence length
       D = feature dimension

    Returns
    -------
    torch.Tensor
       Attention weights of shape (B, T, n_classes),
       normalized over the temporal dimension.
    """
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.att = nn.Linear(in_dim, n_classes)
        self.gate = nn.Linear(in_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Compute gated attention weights.

        Parameters
        ----------
        x : torch.Tensor
           Input tensor of shape (B, T, D).

        Returns
        -------
        torch.Tensor
           Attention weights of shape (B, T, n_classes).
       """
        attn_logits = torch.tanh(self.att(x)) * torch.sigmoid(self.gate(x))
        attn_weights = self.softmax(attn_logits)  # (B, T, n_classes)
        return attn_weights