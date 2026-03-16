import torch
import torchvision
from torch import nn

class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss is designed for highly imbalanced multi-label problems.
    It extends Binary Cross Entropy (BCE) with asymmetric focusing
     (different gamma for positives/negatives)

    Parameters
    ----------
    gamma_neg : float, optional
        Focusing parameter for negative samples (default: 4).
    gamma_pos : float, optional
        Focusing parameter for positive samples (default: 1).
    clip : float or None, optional
        Probability clipping factor applied to negative
        probabilities to reduce dominance of easy negatives.
        Default: 0.05.
    eps : float, optional
        Small constant for numerical stability in log.
        Default: 1e-8.
    disable_torch_grad_focal_loss : bool, optional
        If True, disables gradient tracking for focal weight
        computation (slightly more memory-efficient).
    reduction : str, optional
        Reduction method: "mean", "sum", or "none".

    Input
    -----
    x : torch.Tensor
        Logits of shape (B, num_classes).
    y : torch.Tensor
        Multi-label binary targets of same shape.
    activated : bool, optional
        If True, `x` is assumed to already be passed
        through sigmoid.

    Returns
    -------
    torch.Tensor
        Scalar loss (if reduced) or per-element loss.

    Notes
    -----
    - Suitable for imbalanced multi-label datasets.
    - Reduces penalty for easy negative samples.
    - Generalizes focal loss to asymmetric behavior.
    """
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
        reduction="mean",
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y, activated=False):
        """
        Compute asymmetric multi-label loss.

        Parameters
        ----------
        x : torch.Tensor
            Model logits or probabilities.
        y : torch.Tensor
            Binary multi-label targets.
        activated : bool
            If True, x is already sigmoid-activated.

        Returns
        -------
        torch.Tensor
            Reduced loss value.
        """

        # Calculating Probabilities
        if activated:
            x_sigmoid = x
        else:
            x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()

        return -loss


class FocalLossBCE(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) and Focal Loss.

    This loss combines standard BCEWithLogitsLoss with
    sigmoid focal loss. It allows weighting both components,
    providing flexibility between stable optimization (BCE)
    and hard-example focusing (Focal Loss).


    Parameters
    ----------
    alpha : float, optional
       Balancing factor for focal loss (default: 0.25).
       Controls class weighting between positive and negative samples.
    gamma : float, optional
       Focusing parameter for focal loss (default: 2).
       Higher values increase focus on hard examples.
    reduction : str, optional
       Reduction method: "mean", "sum", or "none".
       Default: "mean".
    bce_weight : float, optional
       Scaling factor for BCE loss component.
       Default: 1.0.
    focal_weight : float, optional
       Scaling factor for focal loss component.
       Default: 1.0.

    Input
    -----
    logits : torch.Tensor
       Raw model outputs (before sigmoid), shape (B, num_classes).
    targets : torch.Tensor
       Binary multi-label targets of same shape.

    Returns
    -------
    torch.Tensor
       Combined loss value.

    Notes
    -----
    - Uses `torchvision.ops.sigmoid_focal_loss`.
    - Expects logits (not probabilities).
    - Suitable for imbalanced multi-label problems.
    """
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        """
        Compute combined BCE + Focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs (before sigmoid).
        targets : torch.Tensor
            Multi-label binary targets.

        Returns
        -------
        torch.Tensor
            Weighted sum of BCE and focal loss.
        """
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss
