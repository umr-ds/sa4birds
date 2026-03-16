import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics import AUROC, Metric
from torchmetrics.classification import MultilabelAveragePrecision
import torch

class TopKAccuracy(Metric):
    """
    Compute Top-K accuracy for multi-label classification tasks.

    This metric evaluates whether at least one true label appears
    within the model's top-K predicted classes for each sample.

    Optionally, it can handle "no-call" samples (instances with no
    positive labels) by verifying that all top-K prediction scores
    fall below a specified threshold.

    Args:
        topk (int, optional):
            Number of highest-scoring predictions to consider.
            Default: 1 (Top-1 accuracy).
        include_nocalls (bool, optional):
            Whether to include samples with no positive labels
            (all-zero targets) in the metric calculation.
            Default: False.
        threshold (float, optional):
            Score threshold used when evaluating no-call samples.
            Only used if `include_nocalls=True`.
            Default: 0.5.
        **kwargs:
            Additional keyword arguments passed to the base `Metric`
            class (e.g., for distributed settings).

    Notes:
        - Assumes `preds` are prediction scores (e.g., probabilities
          or logits).
        - Assumes `targets` are multi-hot encoded tensors.
        - Uses distributed-safe state tracking via `add_state`.
     Source:
        Adapted from BirdSet repository:
        https://github.com/DBD-research-group/BirdSet/blob/9929cc436153441d617f3799ea8a94ec2649b35e/birdset/modules/metrics/multilabel.py
    """
    def __init__(self, topk=1, include_nocalls=False, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """
        Update metric state with a batch of predictions and targets.

        Args:
          preds (torch.Tensor):
              Model prediction scores of shape (batch_size, num_classes).
          targets (torch.Tensor):
              Multi-hot ground-truth labels of shape
              (batch_size, num_classes).
        """

        # Get the indices of the top-K predictions per sample
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)

        # Ensure targets are on the same device
        targets = targets.to(preds.device)

        # Identify "no-call" samples (all-zero targets)
        no_call_targets = targets.sum(dim=1) == 0

        # --------------------------------------------------------
        # Handle no-call instances
        # --------------------------------------------------------
        if self.include_nocalls:
            # check if top-k predictions for all-negative instances are less than threshold
            no_positive_predictions = (
                preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            )
            correct_all_negative = no_call_targets & no_positive_predictions.all(dim=1)

        else:
            # If no-calls are excluded, they do not contribute to correct
            correct_all_negative = torch.tensor(0).to(targets.device)

        # --------------------------------------------------------
        # Handle positive (normal) samples
        # --------------------------------------------------------

        # Expand targets to match top-K shape
        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)

        # Check whether any of the top-K predictions match a true label
        correct_positive = expanded_targets.gather(
            2, topk_pred_indices.unsqueeze(-1)
        ).any(dim=1)

        # --------------------------------------------------------
        # Update internal counters
        # ---
        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            # Exclude no-call samples from denominator
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            # Include all samples
            self.total += targets.size(0)

    def compute(self):
        """
        Compute final Top-K accuracy.

        Returns:
            torch.Tensor:
                Accuracy value (correct / total).
        """
        return self.correct.float() / self.total


def calculate_auc(targets, outputs):
    """
    Compute per-class and mean ROC-AUC scores for multi-label predictions.

    This function calculates the Area Under the Receiver Operating
    Characteristic Curve (ROC-AUC) independently for each class.
    Classes without any positive samples in the ground-truth labels
    are skipped when computing the mean AUC.

    Args:
        targets (torch.Tensor):
            Ground-truth labels of shape (num_samples, num_classes).
            Expected to contain binary or probabilistic labels.
        outputs (torch.Tensor):
            Model prediction scores of shape (num_samples, num_classes).
            Typically sigmoid outputs or logits converted to probabilities.

    Returns:
        tuple:
            - float: Mean ROC-AUC across all valid classes (those with at least
              one positive sample).
            - list[float]: Per-class ROC-AUC scores. For classes without
              positive samples, the value is set to -1.

    Notes:
        - Targets are binarized using a threshold of 0.5.
        - Classes with no positive ground-truth samples are excluded
          from the mean AUC calculation.
        - Requires `roc_auc_score` from sklearn.metrics.
    """
    np_targets = targets.cpu().numpy()
    np_outputs = outputs.cpu().numpy()
    num_classes = np_targets.shape[1]
    aucs = []
    aucs2 = []

    for i in range(num_classes):
        np_targets[:, i] = np.where(np_targets[:, i] > 0.5, 1, 0)

        if np.sum(np_targets[:, i]) > 0:
            class_auc = roc_auc_score(np_targets[:, i], np_outputs[:, i])
            aucs.append(class_auc)
            aucs2.append(class_auc)
        else:
            aucs2.append(-1)

    return np.mean(aucs), aucs2


def calculate_map(targets, outputs):
    """
    Compute per-class and mean Average Precision (mAP) for multi-label predictions.

    This function calculates the Average Precision (AP) independently
    for each class and returns both the mean AP across valid classes
    and the per-class AP values.

    Args:
       targets (torch.Tensor):
           Ground-truth labels of shape (num_samples, num_classes).
           Expected to contain binary or probabilistic labels.
       outputs (torch.Tensor):
           Model prediction scores of shape (num_samples, num_classes).
           Typically sigmoid outputs or probabilities.

    Returns:
       tuple:
           - float: Mean Average Precision (mAP) across all valid classes
             (those with at least one positive sample).
           - list[float]: Per-class AP values. For classes without
             positive samples, the value is set to -1.

    Notes:
       - Targets are binarized using a threshold of 0.5.
       - Classes with no positive ground-truth samples are excluded
         from the mean AP calculation.
       - Requires `average_precision_score` from sklearn.metrics.
   """
    np_targets = targets.cpu().numpy()
    np_outputs = outputs.cpu().numpy()
    num_classes = targets.shape[1]
    aps = []
    aps2 = []
    for i in range(num_classes):
        np_targets[:, i] = np.where(np_targets[:, i] > 0.5, 1, 0)

        if np.sum(np_targets[:, i]) > 0:
            class_ap = average_precision_score(np_targets[:, i], np_outputs[:, i])
            aps.append(class_ap)
            aps2.append(class_ap)
        else:
            aps2.append(-1)

    return np.mean(aps), aps2