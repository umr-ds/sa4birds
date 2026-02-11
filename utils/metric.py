import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics import AUROC, Metric
from torchmetrics.classification import MultilabelAveragePrecision
import torch

class TopKAccuracy(Metric):
    def __init__(self, topk=1, include_nocalls=False, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        # Get the top-k predictions
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
        targets = targets.to(preds.device)
        no_call_targets = targets.sum(dim=1) == 0

        # consider no_call instances (a threshold is needed here!)
        if self.include_nocalls:
            # check if top-k predictions for all-negative instances are less than threshold
            no_positive_predictions = (
                preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            )
            correct_all_negative = no_call_targets & no_positive_predictions.all(dim=1)

        else:
            # no_calls are removed, set to 0
            correct_all_negative = torch.tensor(0).to(targets.device)

        # convert one-hot encoded targets to class indices for positive cases
        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        correct_positive = expanded_targets.gather(
            2, topk_pred_indices.unsqueeze(-1)
        ).any(dim=1)

        # update correct and total, excluding all-negative instances if specified
        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            self.total += targets.size(0)

    def compute(self):
        return self.correct.float() / self.total


def calculate_auc(targets, outputs):
    # auroc_fn = AUROC(
    #     task="multilabel",
    #     num_labels=targets.shape[-1],
    #     average="macro",
    #     thresholds=None,
    # )
    np_targets = targets.cpu().numpy()
    np_outputs = outputs.cpu().numpy()
    num_classes = np_targets.shape[1]
    aucs = []
    aucs2 = []

    # probs = 1 / (1 + np.exp(-np_outputs))

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
    np_targets = targets.cpu().numpy()
    np_outputs = outputs.cpu().numpy()
    num_classes = targets.shape[1]
    aps = []
    aps2 = []
    # probs = 1 / (1 + np.exp(-np_outputs))
    for i in range(num_classes):
        np_targets[:, i] = np.where(np_targets[:, i] > 0.5, 1, 0)

        if np.sum(np_targets[:, i]) > 0:
            class_ap = average_precision_score(np_targets[:, i], np_outputs[:, i])
            aps.append(class_ap)
            aps2.append(class_ap)
        else:
            aps2.append(-1)

    #cmap = cmAP(targets.shape[-1])
    #print(cmap)
    return np.mean(aps), aps2


class cmAP(MultilabelAveragePrecision):
    def __init__(self, num_labels, thresholds=None):
        super().__init__(num_labels=num_labels, average="macro", thresholds=thresholds)

    def __call__(self, logits, labels):
        macro_cmap = super().__call__(logits, labels)
        return macro_cmap
