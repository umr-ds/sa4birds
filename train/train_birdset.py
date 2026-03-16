import gc
import json
import logging
import os
import random
import sys
import warnings
import zipfile
from collections import Counter
from functools import partial
from typing import Tuple

import hydra
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Audio, Dataset, Sequence, Value, load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, get_scheduler

from train.transform import TrainTransform
from utils.event_mapper import XCEventMapping
from utils.loss import AsymmetricLossMultiLabel, FocalLossBCE
from utils.metric import TopKAccuracy, calculate_auc, calculate_map
from utils.transform import ValTransform
from validate_birdset import add_full_soundscape_path, build_model, test

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int =42) -> None:
    """
    Set global random seed for reproducibility.

    This function ensures deterministic behavior across Python,
    NumPy, and PyTorch by setting all relevant random seeds and
    configuring CUDA backend settings.

    Parameters
    ----------
    seed : int, optional
        Random seed value used across libraries.
        Default: 42.

    Effects
    -------
    - Sets Python built-in random seed.
    - Sets PYTHONHASHSEED for deterministic hashing.
    - Sets NumPy random seed.
    - Sets PyTorch CPU and CUDA seeds.
    - Forces deterministic CuDNN behavior.
    - Disables CuDNN benchmark mode for reproducibility.

    Notes
    -----
    - `torch.backends.cudnn.deterministic = True` ensures deterministic
      convolution algorithms but may reduce performance.
    - `torch.backends.cudnn.benchmark = False` prevents dynamic algorithm
      selection, improving reproducibility but potentially reducing speed.
    - Full reproducibility may still depend on:
        * PyTorch version
        * CUDA version
        * Hardware
        * Multi-GPU synchronization
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smart_sampling(dataset: Dataset,
                   label_name: str,
                   class_limit: int,
                   event_limit: int)  -> Dataset:
    """
   Perform class-balanced event sampling with per-file constraints.

   This function reduces dataset imbalance by limiting:

       1. The total number of samples per class (`class_limit`)
       2. The maximum number of events per file (`event_limit`)

   It operates by:
       - Creating unique file-label identifiers
       - Computing per-file event counts
       - Iteratively reducing overrepresented files
       - Randomly subsampling events to match the computed limits

   Parameters
   ----------
   dataset : Dataset
       HuggingFace dataset containing at least:
           - "filepath"
           - label_name column
   label_name : str
       Name of the label column.
   class_limit : int or None
       Maximum allowed number of samples per class.
       If None, no class-level limit is applied.
   event_limit : int
       Maximum allowed number of events per file.

   Returns
   -------
   Dataset
       Subsampled dataset with reduced imbalance.

   Notes
   -----
   - Sampling is random within each file.
   - Uses iterative reduction to enforce limits.
   - Assumes dataset fits into memory (converts to pandas).
   - Adds and removes a temporary "id" column.
   """

    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name), desc="unique_id",
    )
    df = dataset.to_pandas()
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="smart sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        c_event_limit = event_limit

        most = current["size"].max()

        while total > class_limit or most != c_event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                    total - (to_del * largest_count) < class_limit
                    or most == c_event_limit
                    or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                        path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()

    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)


def get_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    """
   Create and return a PyTorch optimizer instance based on the configuration.

   The optimizer type is selected using `cfg.train.optimizer`. Supported
   optimizers include Adam, AdamW, and SGD. Optimizer-specific parameters
   (such as momentum for SGD) are automatically applied.

   Args:
       model (torch.nn.Module): The model whose parameters will be optimized.
       cfg (object): Configuration object containing training parameters:
           - cfg.train.optimizer (str): Name of the optimizer ("Adam", "AdamW", "SGD").
           - cfg.train.lr (float): Learning rate.
           - cfg.train.weight_decay (float): Weight decay coefficient.

   Returns:
       torch.optim.Optimizer: An initialized optimizer instance.

   Raises:
       NotImplementedError: If the requested optimizer is not supported.
   """
    optimizers = {
        "Adam": (optim.Adam, {}),
        "AdamW": (optim.AdamW, {}),
        "SGD": (optim.SGD, {"momentum": 0.9}),
    }

    try:
        optimizer_cls, extra_kwargs = optimizers[cfg.train.optimizer]
    except KeyError:
        raise NotImplementedError(f"Optimizer {cfg.train.optimizer} not implemented")

    return optimizer_cls(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        **extra_kwargs
    )


def get_lr_scheduler(optimizer: Optimizer, cfg: DictConfig, steps_per_epoch: int) -> optim.lr_scheduler :
    """
    Create and return a learning rate scheduler.

    This function constructs a scheduler based on the configuration
    specified in `cfg.train.scheduler`. It supports cosine decay
    with optional warmup strategies.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    cfg : object
        Configuration object containing:
            - cfg.train.scheduler (str)
            - cfg.train.n_epochs (int)
            - cfg.train.num_warmup_epochs (int)
    steps_per_epoch : int
        Number of optimization steps per epoch.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Configured learning rate scheduler instance.
        Returns None if no scheduler is specified.

    Supported Schedulers
    --------------------
    "CosineWithWarmup"
        Cosine decay schedule with linear warmup.
        Uses `get_cosine_schedule_with_warmup`.

    "cosine_transformer"
        HuggingFace-style cosine scheduler with:
            - 5% warmup
            - 0.5 cosine cycles
        Uses `get_scheduler(name="cosine")`.

    Notes
    -----
    - Total training steps are computed as:
          total_steps = steps_per_epoch * n_epochs
    - Warmup steps for "CosineWithWarmup" are computed as:
          warmup_steps = steps_per_epoch * num_warmup_epochs
    - For "cosine_transformer", warmup is defined as a ratio (5%).
    """
    total_steps = steps_per_epoch * cfg.train.n_epochs
    warmup_steps = steps_per_epoch * cfg.train.num_warmup_epochs

    name = cfg.train.scheduler

    if name == "CosineWithWarmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    elif name == "cosine_transformer":
        warmup_ratio = 0.05

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_training_steps=total_steps,
            num_warmup_steps=int(total_steps * warmup_ratio),
            scheduler_specific_kwargs={
                "num_cycles": 0.5,
                "last_epoch": -1,
            },
        )

    else:
        scheduler = None

    return scheduler


def get_criterion(cfg: DictConfig) -> nn.Module:
    """
    Create and return a learning rate scheduler configured from training settings.

    The scheduler type is determined by `cfg.train.scheduler`. The function
    computes the total number of training steps and warmup steps from the
    training configuration and initializes the corresponding scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        cfg (object): Configuration object containing scheduler parameters:
            - cfg.train.scheduler (str): Scheduler name ("CosineWithWarmup", "cosine_transformer", etc.).
            - cfg.train.n_epochs (int): Total number of training epochs.
            - cfg.train.num_warmup_epochs (int): Number of warmup epochs.
        steps_per_epoch (int): Number of optimization steps in one epoch.

    Returns:
        tuple:
            - scheduler (torch.optim.lr_scheduler._LRScheduler | None): The initialized
              learning rate scheduler, or None if no scheduler is configured.
            - str: The scheduler step interval ("step").

    Notes:
        - "CosineWithWarmup" uses `get_cosine_schedule_with_warmup`.
        - "cosine_transformer" uses the HuggingFace `get_scheduler` utility
          with a cosine schedule and a fixed warmup ratio.
    """
    criterions = {
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "FocalLossBCE": FocalLossBCE,
        "AsymmetricLossMultiLabel": AsymmetricLossMultiLabel,
        "BCELoss": nn.BCELoss,
    }

    try:
        criterion_cls = criterions[cfg.train.criterion]
    except KeyError:
        raise NotImplementedError(
            f"Criterion {cfg.train.criterion} not implemented"
        )

    return criterion_cls()


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: Optimizer,
                    criterion: nn.Module,
                    cfg:DictConfig, scheduler: optim.lr_scheduler._LRScheduler | None =None) -> float:
    """
    Execute one full training epoch for a model.

    Performs the standard training loop consisting of forward pass, loss
    computation, backpropagation, and optimizer update for each batch.
    If provided, the learning rate scheduler is stepped after each batch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        loader (torch.utils.data.DataLoader): DataLoader yielding batches
            containing:
                - "audio": input tensors
                - "label": target tensors
        optimizer (torch.optim.Optimizer): Optimizer responsible for updating
            the model parameters.
        criterion (callable): Loss function used during training. Expected
            signature: `criterion(outputs, targets, activated=True)`.
        cfg (DictConfig): Configuration object containing training settings:
            - cfg.train.device (torch.device | str): Device used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning
            rate scheduler stepped after each optimizer update.

    Returns:
        - float: Average training loss across all batches.
    """
    model.train()
    device = cfg.train.device

    running_loss = 0.0
    progress_bar = tqdm(loader, total=len(loader), desc="Training")

    for step, batch in enumerate(progress_bar):
        inputs = batch["audio"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs, l_att, g_att = model(inputs)
        loss = criterion(outputs, targets, activated=True)
        if cfg.network.att_loss:
            entropy_l = - (l_att * torch.log(l_att + 1e-8)).sum(dim=-1).mean()
            entropy_g = - (g_att * torch.log(g_att + 1e-8)).sum(dim=-1).mean()
            loss = loss - 0.01 * (entropy_l + entropy_g)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        avg_loss = running_loss / (step + 1)

        postfix = {"train_loss": avg_loss}
        for i, g in enumerate(optimizer.param_groups):
            postfix[f"lr_g{i}"] = g["lr"]

        progress_bar.set_postfix(postfix)

    return running_loss / len(loader)


def run_training(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, cfg: DictConfig):
    """
    Execute the complete training pipeline including training, checkpointing,
    metric logging, and evaluation of an averaged model.

    The function performs the following steps:
    1. Initializes the optimizer, loss function, and learning rate scheduler.
    2. Iteratively trains the model for `cfg.train.n_epochs`.
    3. Logs training metrics and optionally saves checkpoints after each epoch.
    4. Maintains a moving average of model weights across epochs.
    5. Evaluates the averaged model on the validation dataset.
    6. Saves validation metrics and prints a summary of results.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader providing
            training batches.
        val_loader (torch.utils.data.DataLoader): DataLoader providing
            validation batches.
        model (torch.nn.Module): Model to be trained.
        cfg (DictConfig): Configuration object containing experiment settings,
            optimizer parameters, scheduler configuration, and output paths.

    Returns:
        None
    """

    # --------------------------------------------------
    # Prepare experiment paths
    # --------------------------------------------------
    exp_dir = cfg.train.exp_dir
    models_dir = os.path.join(exp_dir, "models")
    metrics_path = os.path.join(exp_dir, "metrics.json")

    # --------------------------------------------------
    # Initialize optimizer, loss function, and scheduler
    # --------------------------------------------------
    optimizer = get_optimizer(model, cfg)
    criterion = get_criterion(cfg)
    scheduler = get_lr_scheduler(
        optimizer, cfg, steps_per_epoch=len(train_loader)
    )

    # Dictionary to store metrics across epochs
    metrics = {}
    # Track the best AUC (currently unused but kept for extension)
    best_auc = 0

    # Store moving average of model weights ("model soup")
    moving_avg_state_dict = None

    # ==================================================
    #                Training Loop
    # ==================================================
    for epoch in range(cfg.train.n_epochs):
        logger.info(
            "Epoch %d/%d",
            epoch + 1,
            cfg.train.n_epochs,
        )

        # ----------------------------------------------
        # Train one full epoch
        # ----------------------------------------------
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg,
            scheduler=scheduler
        )

        metrics[epoch] = {"train_loss": train_loss}

        # Persist metrics after each epoch
        with open(metrics_path, "w") as file:
            json.dump(metrics, file)

        # Log learning rate
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Train Loss: %.4f", train_loss)
        logger.info("Learning Rate: %.7f", lr)

        # ----------------------------------------------
        # Save checkpoint (optional)
        # ----------------------------------------------
        if cfg.save_checkpoints:
            ckpt_path = os.path.join(models_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)



        # ----------------------------------------------
        # Update moving average of model weights
        # (simple arithmetic mean across epochs)
        # ----------------------------------------------
        current_state = model.state_dict()
        if moving_avg_state_dict is None:
            moving_avg_state_dict = current_state
        else:
            moving_avg_state_dict = average_state_dicts([moving_avg_state_dict, current_state])

    # ==================================================
    #        Evaluate Moving-Averaged Model
    # ==================================================
    # Load averaged weights into model
    model.load_state_dict(moving_avg_state_dict)

    # Save averaged model checkpoint
    if cfg.save_checkpoints:
        torch.save(model.state_dict(), os.path.join(cfg.train.exp_dir, "models", f"model_mv.pth"))

    # Validate averaged model
    mv_val_auc, mv_val_map, mv_val_top1_acc = test(model, val_loader, list(range(cfg.train.num_classes)),
                                          device=cfg.train.device)


    # Store final averaged ("soup") results
    metrics["soup"] = {"val_soup_auc": mv_val_auc,
                       "val_soup_map": mv_val_map,
                       "val_soup_acc": mv_val_top1_acc}

    # Save final metrics
    with open(os.path.join(cfg.train.exp_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file)

    # --------------------------------------------------
    # Cleanup to free GPU memory
    # --------------------------------------------------
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    # --------------------------------------------------
    # Final summary print (stderr for logging systems)
    # --------------------------------------------------
    logger.info("=" * 60)
    logger.info("Test Results")
    logger.info("Dataset: %s", cfg.train.dataset_name)
    logger.info("SEED: %s", cfg.seed)
    logger.info("AUROC: %.4f", mv_val_auc)
    logger.info("MAP: %.4f", mv_val_map)
    logger.info("TOP1-ACC: %.4f", mv_val_top1_acc)
    logger.info("=" * 60)


def average_state_dicts(state_dicts: list[dict]):
    """
    Compute the element-wise average of multiple PyTorch state dictionaries.

    This function averages model parameters across several state_dicts,
    assuming all models share identical architectures and parameter keys.

    Parameters
    ----------
    state_dicts : list[dict]
        List of PyTorch state_dict objects (as returned by
        `model.state_dict()`), each containing identical keys
        and tensor shapes.

    Returns
    -------
    dict
        A new state_dict where each parameter tensor is the
        arithmetic mean across the provided models.

    Raises
    ------
    ValueError
        If the input list is empty.
    AssertionError
        If any state_dict is missing a key present in the first one.

    Notes
    -----
    - All models must have identical architectures.
    - All tensors must have matching shapes.
    - The averaging is performed as:

          averaged_param = sum(param_i) / N

      where N is the number of models.

    - Commonly used for:
        * Model ensembling
        * Checkpoint averaging
        * SWA-style parameter smoothing
    """
    if not state_dicts:
        raise ValueError("No state_dicts provided for averaging")

    num_models = len(state_dicts)
    averaged_state_dict = {}

    # Initialize the averaged_state_dict with zeros
    for key in state_dicts[0].keys():
        # Ensure all state_dicts have the same keys
        for sd in state_dicts:
            assert key in sd, f"Key {key} missing in one of the state_dicts"

        # Sum up all the tensors
        averaged_state_dict[key] = sum(sd[key] for sd in state_dicts) / num_models

    return averaged_state_dict


def n_hot(batch: dict, num_classes: int=21, label_map: dict=None, secondary_label_weight: int=0) -> dict:
    """
    Convert label indices into n-hot encoded targets.

    This function transforms primary class labels into an n-hot
    representation and optionally incorporates secondary labels
    with a configurable weight.

    Parameters
    ----------
    batch : dict
        Batch dictionary containing:
            - "labels": iterable of primary label indices
              (int or iterable of int per sample)
            - optionally "ebird_code_secondary": iterable of
              secondary labels (list per sample)
    num_classes : int, optional
        Total number of classes. Default: 21.
    label_map : dict, optional
        Mapping from secondary label names (e.g., strings)
        to class indices. Required if secondary labels are used.
    secondary_label_weight : float, optional
        Weight assigned to secondary labels.
        - 0.0 → ignore secondary labels
        - >0.0 → assign this value to secondary class positions

    Returns
    -------
    dict
        {
            "labels": torch.FloatTensor of shape
                      (batch_size, num_classes)
        }

    Notes
    -----
    - Primary labels are assigned value 1.0.
    - Secondary labels are assigned `secondary_label_weight`.
    - If a sample has multiple primary labels, all are set to 1.
    - If `label_map` is None, secondary labels are ignored.
    """
    label_list = [y for y in batch["labels"]]
    class_one_hot_matrix = torch.zeros(
        (len(label_list), num_classes), dtype=torch.float
    )

    # --------------------------------------------------
    # Primary labels
    # --------------------------------------------------
    for class_idx, idx in enumerate(label_list):
        class_one_hot_matrix[class_idx, idx] = 1

    # --------------------------------------------------
    # Secondary labels (optional)
    # --------------------------------------------------
    if secondary_label_weight > 0 and "ebird_code_secondary" in batch:
        if label_map is not None:
            sec_label_list = [y for y in batch["ebird_code_secondary"]]
            for i, idx in enumerate(sec_label_list):
                if idx is not None:
                    for j in idx:
                        if j in label_map:
                            class_one_hot_matrix[i, label_map[j]] = secondary_label_weight

    class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
    return {"labels": class_one_hot_matrix}


def get_train_val_loader(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    This function performs the full dataset preparation pipeline:

       1. Load BirdSet dataset
       2. Apply event mapping for training split
       3. Perform smart sampling (class balancing)
       4. Generate label mapping
       5. Select required columns
       6. Convert labels to n-hot encoding
       7. Attach full soundscape paths (validation)
       8. Apply train/validation transforms
       9. Construct PyTorch DataLoaders

    Parameters
    ----------
    config : object
       Configuration object containing:
           - config.train.dataset_name
           - config.train.class_limit
           - config.train.event_limit
           - config.train.total_thresh
           - config.train.secondary_label_weight
           - config.train.batch_size
           - config.train.num_workers
           - config.columns
           - config.event_decoder.train
           - config.event_decoder.val

    Returns
    -------
    tuple
       (train_loader, val_loader)

       train_loader : torch.utils.data.DataLoader
       val_loader   : torch.utils.data.DataLoader

    Side Effects
    ------------
    - Modifies:
       config.train.label_map
       config.train.num_classes

    Notes
    -----
    - Uses HuggingFace datasets.
    - Applies smart class balancing to training data.
    - Uses n-hot encoding for multilabel targets.
    - Applies different transforms for train and validation.
    """

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    dataset = load_dataset("DBD-research-group/BirdSet", config.train.dataset_name)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=config.frontend.sample_rate, mono=True, decode=False))

    train_data = dataset["train"]

    # --------------------------------------------------
    # 2. Event mapping (expand file-level to event-level)
    # --------------------------------------------------
    mapper = XCEventMapping(n_time_random_sample_per_file=1)

    train_data = train_data.map(
        mapper,
        remove_columns=["audio"],
        batched=True,
        batch_size=300,
        load_from_cache_file=False,
        num_proc=16,
        desc="Train event mapping",
    )

    # --------------------------------------------------
    # 3. Smart sampling (class balancing)
    # --------------------------------------------------
    train_data = smart_sampling(
        dataset=train_data,
        label_name="ebird_code",
        class_limit=config.train.class_limit,
        event_limit=config.train.event_limit)

    # --------------------------------------------------
    # 4. Generate label map
    # --------------------------------------------------
    features = train_data.features
    label_feature = features["ebird_code_multilabel"]
    labels = label_feature.feature.names

    label_map = {c: i for i, c in enumerate(labels)}

    # reset labels for the additional data
    config.train.label_map = label_map
    config.train.num_classes = len(labels)

    train_data = train_data.select_columns(column_names=config.columns)

    # --------------------------------------------------
    # 5. Validation split
    # --------------------------------------------------
    val_data = dataset['test_5s']
    val_data = val_data.select_columns(column_names=config.columns)

    logger.info("Training events: %d", len(train_data))
    logger.info("Validation events: %d", len(val_data))
    logger.info("Number of classes: %d", config.train.num_classes)

    train_data = train_data.rename_column("ebird_code_multilabel", "labels")
    val_data = val_data.rename_column("ebird_code_multilabel", "labels")

    label_counts = Counter(train_data['ebird_code'])
    logger.info("Label distribution: %s", label_counts)
    train_data = train_data.cast_column("labels", Sequence(Value("float32")))

    # --------------------------------------------------
    # 6. N-hot encoding
    # --------------------------------------------------
    logger.info("Encoding labels to n-hot vectors")
    train_data = train_data.map(
        partial(n_hot,
                num_classes=config.train.num_classes,
                label_map=label_map,
                secondary_label_weight=config.train.secondary_label_weight),
        batched=True,
        batch_size=300,
        load_from_cache_file=False,
        num_proc=1,
    )

    # --------------------------------------------------
    # 7. Add full soundscape path for validation
    # --------------------------------------------------
    val_data = val_data.map(
        partial(n_hot, num_classes=config.train.num_classes, ),
        batched=True,
        batch_size=300,
        load_from_cache_file=False,
        num_proc=1,
    )

    val_data = add_full_soundscape_path(dataset, val_data)

    # --------------------------------------------------
    # 8. Transforms
    # --------------------------------------------------
    train_transform = TrainTransform(
        config=config,
        train=True,
        event_decoder=hydra.utils.instantiate(config.event_decoder.train)
    )

    val_transform = ValTransform(
        config=config,
        train=False,
        event_decoder=hydra.utils.instantiate(config.event_decoder.val)
    )

    train_data.set_transform(train_transform)
    val_data.set_transform(val_transform)

    # --------------------------------------------------
    # 9. DataLoaders
    # --------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=config.train.num_workers,
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=config.train.num_workers,
        batch_size=config.train.batch_size,
        drop_last=False,
        shuffle=False
    )

    return train_loader, val_loader


def download_ckpt(url, extract_dir):
    zip_filename = "file.zip"

    # -----------------------
    # Download ZIP file
    # -----------------------
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete.")

    # -----------------------
    # Extract ZIP file
    # -----------------------
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Unzipped to '{extract_dir}'")


@hydra.main(version_base=None, config_path="../configs", config_name="base_finetune")
def main(cfg: DictConfig):
    """
    Main entry point for supervised fine-tuning.

    This function is executed via Hydra and orchestrates the full
    fine-tuning workflow using a configuration-driven setup.

    Pipeline Overview
    -----------------
    1. Print resolved Hydra configuration
    2. Set global random seed for reproducibility
    3. Create training and validation dataloaders
    4. Build model architecture
    5. Load pretrained checkpoint weights
    6. Create experiment directory and save configuration
    7. Launch supervised training loop

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing:
            - dataset configuration
            - frontend parameters
            - network architecture settings
            - training hyperparameters
            - checkpoint paths
            - experiment directory paths

    Side Effects
    ------------
    - Creates experiment directory if it does not exist.
    - Saves resolved configuration to:
        <cfg.train.exp_dir>/config.yaml
    - Loads pretrained weights into model (non-strict).
    - Starts training process.

    Notes
    -----
    - Uses `strict=False` when loading pretrained weights,
      allowing partial weight loading.
    - Hydra automatically manages working directory changes.
    - Requires:
        - get_train_val_loader
        - build_model
        - run_training
        - set_seed
    """
    logger.info("\nResolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    # ------------------------------------------
    # 2. Set seed
    # ------------------------------------------
    set_seed(cfg.seed)

    # ------------------------------------------
    # 3. Create dataloaders
    # ------------------------------------------
    train_loader, val_loader = get_train_val_loader(cfg)

    # ------------------------------------------
    # 4. Build model
    # ------------------------------------------
    audio_model = build_model(cfg)

    # ------------------------------------------
    # 5. Load pretrained checkpoint
    # ------------------------------------------
    logger.info("Loading pretrained checkpoint: %s", cfg.network.pretrain_checkpoint)
    if not os.path.exists(cfg.network.pretrain_checkpoint):
        logger.info("Downloading pretrained checkpoint")
        download_ckpt('https://next.hessenbox.de/index.php/s/F5RWCp9ppegTigo/download',
                      extract_dir='checkpoints/')

    state_dict = torch.load(cfg.network.pretrain_checkpoint)['state_dict']
    audio_model.load_state_dict(state_dict, strict=False)

    # ------------------------------------------
    # 6. Prepare experiment directory
    # ------------------------------------------
    exp_dir = cfg.train.exp_dir
    models_dir = os.path.join(exp_dir, "models")
    logger.info("Creating experiment directory: %s", exp_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    OmegaConf.save(cfg, f"{exp_dir}/config.yaml")

    # ------------------------------------------
    # 7. Launch training
    # ------------------------------------------
    logger.info(
        "Starting supervised fine-tuning for %d epochs",
        cfg.train.n_epochs,
    )
    run_training(train_loader, val_loader, audio_model, cfg)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
