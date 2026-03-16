import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Audio
from tqdm import tqdm

from models.linear import LinClsModel
from models.ssa import SSA
from models.timeattention import TimeAttModel
from utils.event_decoder import EventDecoder
from models.dsa import DSA
from utils.transform import ValTransform
from utils.metric import calculate_auc, calculate_map, TopKAccuracy
from checkpoints import DT, MT, LT
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# BirdSet Down tasks
TASKS = ["HSN", "POW", "NES", "NBP", "UHH", "SNE", "SSW", "PER"]

REGIMES = ["DT", "MT", "LT"]


def parse_args():
    """
    Parse command-line arguments for evaluation script.

    This function defines and parses arguments required to run
    model evaluation under different training regimes and
    downstream tasks.

    Arguments
    ---------
    --mode : str, optional
        Training regime to evaluate.
        Choices:
            - "DT" : Downstream Task models
            - "MT" : Multi-Task models
            - "LT" : Long-Term training models
        Default: "DT"

    --down_task : str, optional
        Downstream task to evaluate.
        Choices:
            - Individual task names defined in TASKS
            - "ALL" to evaluate all tasks
        Default: "HSN"

    --cpu : flag, optional
        If provided, forces computation on CPU even if a GPU
        is available. Default behavior uses GPU when available.

    --num_workers : int, optional
        Number of worker processes for DataLoader.
        Higher values can improve loading speed but increase
        memory usage. Default: 12.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation for a selected downstream task under a specified training regime.")
    parser.add_argument(
        "--mode",
        choices=["DT", "MT", "LT"],
        default="DT",
        help="Regime to use: DT, MT, or LT (default: DT)"
    )
    parser.add_argument(
        "--down_task",
        choices=["ALL"] + TASKS,
        default="HSN",
        help="Specify which downstream task to execute; use ALL to run every task (default: HSN)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force computation on CPU instead of GPU (default: use GPU if available)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of worker processes used for loading data batches in parallel (default: 12)"
    )

    return parser.parse_args()


def get_device(force_cpu: bool):
    """
    Determine the computation device (CPU or GPU).

    This function selects the appropriate PyTorch device
    depending on user preference and hardware availability.

    Parameters
    ----------
    force_cpu : bool
        If True, forces computation on CPU even if a CUDA-capable
        GPU is available.

    Returns
    -------
    torch.device
        - "cpu"  : if forced or no GPU is available.
        - "cuda" : if a CUDA-capable GPU is available and
                   `force_cpu` is False.

    Notes
    -----
    - Uses `torch.cuda.is_available()` to detect GPU support.
    - Does not handle multi-GPU selection explicitly.
    """
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def to_n_hot(batch, num_classes=21):
    """
    Convert a batch of class indices into an n-hot encoded tensor.

    This function takes a batch dictionary containing a "labels" key
    with class indices and converts them into a tensor of shape
    (batch_size, num_classes), where each row is an n-hot vector
    (i.e., positions corresponding to class indices are set to 1).

    Args:
        batch (dict): A dictionary containing:
            - "labels" (Iterable[int] or Iterable[Tensor]): Class indices
              for each sample in the batch.
        num_classes (int, optional): Total number of possible classes.
            Defaults to 21.

    Returns:
        dict: A dictionary with:
            - "labels" (torch.FloatTensor): A tensor of shape
              (batch_size, num_classes) containing n-hot encoded labels.
    """
    labels = [y for y in batch["labels"]]
    labels_n_hot = torch.zeros((len(labels), num_classes), dtype=torch.float32)
    for i, idx in enumerate(labels):
        labels_n_hot[i, idx] = 1
    return {"labels": labels_n_hot}


def add_full_soundscape_path(dataset, val_data):
    """
    Add the full soundscape file path to each validation item.

    This function matches each validation sample to its corresponding
    full-length soundscape file from the dataset's test split. It
    reconstructs the expected soundscape filename from the sample's
    segmented filepath and searches for a matching file in the test
    filepaths. The matched path is stored under the key
    "soundscape_filepath".

    Args:
        dataset (dict-like): A dataset object containing a "test" split.
            Each element in dataset["test"] must include a "filepath" key.
        val_data (Dataset): A dataset (e.g., Hugging Face Dataset) where
            each item contains a "filepath" key corresponding to a
            segmented audio file.

    Returns:
        Dataset: The updated validation dataset where each item includes
        an additional key:
            - "soundscape_filepath" (str): The matched full soundscape path.

    Raises:
        AssertionError: If no matching soundscape file is found for a
        validation sample.
    """
    soundscape_filepaths = [x['filepath'] for x in dataset['test']]

    def add_soundscape_path(item, filepaths):
        filename = Path("_".join(item['filepath'].split("_")[:-2]) + ".ogg").name
        found = False
        for f in filepaths:
            if filename in f:
                item['soundscape_filepath'] = f
                found = True
        assert found
        return item

    val_data = val_data.map(partial(add_soundscape_path, filepaths=soundscape_filepaths))
    return val_data


def get_test_loader(config):
    """
    Build and return the test DataLoader and class names for evaluation.

    This function:
       1. Loads the BirdSet downtask.
       2. Casts the audio column to a fixed sampling rate (32 kHz).
       3. Extracts class names from the test split metadata.
       4. Selects and preprocesses the 5-second test segments.
       5. Converts multilabel annotations to n-hot encoding.
       6. Adds the corresponding full soundscape file paths.
       7. Applies validation transforms.
       8. Wraps the dataset in a PyTorch DataLoader.

    Args:
       config (object): Configuration object containing at least:
           - config.train.dataset_name (str): Name of the dataset configuration.
           - config.columns (list[str]): Columns to keep in the test split.
           - config.train.num_workers (int): Number of DataLoader workers.
           - config.event_decoder.val.extracted_interval: Interval parameter
             used for event decoding.
           - Any additional fields required by ValTransform.

    Returns:
       tuple:
           - torch.utils.data.DataLoader: DataLoader for the processed
             test dataset.
           - list[str]: List of class names corresponding to label indices.

    Notes:
       - The function assumes the presence of the following utilities:
           * `to_n_hot` for label encoding.
           * `add_full_soundscape_path` for matching full soundscape files.
           * `ValTransform` for dataset transformations.
           * `EventDecoding` for event interval processing.
       - The DataLoader uses a fixed batch size of 128 and does not shuffle.
    """
    dataset = load_dataset("DBD-research-group/BirdSet",
                           config.train.dataset_name)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000, decode=False))

    class_names = dataset['test'].features['ebird_code_multilabel'].feature.names

    val_data = dataset['test_5s']
    val_data = val_data.sort('filepath')
    val_data = val_data.select_columns(column_names=config.columns)

    logger.info("Task: %s | Number of events: %s", config.train.dataset_name, len(val_data))

    val_data = val_data.rename_column("ebird_code_multilabel", "labels")

    val_data = val_data.map(
        partial(to_n_hot, num_classes=len(class_names), ),
        batched=True,
        batch_size=300,
        load_from_cache_file=False,
        num_proc=1,
    )

    val_data = add_full_soundscape_path(dataset, val_data)
    val_transform = ValTransform(
        config=config,
        train=False,
        event_decoder=EventDecoder(extracted_interval=config.event_decoder.val.extracted_interval),
    )

    val_data.set_transform(val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=config.train.num_workers,
        batch_size=128,
        drop_last=False,
        shuffle=False
    )

    return val_loader, class_names


def load_model(checkpoint, device=torch.device('cuda')):
    """
    Load a trained model and its configuration from a checkpoint directory.

    This function:
        1. Loads the serialized checkpoint file located at
           "<checkpoint>/models/model.pth".
        2. Extracts the model state dictionary and configuration.
        3. Reconstructs the model using the stored configuration.
        4. Loads the saved weights into the model.
        5. Sets the model to evaluation mode.

    Args:
        checkpoint (str): Path to the checkpoint directory containing
            "models/model.pth".
        device (str or torch.device): Target device
              (e.g., "cpu", "cuda") to move the model to.
    Returns:
        tuple:
            - torch.nn.Module: The reconstructed model with loaded weights,
              set to evaluation mode.
            - object: The configuration object stored in the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If the checkpoint file does not contain the expected
            "state_dict" or "config" entries.

    Notes:
        - Assumes the checkpoint file is a dictionary containing:
            * "state_dict": Model parameters.
            * "config": Configuration used to build the model.
        - Requires `build_model(config)` to reconstruct the architecture.
    """
    # state_dict = torch.load(f"{checkpoint}/models/model.pth")
    state_dict = torch.load(f"{checkpoint}/models/model.pth", map_location=device)
    state_dict, config = state_dict['state_dict'], state_dict['config']
    model = build_model(config, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def build_model(cfg, device='cuda'):
    """
    Construct and return a model instance based on the configuration.

    This function selects the model architecture specified in the
    configuration and initializes it. Currently, only the "DSA"
    classifier is supported. The constructed model is moved to the
    device defined in the training configuration.

    Args:
        cfg (object): Configuration object containing at least:
            - cfg.network.classifier (str): Name of the classifier
              architecture to build.
        device (str or torch.device): Target device
              (e.g., "cpu", "cuda") to move the model to.

    Returns:
        torch.nn.Module: The initialized model moved to the specified device.

    Raises:
        NotImplementedError: If the requested classifier architecture
        is not supported.
    """
    #print(cfg)
    if cfg.network.classifier == "DSA":
        base_model = DSA(cfg)
    elif cfg.network.classifier == "SSA":
        base_model = SSA(cfg)
    elif cfg.network.classifier == "linear":
        base_model = LinClsModel(cfg)
    elif cfg.network.classifier == "timeattention":
        base_model = TimeAttModel(cfg)
    else:
        raise NotImplementedError

    base_model.to(device)
    return base_model


def test(model, loader, relevant_ids, center_5s=True, device='cuda'):
    """
    Evaluate a trained model on a test dataset.

    This function performs inference on the provided DataLoader,
    collects model predictions and ground-truth targets, and computes
    evaluation metrics including AUC, mAP, and top-1 accuracy.

    Args:
      model (torch.nn.Module): Trained model to evaluate.
      loader (torch.utils.data.DataLoader): DataLoader providing
          test batches. Each batch must contain:
              - 'audio': Input tensor.
              - 'label': Ground-truth labels.
      relevant_ids (list or Tensor): Indices of relevant class
          outputs to select from the model predictions.
      center_5s (bool): predict only the center 5s
      device (str or torch.device): Device used
          for computation.

    Returns:
      tuple:
          - float: Macro AUC score across classes.
          - float: Mean Average Precision (mAP).
          - float: Top-1 accuracy.

    Notes:
      - The model is set to evaluation mode (`model.eval()`).
      - Gradients are disabled using `torch.no_grad()`.
      - Only the columns corresponding to `relevant_ids` are used
        from the model outputs.
      - Requires the following external utilities:
          * `calculate_auc`
          * `calculate_map`
          * `TopKAccuracy`
      - Assumes model forward signature: `model(inputs, inference_flag)`
        where the second argument is set to True during testing.
    """
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            inputs = batch['audio'].to(device)
            targets = batch['label'].to(device)

            outputs = model(inputs, center_5s)[:, relevant_ids]
            outputs = outputs.detach()
            targets = targets.detach()

            all_outputs.append(outputs)
            all_targets.append(targets)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    auc, aucs = calculate_auc(all_targets, all_outputs)
    map, aps = calculate_map(all_targets, all_outputs)

    ACC = TopKAccuracy()
    top1_acc = float(ACC(all_outputs.cpu(), all_targets.cpu()))
    return auc, map, top1_acc


def run_testing(mode=DT, down_task="HSN", device='cuda'):
    """
    Evaluate one or multiple checkpoints on a downstream task and
    return aggregated performance metrics.

    This function:
       1. Retrieves checkpoint paths from `mode` for the given downstream task.
          Falls back to mode['ALL'] if no task-specific checkpoints exist.
       2. Loads all corresponding models and configurations.
       3. Builds the test DataLoader for the downstream task.
       4. Identifies relevant class indices shared between the model
          label map and dataset class names.
       5. Evaluates each model.
       6. Aggregates metrics across checkpoints (mean).
       7. Returns averaged metrics across tasks.

    Args:
       mode (dict, optional): Dictionary mapping downstream task names
           to lists of checkpoint paths. Must contain either:
               - mode[down_task], or
               - mode["ALL"] as fallback.
           Defaults to DT.
       down_task (str, optional): Name of the downstream task (dataset)
           to evaluate on. Defaults to "HSN".
       device (str or torch.device): Device used
          for computation.

    Returns:
       dict: Dictionary containing averaged evaluation metrics:
           - "auroc" (float): Mean AUROC across checkpoints.
           - "cmap" (float): Mean mAP across checkpoints.
           - "top1_acc" (float): Mean Top-1 accuracy across checkpoints.

    Raises:
       ValueError: If no checkpoints are found for the task and no
           fallback checkpoints exist.

    Notes:
       - The first loaded configuration is modified to:
           * Set validation target length.
           * Adjust event decoding interval.
           * Override dataset name to match `down_task`.
       - Assumes the existence of the following functions:
           * `load_model`
           * `get_test_loader`
           * `test`
       - Metrics are averaged across all checkpoints associated with
         the downstream task.
    """
    ckpts = mode.get(down_task, [])
    if not ckpts:
        ckpts = mode['ALL']
    if not ckpts:
        raise ValueError("No checkpoints found.")

    models = []
    configs = []

    # Load configs & models
    for ckpt in ckpts:
        m, cfg = load_model(ckpt, device=device)
        models.append(m)
        cfg.frontend.val_target_length = 701
        cfg.event_decoder.val.extracted_interval = 7
        cfg.train.dataset_name = down_task
        configs.append(cfg)

    results = dict()

    val_loader, class_names = get_test_loader(configs[0])

    label2id = {k: v for k, v in configs[0].train.label_map.items()}
    relevant = [label2id[c] for c in class_names]

    metrics = {"auroc": [],
               "cmap": [],
               "top1_acc": []}

    # test models
    for model, cfg in zip(models, configs):
        auroc, cmap, top1_acc = test(model, val_loader, relevant, device=device)
        metrics["auroc"].append(auroc)
        metrics["cmap"].append(cmap)
        metrics["top1_acc"].append(top1_acc)

    # Aggregate
    results[down_task] = {key: float(np.mean(values)) for key, values in metrics.items()}
    return results


def run_tasks(mode, task, device):
    """
    Execute evaluation for one or multiple downstream tasks.

    This function runs testing under a specified training regime
    (DT, MT, or LT) and logs evaluation results for each task.

    Parameters
    ----------
    mode : str
      Training regime identifier (e.g., "DT", "MT", "LT").
      Must correspond to a global variable containing
      checkpoint mappings.
    task : str
      Downstream task name or "ALL".
      - If "ALL", evaluation runs on every task in TASKS.
      - Otherwise, runs only the specified task.
    device : torch.device
      Device used for computation (CPU or CUDA).

    Returns
    -------
    list[dict]
      List of result dictionaries, one per evaluated task.
      Each result contains:
          {
              <task_name>: {
                  "auroc": float,
                  "cmap": float,
                  "top1_acc": float
              }
          }

    Notes
    -----
    - Uses `run_testing(...)` to perform actual evaluation.
    - Logs regime, task name, and key metrics.
    - Expects `TASKS`, `logger`, and `run_testing`
    to be defined in the global scope.
    """
    tasks = TASKS if task == "ALL" else [task]
    logger.info("Running evaluation")
    logger.info("Regime: %s | Device: %s | Tasks: %s", mode, device, tasks)
    results = []
    for t in tasks:
        result = run_testing(mode=globals()[mode], down_task=t, device=device)
        results.append(result)
        logger.info("Regime: %s | Task: %s | AUROC: %s | CMAP: %s | TOP1-ACC: %s ",
                    mode, t, result[t]['auroc'], result[t]['cmap'], result[t]['top1_acc'])
    return results


if __name__ == '__main__':
    args = parse_args()
    device = get_device(args.cpu)
    results = run_tasks(args.mode, args.down_task, device)
    print(results)
