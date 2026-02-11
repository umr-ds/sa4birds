import argparse
import sys
from functools import partial

import numpy as np
import torch
from datasets import load_dataset, Audio
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.event_decoder import EventDecoding
from models.dsa import DSA
from utils.transform import ValTransform
from utils.metric import calculate_auc, calculate_map, TopKAccuracy
from checkpoints import DT, MT, LT
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def to_1hot(batch, num_classes=21):
    labels = [y for y in batch["labels"]]
    labels_1hot = torch.zeros( (len(labels), num_classes), dtype=torch.float32 )
    for i, idx in enumerate(labels):
        labels_1hot[i, idx] = 1
    return {"labels": labels_1hot}


def add_full_soundscape_path(dataset, val_data):
    soundscape_filepaths = [x['filepath'] for x in dataset['test']]

    def add_soundscape_path(item, filepaths):
        filename = ("_".join(item['filepath'].split("_")[:-2]) + '.ogg').split('/')[-1]
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
    # Load and process dataset
    dataset = load_dataset("DBD-research-group/BirdSet", config.train.dataset_name)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000, decode=False)) # mono=True
    class_names = dataset['train'].features['ebird_code_multilabel'].feature.names

    val_data = dataset['test_5s']
    val_data = val_data.select_columns(column_names=config.columns)

    print("number of test events in {}: {}".format(config.train.dataset_name,len(val_data)))
    val_data = val_data.rename_column("ebird_code_multilabel", "labels")

    val_data = val_data.map(
        partial(to_1hot, num_classes=len(class_names),),
        batched=True,
        batch_size=300,
        load_from_cache_file=False,
        num_proc=1,
    )

    val_data = add_full_soundscape_path(dataset, val_data)
    val_transform = ValTransform(
        config=config,
        train=False,
        event_decoder=EventDecoding(extracted_interval=config.event_decoder.val.extracted_interval),
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


def load_model(checkpoint):
    state_dict = torch.load(f"{checkpoint}/models/model.pth")
    state_dict, config = state_dict['state_dict'], state_dict['config']
    model = build_model(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config




def build_model(cfg):
    if cfg.network.classifier == "DSA":
        base_model = DSA(cfg)
    else:
        raise NotImplementedError

    base_model.to(cfg.train.device)
    return base_model



def test(model, loader, cfg, relevant_ids):
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            inputs = batch['audio'].to(cfg.train.device)
            targets = batch['label'].to(cfg.train.device)

            outputs = model(inputs, True)[:,relevant_ids]
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



def evaluate(mode=DT, down_task="HSN"):
    ckpts = mode.get(down_task, [])
    if not ckpts:
        ckpts = mode['ALL']
    if not ckpts:
        raise ValueError("No checkpoints found.")

    models = []
    configs = []

    # Load configs & models
    for ckpt in ckpts:
        m, cfg = load_model(ckpt)
        models.append(m)
        configs.append(cfg)

    results = dict()

    configs[0].frontend.val_target_length = 701
    configs[0].event_decoder.val.extracted_interval = 7

    configs[0].train.dataset_name = down_task
    val_loader, class_names = get_test_loader(configs[0])

    label2id = {k: v for k, v in configs[0].train.label_map.items()}
    relevant = [label2id[c] for c in class_names]

    metrics = {"auroc": [], "cmap": [], "top1_acc": []}

    # Evaluate
    for model, cfg in zip(models, configs):
        auroc, cmap, top1_acc = test(model, val_loader, cfg, relevant)
        metrics["auroc"].append(auroc)
        metrics["cmap"].append(cmap)
        metrics["top1_acc"].append(top1_acc)

    # Aggregate
    print(metrics)
    results[down_task] = {key: float(np.mean(values)) for key, values in metrics.items()}
    print(results)
    metrics = results[next(iter(results))].keys()
    # Compute averages
    averages = {}
    for m in metrics:
        averages[m] = sum(d[m] for d in results.values()) / len(results)
    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example Python script with parameters")
    parser.add_argument("--mode", choices=["DT", "MT", "LT"], default="DT")
    parser.add_argument("--down_task", choices=["HSN", "POW", "NES", "NBP", "UHH", "SNE", "SSW", "PER", "ALL"], default="HSN")
    args = parser.parse_args()
    if args.down_task == "ALL":
        res = []
        for task in ["HSN", "POW", "NES", "NBP", "UHH", "SNE", "SSW", "PER"]:
            o = evaluate(mode=getattr(sys.modules[__name__], args.mode), down_task=task)
            res.append({f"{task}": o})
            print(o)
        print(res)
    else:
        res = evaluate(mode=getattr(sys.modules[__name__], args.mode), down_task=args.down_task)
        print(res)