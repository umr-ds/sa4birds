# Training Guide

This document describes how to train models using the **Dedicated Training (DT)** regime on the **BirdSet benchmark**.

The guide covers:

- Hardware and software requirements  
- Additional datasets required for augmentation  
- Dataset preparation  
- Running the training pipeline  

For evaluation instructions, see the [main project README](../README.md).

---

# Hardware Requirements

Training is designed to run on a **CUDA-capable GPU**.

While CPU execution is technically possible, it is **not recommended** due to significantly slower training times.

Recommended setup:
- **Operating System:** Ubuntu or other Debian-based Linux distribution (tested on Ubuntu 24.04.4 LTS)
- **NVIDIA GPU with CUDA support:** ≥ 24 GB GPU memory

---

# Software Requirements

Training requires the same dependencies as evaluation (see the [project guide](../README.md)), plus the following additional Python packages:

- `audiomentations`
- `torch_audiomentations`

Install the additional packages with:

```bash
pip install -r additional_requirements.txt
```

---

# Additional Datasets

## Negative Samples (No-Call)

Training uses several environmental audio datasets as **negative samples** for **data augmentation and robustness training**.

The following datasets are used as no-calls samples during training:

- [Freefield1010](https://dcase.community/challenge2018/task-bird-audio-detection)
- [ESC-50](https://github.com/karolpiczak/ESC-50) 
- [BirdVox-DCASE-20k](https://dcase.community/challenge2018/task-bird-audio-detection) 

### iNaturalist

We additionally crawl negative samples from **iNaturalist**, primarily containing recordings of **insects and frogs**.

The observation IDs used for crawling are listed in:

```
../additional_data/inaturalist_observation_ids.txt
```

---

# Dataset Preparation

All required datasets can be downloaded and prepared automatically by running:

```bash
python prepare_additional_data.py
```

The script performs the following steps:

1. Downloads the required datasets.
2. Extracts only the samples needed for training
3. Resamples all audio files to the target sample rate (32 kHz).

The processed datasets are stored in:

```
../additional_data
```

with the following directory structure (~33GB):

```
sa4birds/
│
├── additional_data/
│   ├── BirdVox-DCASE-20k/
│   │   └── wav/
│   │       ├── *.wav
│   │       └── ...
│   │
│   ├── ESC-50/
│   │   └── ESC-50-master/
│   │       └── audio/
│   │           ├── *.wav
│   │           └── ...
│   │
│   ├──  freefield1010/
│   │    └── wav/
│   │        ├── *.wav
│   │        └── ...
│   │ 
│   │
│   └── iNat/
│       └── *.mp3
│       └── ...
```

The full directory structure shown above requires approximately 33 GB of disk space.

---

# BirdSet Dataset

The **BirdSet benchmark dataset** is automatically downloaded during training or evaluation via the HuggingFace `datasets` library.

Example:

```python
import datasets

down_task = "HSN"
datasets.load_dataset("DBD-research-group/BirdSet", down_task)
```

---

# Training

After installing dependencies and preparing the datasets, training can be started from the project root with:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 train/train_birdset.py --config-name=base_finetune \
              train.dataset_name=HSN \
              train.n_epochs=5 \
              train.lr=6e-5 \
              train.lr_head_factor=1 \
              train.secondary_label_weight=0.9 \
              network.classifier=DSA \
              network.num_att_heads=1 \
              network.dropout_rate=0.4 \
              network.pretrain_checkpoint=checkpoints/XCL_PRETRAIN/XCL_eca_nfnet_l1_2025-09-02_195817/models/model.pth \
              augmentation.spec_aug.mix.p=0.5 \
              augmentation.spec_aug.masking.p=0.5 \
              augmentation.spec_aug.masking.timem=100 \
              augmentation.spec_aug.masking.freqm=50 \
              augmentation.wave_aug.no_call.p=1.0 \
              augmentation.wave_aug.no_call.num_per_batch=24 \
              augmentation.wave_aug.no_call.dirs="[$(pwd)/additional_data/BirdVox-DCASE-20k/wav,$(pwd)/additional_data/ESC-50/ESC-50-master/audio,$(pwd)/additional_data/freefield1010/wav,$(pwd)/additional_data/iNat]"\
              frontend.train_target_length=701 \
              frontend.val_target_length=701 \
              frontend.std=8.8109 \
              frontend.mean=-18.4622 \
              event_decoder.train.extracted_interval=7 \
              event_decoder.val.extracted_interval=7 \
              save_checkpoints=False \
              seed=1
```

The downstream task can be changed using the `train.dataset_name` parameter.

---

# Training Pipeline

The training script performs the following steps:

1. Downloads and loads the BirdSet downstream task via Hugging Face if it is not already available locally.
2. Loads the pretrained backbone checkpoint.
3. Trains the DSA using the **Dedicated Training (DT)** regime.
4. Evaluates the trained model on the test split of the selected downstream task.

---

# Notes

- Ensure that the additional datasets are successfully downloaded before starting training.
- Training may require substantial GPU memory depending on the configuration.
