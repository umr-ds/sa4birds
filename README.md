# Spectrogram Attention for Acoustic Bird Species Recognition

[![Python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)

This repository contains the official implementation of **Spectrogram Attention for Acoustic Bird Species Recognition**. The proposed approach introduces **Spectrogram Attention (SA)**, a novel mechanism for jointly modeling fine-grained spectro-temporal patterns in log-mel spectrograms using feature maps extracted from a pretrained convolutional neural network. The model is pretrained on a large-scale corpus of **9,735 bird species** from the Xeno-Canto dataset and subsequently fine-tuned on **eight BirdSet soundscape corpora** under three different training regimes.

<div align="center">
  <img src="figures/dsa_head.png" alt="Spectrogram Attention Head" width="400">
  <img src="figures/model_overview.png" alt="Model Overview" width="558">
</div>

---

## Checkpoints

Download the model checkpoints and place them in the `checkpoints` directory, organized by training regime (`DT`, `MT`, or `LT`).


| Training Regime |   Task    |                                Url                                |
|:---------------:|:---------:|:-----------------------------------------------------------------:|
|    Dedicated    |    HSN    | [Download](https://next.hessenbox.de/index.php/s/KR92DHDjYCSMREc) |
|    Dedicated    |    POW    | [Download](https://next.hessenbox.de/index.php/s/fYKk7FDG446jgxD) |
|    Dedicated    |    SNE    | [Download](https://next.hessenbox.de/index.php/s/7YRQ2NopSGmsxFX) |
|    Dedicated    |    PER    | [Download](https://next.hessenbox.de/index.php/s/dKjJgk3WEpFGpf4) |
|    Dedicated    |    NES    | [Download](https://next.hessenbox.de/index.php/s/GHMrTbregzZ66CE) |
|    Dedicated    |    UHH    | [Download](https://next.hessenbox.de/index.php/s/Jr3KWKMMJyF4Zgb) |
|    Dedicated    |    NBP    | [Download](https://next.hessenbox.de/index.php/s/qKmMDPyQSzRRzoo) |
|    Dedicated    |    SSW    | [Download](https://next.hessenbox.de/index.php/s/GPk5MdGsLikmHKa) |
|     Medium      | All tasks | [Download](https://next.hessenbox.de/index.php/s/ck8J8A95DdssSo4) |
|      Large      | All tasks | [Download](https://next.hessenbox.de/index.php/s/xxE5XTaNcHCXidy) |



## Validation

After installing the dependencies listed in `requirements.txt`, you can evaluate a model on a specific downstream task by first downloading the model checkpoints and placing them in the `checkpoints` director. Use the following to download all checkpoints: 

```bash
python prepare_checkpoints.py
```

You can then run the evaluation using `evaluate.py`. For example, to evaluate on **HSN** using the **DT** regime run:

```bash
python evaluate.py mode=DT downtask=HSN
```

To evaluate on all Birdset downtasks: 

```bash
python evaluate.py mode=DT downtask=ALL
```

## Inference 

For running on custom samples, refer to the `infer.ipynb`notebook for an example.