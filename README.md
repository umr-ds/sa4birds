# Spectrogram Attention for Acoustic Bird Species Recognition

[![Python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

This repository contains the official implementation of **Spectrogram Attention for Acoustic Bird Species Recognition**. The proposed Deep learning approach introduces **Spectrogram Attention (SA)**, a novel mechanism for jointly modeling fine-grained spectro-temporal patterns in log-mel spectrograms using feature maps extracted from a pretrained convolutional neural network. The model is pretrained on a large-scale corpus of **9,735 bird species** from the Xeno-Canto dataset and subsequently fine-tuned on **eight BirdSet soundscape corpora** under three different training regimes.

<div align="center">
  <img src="figures/dsa_head.png" alt="Spectrogram Attention Head" width="400">
  <img src="figures/model_overview.png" alt="Model Overview" width="558">
</div>

---

# Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Checkpoints](#checkpoints)
- [Model Demo](#model-demo)
- [Validation](#validation)
- [Citation](#citation)


---

# Project Structure

```
sa4birds/
│
├── additional_data/          # Auxiliary data used by the project
├── ckpts/              # Pretrained model checkpoints
├── figures/                  # Architecture and documentation figures
│
├── models/                   # Model architectures and implementations
├── train/                    # Training scripts and utilities
│
├── notebooks/                # Example notebooks
│   ├── model_demo.ipynb      # Demonstrates model usage
│   └── evaluation_birdset.ipynb # Demonstrates evaluation on all Birdset down tasks 
│   └── evaluation_ablation_study.ipynb # Demonstrates ablation study experiments
│   └── evaluation_beans.ipynb # Demonstrates transfer learning experiments on BEANS benchmark

│
├── validate_birdset.py       # Evaluation entry point
├── prepare_checkpoints.py    # Script to download / prepare checkpoints for testing
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```


**Note:**  
For all notebooks provided in the `notebooks/` directory, we also uploaded the **cell outputs corresponding to the expected results**. This allows users to inspect the expected outputs without rerunning the full experiments, which can require significant computational resources and large datasets.



## Requirements

This project requires **Python 3+** and a **CUDA-capable GPU with >12 GB** of VRAM when evaluating on BirdSet, due to the size of the trained models. Running a single model on a single sample on a CPU is possible (see [Model demo](#model-demo)), but GPU execution is strongly recommended for evaluation or repeated inference to avoid extremely long runtimes.
### System Requirements
- **Python:** > 3.0 (recommended: Python 3.9+)
- **GPU (recommended):** NVIDIA GPU with CUDA support
- **NVIDIA Drivers:** Required for CUDA support

### Install Python Dependencies
After cloning the repository, navigate into the project directory:

```bash
git clone git@github.com:umr-ds/sa4birds.git
cd sa4birds
```

The project relies on the following main packages (see `requirements.txt` for the complete list):
```text
datasets
hydra-core
librosa
numpy
scikit-learn
soundfile
timm
torch
torchaudio
torchmetrics
torchvision
transformers
```
Create a Python virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate 
```
In the case of a Windows system, run the following command in PowerShell:

```powershell
python3 -m venv venv
./venv/scripts/Activate.ps1
```

Install the required Python packages listed in **`requirements.txt`**:

```bash
pip install -r requirements.txt
```

### Memory requirements

Evaluation on BirdSet (see [Datasets](#datasets)) across all downstream tasks requires approximately **160 GB of disk space** to download and store the datasets. In addition, about **8.5 GB** of storage is needed for the trained model checkpoints across all regimes.

Evaluation on BEANS (see [Datasets](#datasets)) across all downstream tasks requires approximately **320 GB of disk space** to download and store the datasets. 


<!--
| Regime  | Space (GB) |
|---------|------------|
| **HSN** | 6.7        |
| **POW** | 17         |
| **SNE** | 23         |
| **PER** | 12         |
| **NES** | 16         |
| **UHH** | 6.6        |
| **UHH** | 31        |
| **SSW** | 45         |
-->

## Datasets

### Birdset
Training and evaluation primarily rely on the **[BirdSet](https://github.com/DBD-research-group/BirdSet)** benchmark.

BirdSet contains eight downstream tasks, each consisting of:

- **Training data:** weakly labeled recordings from **Xeno-Canto**
- **Test data:** strongly annotated **regional soundscapes**

For details see the **[BirdSet paper](https://arxiv.org/abs/2403.10380)**.

Datasets are automatically downloaded via the HuggingFace `datasets` library.

Example:

```python
import datasets 

down_task = "HSN"
datasets.load_dataset("DBD-research-group/BirdSet", down_task)
```

Cached datasets are stored in:

```
~/.cache/huggingface/
```
### Additional datasets:

The following datasets are used as no-call samples during training: 

- [Freefield1010](https://dcase.community/challenge2018/task-bird-audio-detection)
- [ESC-50](https://github.com/karolpiczak/ESC-50) 
- [BirdVox-DCASE-20k](https://dcase.community/challenge2018/task-bird-audio-detection) 

In addition, we used a subset of insect and frog sounds collected from [iNaturalist](https://www.inaturalist.org/) as no-call samples. For more details see the [training guide](train/README.md) 

### BEANS

For transfer learning experiments, we use the **BEANS benchmark**, which consists of different downstream tasks related to various animal sounds, such as bats. For more details, see [BEANS](https://github.com/earthspecies/beans).
## Checkpoints

Our pretrained checkpoints for BirdSet are available for three training regimes:

| Regime | Description                               |
|--------|-------------------------------------------|
| **DT** | Dedicated training (task-specific models) |
| **MT** | Medium training                           |
| **LT** | Large  training                           |

Download the model checkpoints and place them in the `ckpts` directory, organized by training regime (`DT`, `MT`, or `LT`).


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


Download checkpoints manually or run:

```bash
python prepare_checkpoints.py
```

This will download the main checkpoints trained on BirdSet with the following structure:

```
sa4birds/
│
├── ckpts/                         # pretrained model checkpoints
│   ├── DT/
│   │   └── HSN/                         # downstream task name
│   │       ├── HSN_eca_nfnet_l1_2025-10-20_112131/   # DT HSN first model checkpoint
│   │       └── ...
│   │
│   ├── MT/
│   │   ├── MT_eca_nfnet_l1_2025-11-25_151907/        # MT first model checkpoint
│   │   └── ...
│   │
│   └── LT/
│       ├── LT_eca_nfnet_l1_2025-11-24_180849/        # LT first model checkpoint
│       └── ...
```


## Model Demo
A demonstration of how to run one of the trained BirdSet models is provided in the notebook:

```
model_demo.ipynb
```

This notebook shows how to:

- load the trained model
- run model 
- inspect the outputs

To run the demo notebook, install [Jupyter](https://jupyter.org/):

```bash
pip install jupyterlab
```

### Launching Jupyter

Start the Jupyter notebook server from the project directory:

```bash
jupyter lab
```

Your browser will open automatically. Then open:

```
notebooks/model_demo.ipynb
```

and run the cells to see the model in action.

---

## Validation

### Birdset:

After installing the dependencies listed in `requirements.txt` and downloading the checkpoints (see [Checkpoints](#checkpoints)), you can run the evaluation on BirdSet using `validate_birdset.py`.

For example, to evaluate **HSN** using the **DT** regime, run:
```bash
python validate_birdset.py mode=DT downtask=HSN
```

To evaluate on all Birdset downtasks: 

```bash
python validate_birdset.py mode=DT downtask=ALL
```

A demonstration of the evaluation on BirdSet is provided in the notebook:

```
notebooks/evaluation_birdset.ipynb
```
The ablation study experiments are provided the notebook:

```
notebooks/evaluation_ablation_study.ipynb
```
### BEANS:

To rerun all tests for our trained models on the BEANS benchmark, please use the following notebook:
```
notebooks/evaluation_beans.ipynb
```