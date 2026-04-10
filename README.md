# Uterine MRI Segmentation Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#installation)
[![MONAI](https://img.shields.io/badge/MONAI-3D%20Medical%20Imaging-7b61ff.svg)](#installation)
[![nnU--Net](https://img.shields.io/badge/nnU--Net-v2.6.2-00a36c.svg)](#workflow-overview)
[![License](https://img.shields.io/badge/License-Add%20your%20license-lightgrey.svg)](#license)

A reproducible training, inference, and benchmarking pipeline for **3D uterine MRI segmentation** using **MONAI** models trained on **nnU-Net-preprocessed data**.

This repository is intended for projects such as uterine anatomy, myoma/fibroid, endometrial cancer, and uterine leiomyosarcoma segmentation, with support for both **binary** and **multiclass** experiments.

## Highlights

- Train MONAI models on **nnU-Net-preprocessed `.b2nd` data**
- Supports `UNet3D`, `UNETR`, `DynUNet`, `SegResNet`, and `SwinUNETR`
- Multi-GPU **k-fold launcher** for large experiments
- **Training-faithful inference** with nnU-Net v2.6.2 preprocessing/export
- Multi-dataset validator with Dice, Jaccard, HD95, ASSD, volume error, pairwise bootstrap comparisons, and figure generation
- Handles both legacy dataset layouts and nnU-Net raw dataset layouts

## Workflow overview

This repository assumes the following workflow:

### Step 1: Prepare your dataset in nnU-Net format
Organize your data as an nnU-Net raw dataset, for example:

```text
DatasetXXX_YourTask/
├── dataset.json
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── labelsTs/   # optional, for held-out evaluation
```

### Step 2: Run nnU-Net preprocessing first
**This repository does not replace nnU-Net preprocessing.** You must install nnU-Net v2 and run preprocessing on your dataset before using the MONAI training scripts here.

At minimum, you need nnU-Net to generate:

- `nnUNetPlans.json`
- `dataset.json`
- `splits_final.json` or your own fold split file
- the preprocessed configuration directory such as `nnUNetPlans_3d_fullres/`
- preprocessed `.b2nd` case files

In other words, the MONAI trainer in this repository is designed to consume **nnU-Net-preprocessed outputs**, not raw images directly.

### Step 3: Train MONAI models
Use the included trainer or fold launcher on the preprocessed `.b2nd` data.

### Step 4: Predict on test data
Run inference from saved checkpoints. Prediction uses nnU-Net v2.6.2 preprocessing/export code paths for fidelity with the original plans.

### Step 5: Benchmark models
Compare model outputs across datasets, labels, and architectures using the validation script.

## Repository structure

```text
.
├── README.md
├── train_umdfibroid_monai.py        # main MONAI trainer on nnU-Net-preprocessed data
├── train_folds_monai.py             # multi-GPU k-fold launcher
├── predict_umdfibroid_monai.py      # inference + nnU-Net v2.6.2 export
├── validate_benchmark.py            # multi-dataset evaluation and figure/table generation
└── vendored_nnunet_sampler.py       # optional vendored sampler if used by the trainer
```

## Models

Supported MONAI architectures:

- `unet3d`
- `unetr`
- `dynunet`
- `segresnet`
- `swinunetr`

The validation script can also compare predictions from external `nnU-Net` runs alongside MONAI models.

## Installation

Create an environment with the core dependencies:

```bash
pip install numpy torch monai SimpleITK matplotlib batchgenerators blosc2
```

For prediction, you also need **nnU-Net v2** available in the environment:

```bash
pip install nnunetv2
```

## Quickstart

### 1. Run nnU-Net preprocessing
This is required before training with these scripts.

Example high-level sequence:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Then confirm you have:

- `nnUNetPlans.json`
- `dataset.json`
- preprocessed data under a configuration directory such as `nnUNetPlans_3d_fullres`
- a split file such as `splits_final.json`

### 2. Launch k-fold training

```bash
python train_folds_monai.py \
  --model swinunetr \
  --task multiclass \
  --gpus 0,1 \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --outdir-base runs/uterine_multiclass
```

### 3. Run direct single-fold training

```bash
python train_umdfibroid_monai.py \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --fold 0 \
  --outdir runs/uterine_multiclass/swinunetr/fold_00 \
  --model swinunetr \
  --task multiclass
```

### 4. Predict from checkpoints

```bash
python predict_umdfibroid_monai.py \
  --input-dir /path/to/imagesTs \
  --output-dir /path/to/outputs_swinunetr \
  --plans /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --configuration 3d_fullres \
  --model swinunetr \
  --runs-root runs/uterine_multiclass \
  --folds all \
  --prefer best \
  --task multiclass \
  --device cuda
```

### 5. Benchmark models
Edit the `CONFIG` block in `validate_benchmark.py`, then run:

```bash
python validate_benchmark.py
```

## Results

The benchmarking script generates per-dataset and compiled outputs that are useful for papers, internal model comparison, and QC.

### Quantitative outputs

For each dataset, the validator writes:

- `per_subject_detailed.csv`
- `summary_per_label.csv`
- `summary_overall.csv`
- `pairwise_by_label.csv`
- `pairwise_overall.csv`
- `manifest.json`

Across all datasets, it also writes compiled CSVs such as:

- `all_per_subject_detailed.csv`
- `all_summary_per_label.csv`
- `all_summary_overall.csv`
- `all_pairwise_by_label.csv`
- `all_pairwise_overall.csv`

### Metrics

Included metrics:

- Dice
- Jaccard
- HD95
- ASSD
- ground-truth volume
- predicted volume
- absolute volume difference
- relative volume difference

### Visual outputs

The validator can generate:

- violin plots for Dice
- boxplots for HD95
- Bland–Altman plots for volumes
- case overlays for qualitative review
- LaTeX tables for manuscript integration

## Script summary

### `train_umdfibroid_monai.py`
Main trainer for MONAI segmentation models on nnU-Net-preprocessed `.b2nd` data.

### `train_folds_monai.py`
Multi-GPU launcher for fold-based training.

### `predict_umdfibroid_monai.py`
Inference script using nnU-Net v2.6.2 preprocessing and export.

### `validate_benchmark.py`
Multi-dataset validation and reporting script.

## Public release notes

This public-facing version should ideally:

- replace local hard-coded paths with your own environment-specific paths
- include a real `requirements.txt` or `environment.yml`
- include a project license
- document dataset label conventions in more detail
- describe how `vendored_nnunet_sampler.py` is obtained if it is required

## Citation

If you use this repository in academic work, cite MONAI, nnU-Net, and your project-specific paper or benchmark manuscript.

## License

Add your chosen license here, for example MIT, BSD-3-Clause, or Apache-2.0.