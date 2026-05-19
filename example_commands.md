# Example Commands

## Train a single fold directly

```bash
python train_umdfibroid_monai.py \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --nnunet-config 3d_fullres \
  --fold 0 \
  --outdir runs/umd_multiclass/swinunetr/fold_00 \
  --model swinunetr \
  --task multiclass \
  --sampler nnunet
```

## Launch folds across multiple GPUs

```bash
python train_folds_monai.py \
  --model unet3d \
  --task multiclass \
  --gpus 0,1 \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --outdir-base runs/umd_multiclass \
  --sampler nnunet
```

## Binary training

```bash
python train_folds_monai.py \
  --model segresnet \
  --task binary \
  --lesion-class-id 3 \
  --gpus 0 \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --outdir-base runs/umd_binary
```

## Inference from all folds

```bash
python predict_umdfibroid_monai.py \
  --input-dir /path/to/imagesTs \
  --output-dir /path/to/outputs_swinunetr \
  --plans /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --configuration 3d_fullres \
  --model swinunetr \
  --runs-root runs/umd_multiclass \
  --folds all \
  --prefer best \
  --task multiclass \
  --device cuda
```

## Inference from explicit checkpoints

```bash
python predict_umdfibroid_monai.py \
  --input-dir /path/to/imagesTs \
  --output-dir /path/to/outputs_unet3d \
  --plans /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --configuration 3d_fullres \
  --model unet3d \
  --checkpoints \
    runs/umd_multiclass/unet3d/fold_00/best_unet3d_fold00.pt \
    runs/umd_multiclass/unet3d/fold_01/best_unet3d_fold01.pt \
  --task multiclass
```


## Train MedNeXt across folds

```bash
python train_folds_monai.py \
  --model mednext \
  --task multiclass \
  --gpus 0,1 \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --outdir-base runs/umd_multiclass \
  --sampler nnunet \
  --mednext-variant S \
  --mednext-kernel-size 3
```

## MedNeXt inference from all folds

```bash
python predict_umdfibroid_monai.py \
  --input-dir /path/to/imagesTs \
  --output-dir /path/to/outputs_mednext \
  --plans /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --configuration 3d_fullres \
  --model mednext \
  --runs-root runs/umd_multiclass \
  --folds all \
  --prefer best \
  --task multiclass \
  --device cuda
```

## Run benchmark validation

Edit the configuration template in `validate_benchmark.py`, then run:

```bash
python validate_benchmark.py
```

## Use native sampler if vendored sampler is absent

```bash
python train_folds_monai.py \
  --model unet3d \
  --sampler native \
  --task multiclass \
  --gpus 0 \
  --plans-json /path/to/nnUNetPlans.json \
  --dataset-json /path/to/dataset.json \
  --preproc-dir /path/to/nnUNetPlans_3d_fullres \
  --splits-json /path/to/splits_final.json \
  --outdir-base runs/umd_multiclass
```
