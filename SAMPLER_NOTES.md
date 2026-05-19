# Sampler Notes

This repository includes two sampler modes for training:

- `--sampler nnunet`
- `--sampler native`

## Recommended default

The default is:

```bash
--sampler nnunet
```

This path uses the included file:

```python
vendored_nnunet_sampler.py
```

That sampler is a self-contained, nnU-Net-v2-style patch sampler that does **not** import `nnunetv2`, but is designed to reproduce the key practical behaviors of nnU-Net foreground patch sampling on preprocessed `.b2nd` data.

## What the vendored sampler does

It implements:

- deterministic per-batch foreground oversampling
- `any` or `perclass` foreground center selection
- optional inverse-frequency class weighting to favor rarer labels
- an optional minimum-foreground-in-patch quality guard with re-sampling
- optional case reuse within batch to reduce repeated loading overhead
- lightweight debugging outputs for foreground sampling behavior

## When to use `native`

Use:

```bash
--sampler native
```

if you want to compare against the simpler built-in patch sampler in `train_umdfibroid_monai.py`.

## Practical guidance

For most experiments in this repository, the intended public-facing default is:

```bash
--sampler nnunet
```

because it is closer to the patch-sampling behavior the training pipeline is trying to emulate.
