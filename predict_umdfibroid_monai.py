#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict NIfTI segmentations using MONAI models trained on **nnU-Net v2 preprocessed (.b2nd) data**,
but with **nnU-Net v2.6.2 prediction-time preprocessing + export** for maximal fidelity.

Key features
- Loads your trainer checkpoints:
    best_<model>_foldXX.pt  or  final_<model>_foldXX.pt
  and supports ensembling across folds (average logits, then export once).
- Uses nnU-Net v2.6.2 preprocessing/export code paths (DefaultPreprocessor + export_prediction_from_logits).
- Single-channel inference (expects imagesTs style *_0000.nii.gz, but also accepts *.nii.gz).
- Rebuilds models from checkpoint metadata first (`config`, `patch_size`, `spacing`) so prediction is
  training-faithful, especially for SwinUNETR.

Example (all folds):
python predict_umdfibroid_monai.py \
  --input-dir /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_raw_data/Dataset004_UMD/imagesTs \
  --output-dir /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_swinunetr \
  --plans /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_preprocessed/Dataset004_UMD/nnUNetPlans.json \
  --dataset-json /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_preprocessed/Dataset004_UMD/dataset.json \
  --configuration 3d_fullres \
  --model swinunetr \
  --runs-root /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UM_nnUnet/runs/umd_multiclass \
  --folds all \
  --prefer best \
  --task multiclass \
  --device cuda \
  --save-probabilities
"""

import os
import re
import gc
import json
import glob
import shutil
import argparse
import inspect
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, UNETR, DynUNet, SegResNet, SwinUNETR


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _has_param(cls, name: str) -> bool:
    try:
        return name in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


def _prune_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(callable_obj).parameters
        return {k: v for k, v in kwargs.items() if k in sig}
    except Exception:
        return kwargs


def _normalize_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("m."):
            k = k[len("m."):]
        out[k] = v
    return out


def _prune_sd_for_model(sd: dict, model: nn.Module) -> dict:
    mdl_sd = model.state_dict()
    keep = {}
    dropped = []
    for k, v in sd.items():
        if (k in mdl_sd) and (mdl_sd[k].shape == v.shape):
            keep[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"[load] Dropping {len(dropped)} unexpected/mismatched keys (showing up to 12):")
        for x in dropped[:12]:
            print("   -", x)
        if len(dropped) > 12:
            print(f"   ... ({len(dropped) - 12} more)")
    return keep


def _count_state_dict_mismatches(sd: dict, model: nn.Module) -> int:
    mdl_sd = model.state_dict()
    bad = 0
    for k, v in sd.items():
        if (k not in mdl_sd) or (mdl_sd[k].shape != v.shape):
            bad += 1
    return int(bad)


def _ckpt_tuple3(state: dict, key: str, fallback=None, cast=float):
    """
    Read a 3-vector from checkpoint metadata. Returns tuple(...) or fallback.
    """
    v = state.get(key, None)
    if v is None:
        return fallback
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(cast(x) for x in v)
    return fallback


def is_anisotropic_spacing(spacing_zyx: Tuple[float, float, float], ratio: float = 3.0) -> bool:
    z, y, x = [float(s) for s in spacing_zyx]
    return z > ratio * max(y, x)


def _ceil_to_multiple(x: int, m: int) -> int:
    return ((int(x) + int(m) - 1) // int(m)) * int(m)


def _ensure_tuple3(x) -> Tuple[int, int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    raise ValueError(f"Expected 3-tuple, got {x}")


# -----------------------------------------------------------------------------
# checkpoint discovery (multi-fold)
# -----------------------------------------------------------------------------
def discover_fold_checkpoints(runs_root: str, model: str, prefer: str = "best") -> List[str]:
    """
    Layout (your launcher):
      runs_root/<model>/fold_00/best_<model>_fold00.pt
      runs_root/<model>/fold_00/final_<model>_fold00.pt
    """
    runs_root = os.path.abspath(runs_root)
    model = model.lower()
    prefer = prefer.lower().strip()
    if prefer not in {"best", "final"}:
        raise ValueError("--prefer must be best or final")

    pat = os.path.join(runs_root, model, "fold_*", f"{prefer}_{model}_fold*.pt")
    cands = sorted(glob.glob(pat))

    if not cands:
        other = "final" if prefer == "best" else "best"
        pat2 = os.path.join(runs_root, model, "fold_*", f"{other}_{model}_fold*.pt")
        cands = sorted(glob.glob(pat2))

    fold_re = re.compile(r"fold_(\d+)")
    by_fold = {}
    for p in cands:
        m = fold_re.search(p)
        if not m:
            continue
        f = int(m.group(1))
        by_fold.setdefault(f, p)

    return [by_fold[k] for k in sorted(by_fold.keys())]


def parse_folds_arg(folds_str: str) -> Optional[List[int]]:
    s = (folds_str or "").strip().lower()
    if s in {"", "all"}:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def filter_ckpts_by_folds(ckpts: List[str], folds: Optional[List[int]]) -> List[str]:
    if folds is None:
        return ckpts
    want = set(int(x) for x in folds)
    fold_re = re.compile(r"fold_(\d+)")
    out = []
    for p in ckpts:
        m = fold_re.search(p)
        if not m:
            continue
        f = int(m.group(1))
        if f in want:
            out.append(p)
    return out


# -----------------------------------------------------------------------------
# model wrappers
# -----------------------------------------------------------------------------
class MainHeadPredictor(nn.Module):
    """
    Your training script returns either a tensor or a list/tuple (deep supervision).
    nnU-Net inference expects main head logits -> return y[0] if list.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        y = self.base(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def _swin_anisotropy_aware_params(
    roi_size_zyx: Tuple[int, int, int],
    spacing_zyx: Optional[Tuple[float, float, float]],
    win: int,
    patch_size_default: int = 2,
    anisotropic_patch: Tuple[int, int, int] = (1, 2, 2),
    anisotropic_win: Tuple[int, int, int] = (4, 7, 7),
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Returns (roi_fixed, patch_size, window_size) for SwinUNETR.

    - Ensure roi divisible by 32.
    - For anisotropy (thick-slice z), use smaller z window and z patch of 1.
    """
    rz, ry, rx = roi_size_zyx
    anis = False
    if spacing_zyx is not None:
        anis = is_anisotropic_spacing(spacing_zyx, ratio=3.0)
    if rz < min(ry, rx) // 2:
        anis = True

    if anis:
        ps = tuple(int(x) for x in anisotropic_patch)
        ws = tuple(int(x) for x in anisotropic_win)
    else:
        ps = (int(patch_size_default),) * 3
        ws = (int(win),) * 3

    def _fix_dim(dim: int, w: int) -> int:
        dim2 = _ceil_to_multiple(dim, 32)
        if w > 0:
            dim2 = _ceil_to_multiple(dim2, w)
            dim2 = _ceil_to_multiple(dim2, 32)
        return dim2

    roi_fixed = (
        _fix_dim(rz, ws[0]),
        _fix_dim(ry, ws[1]),
        _fix_dim(rx, ws[2]),
    )
    return roi_fixed, ps, ws


def build_model(
    name: str,
    in_channels: int,
    out_channels: int,
    roi_size_zyx: Tuple[int, int, int],
    ckpt_config: Optional[dict] = None,
    spacing_zyx: Optional[Tuple[float, float, float]] = None,
    swin_anisotropy_aware: bool = True,
) -> nn.Module:
    name = name.lower().strip()
    cfg = ckpt_config or {}

    if name == "unet3d":
        return UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            num_res_units=2, act="LEAKYRELU", norm="INSTANCE", dropout=0.0
        )

    if name == "unetr":
        return UNETR(
            in_channels=in_channels, out_channels=out_channels,
            img_size=tuple(roi_size_zyx), feature_size=16, hidden_size=768,
            mlp_dim=3072, num_heads=12, norm_name="instance",
            res_block=True, dropout_rate=0.0
        )

    if name == "dynunet":
        kernels = cfg.get("dynunet_kernels", None)
        strides = cfg.get("dynunet_strides", None)
        if kernels is None or strides is None:
            kernels = [[3, 3, 3]] * 5
            strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        up_k = strides[1:]
        base = dict(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            filters=[32, 64, 128, 256, 512],
        )
        if _has_param(DynUNet, "kernels"):
            base.update(kernels=kernels, strides=strides, upsample_kernel_sizes=up_k)
        else:
            base.update(kernel_size=kernels, strides=strides, upsample_kernel_size=up_k)

        maybe = {
            "norm_name": "instance",
            "deep_supervision": False,
            "res_block": True,
            "trans_bias": False,
        }
        base.update({k: v for k, v in maybe.items() if _has_param(DynUNet, k)})
        base = _prune_kwargs(DynUNet, base)
        return DynUNet(**base)

    if name == "segresnet":
        return SegResNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            init_filters=24, blocks_down=(1, 1, 2, 2), blocks_up=(1, 1, 1),
            norm="INSTANCE", dropout_prob=0.0
        )

    if name == "swinunetr":
        fs = int(cfg.get("swin_feature_size", 24))
        heads = tuple(cfg.get("swin_heads", [3, 6, 12, 24]))
        depths = tuple(cfg.get("swin_depths", [2, 2, 2, 1]))
        win = int(cfg.get("swin_window", 7))
        use_ckpt = bool(cfg.get("swin_use_checkpoint", True))

        if fs % 12 != 0:
            raise ValueError(f"swin_feature_size must be divisible by 12; got {fs}")

        roi_use = tuple(int(x) for x in roi_size_zyx)
        patch_size: Any = 2
        window_size: Any = win

        if swin_anisotropy_aware:
            roi_use, patch_size_t, window_size_t = _swin_anisotropy_aware_params(
                roi_size_zyx=roi_use, spacing_zyx=spacing_zyx, win=win
            )
            patch_size = patch_size_t
            window_size = window_size_t

        base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=heads,
            feature_size=fs,
            norm_name="instance",
            spatial_dims=3,
        )
        optional = {
            "window_size": window_size,
            "qkv_bias": True,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "normalize": True,
            "patch_norm": False,
            "use_checkpoint": use_ckpt,
            "downsample": "merging",
            "use_v2": False,
        }
        base.update({k: v for k, v in optional.items() if _has_param(SwinUNETR, k)})
        if _has_param(SwinUNETR, "img_size"):
            base["img_size"] = tuple(roi_use)
        base = _prune_kwargs(SwinUNETR, base)

        print(
            f"[swinunetr] roi={roi_use} patch_size={patch_size} "
            f"window_size={window_size} fs={fs} depths={depths} heads={heads}"
        )
        return SwinUNETR(**base)

    raise ValueError(f"Unknown model: {name}")


# -----------------------------------------------------------------------------
# nnU-Net v2.6.2 preprocessing/export bridge
# -----------------------------------------------------------------------------
def require_nnunetv2():
    try:
        import nnunetv2  # noqa: F401
        return True
    except Exception:
        return False


class NnUNetV2PreprocessorExporter:
    """
    Wrapper around nnU-Net v2.6.2 preprocessing + export.

    Contract:
      - preprocess_case([modality_files]) -> (data_czyx: np.ndarray float32, props: dict)
      - export_logits(logits_czyx, props, out_seg_path, save_probabilities=False, out_prob_path=None)
    """
    def __init__(self, plans_path: str, dataset_json: str, configuration: str):
        self.plans_path = os.path.abspath(plans_path)
        self.dataset_json_path = os.path.abspath(dataset_json)
        self.configuration = str(configuration)

        # Plans/dataset load (avoid nnunetv2.utilities.json_export; it changed across versions)
        with open(self.plans_path, "r") as f:
            self.plans = json.load(f)
        with open(self.dataset_json_path, "r") as f:
            self.dataset = json.load(f)

        self.file_ending = str(self.dataset.get("file_ending", ".nii.gz"))

        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        from nnunetv2.inference.export_prediction import export_prediction_from_logits

        self.PlansManager = PlansManager
        self.plans_manager = self.PlansManager(self.plans)
        self.cfg = self.plans_manager.get_configuration(self.configuration)
        self.preprocessor = DefaultPreprocessor(verbose=False)
        self._export_prediction_from_logits = export_prediction_from_logits

    def get_spacing_zyx(self) -> Tuple[float, float, float]:
        sp = getattr(self.cfg, "spacing", None)
        if sp is None:
            sp = self.plans.get("configurations", {}).get(self.configuration, {}).get("spacing", None)
        if sp is None:
            return (1.0, 1.0, 1.0)
        return (float(sp[0]), float(sp[1]), float(sp[2]))

    def get_patch_size_zyx(self) -> Tuple[int, int, int]:
        ps = getattr(self.cfg, "patch_size", None)
        if ps is None:
            ps = self.plans.get("configurations", {}).get(self.configuration, {}).get("patch_size", None)
        if ps is None:
            raise RuntimeError("Could not determine patch_size from plans/configuration.")
        return (int(ps[0]), int(ps[1]), int(ps[2]))

    def preprocess_case(self, files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        nnU-Net v2.6.2:
          DefaultPreprocessor.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        For test cases, seg_file=None.
        """
        out = self.preprocessor.run_case(
            image_files=files,
            seg_file=None,
            plans_manager=self.plans_manager,
            configuration_manager=self.cfg,
            dataset_json=self.dataset,  # pass dict to avoid internal load_json
        )
        # returns (data, seg, properties)
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise RuntimeError(
                f"Unexpected preprocess output from nnU-Net: type={type(out)}, "
                f"len={getattr(out, '__len__', None)}"
            )
        data, _seg, props = out
        data = np.asarray(data).astype(np.float32, copy=False)
        props = dict(props)

        # Ensure CZYX
        if data.ndim == 3:
            data = data[None]
        if data.ndim != 4:
            raise RuntimeError(f"Expected preprocessed data (C,Z,Y,X), got shape {data.shape}")
        return data, props

    def _truncate_output_path(self, out_seg_path: str) -> str:
        """
        nnU-Net export expects output_file_truncated (WITHOUT file ending).
        It will append dataset_json['file_ending'] internally.
        """
        out_seg_path = os.path.abspath(out_seg_path)
        if out_seg_path.endswith(self.file_ending):
            return out_seg_path[: -len(self.file_ending)]
        if out_seg_path.endswith(".nii.gz"):
            return out_seg_path[: -len(".nii.gz")]
        if out_seg_path.endswith(".nii"):
            return out_seg_path[: -len(".nii")]
        return out_seg_path

    def export_logits(
        self,
        logits_czyx: np.ndarray,
        props: dict,
        out_seg_path: str,
        save_probabilities: bool = False,
        out_prob_path: Optional[str] = None,
    ) -> None:
        """
        nnU-Net v2.6.2 export:
          export_prediction_from_logits(predicted_array_or_file, properties_dict,
                                       configuration_manager, plans_manager,
                                       dataset_json_dict_or_file, output_file_truncated,
                                       save_probabilities=False, ...)
        """
        logits_czyx = np.asarray(logits_czyx, dtype=np.float32)
        if logits_czyx.ndim != 4:
            raise RuntimeError(f"Expected logits (C,Z,Y,X), got {logits_czyx.shape}")

        out_trunc = self._truncate_output_path(out_seg_path)
        os.makedirs(os.path.dirname(out_trunc), exist_ok=True)

        self._export_prediction_from_logits(
            predicted_array_or_file=logits_czyx,
            properties_dict=props,
            configuration_manager=self.cfg,
            plans_manager=self.plans_manager,
            dataset_json_dict_or_file=self.dataset,
            output_file_truncated=out_trunc,
            save_probabilities=bool(save_probabilities),
        )

        if save_probabilities and out_prob_path:
            src_npz = out_trunc + ".npz"
            try:
                if os.path.isfile(src_npz):
                    os.makedirs(os.path.dirname(os.path.abspath(out_prob_path)), exist_ok=True)
                    shutil.copy2(src_npz, out_prob_path)
            except Exception as e:
                print(f"[warn] Could not copy probabilities npz to {out_prob_path}: {e}")


# -----------------------------------------------------------------------------
# prediction core
# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_ensemble_logits(
    models: List[nn.Module],
    x_bczyx: torch.Tensor,
    roi_size_zyx: Tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    gaussian: bool = True,
    amp: bool = False,
) -> torch.Tensor:
    mode = "gaussian" if gaussian else "constant"
    logit_sum = None
    for m in models:
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            y = sliding_window_inference(
                x_bczyx,
                roi_size=roi_size_zyx,
                sw_batch_size=int(sw_batch_size),
                predictor=m,
                overlap=float(overlap),
                mode=mode,
            )
        logit_sum = y if logit_sum is None else (logit_sum + y)
    return logit_sum / float(len(models))


def load_models_from_checkpoints(
    ckpts: List[str],
    model_name: str,
    in_channels: int,
    out_channels: int,
    roi_size_zyx_fallback: Tuple[int, int, int],
    spacing_zyx_fallback: Optional[Tuple[float, float, float]],
    device: torch.device,
    swin_anisotropy_aware: bool = True,
) -> Tuple[List[nn.Module], Tuple[int, int, int], Optional[Tuple[float, float, float]]]:
    """
    Training-faithful loading:
      - rebuild each model from checkpoint metadata first
      - use checkpoint patch_size as authoritative ROI when present
      - use checkpoint spacing as authoritative spacing when present
      - hard-fail if checkpoints disagree on inference ROI across folds
      - hard-fail if state_dict pruning would drop keys for SwinUNETR
    """
    models: List[nn.Module] = []
    roi_ref: Optional[Tuple[int, int, int]] = None
    spacing_ref: Optional[Tuple[float, float, float]] = None

    for p in ckpts:
        state = torch.load(p, map_location="cpu")
        cfg = state.get("config", {}) if isinstance(state, dict) else {}
        sd = state.get("model_state", state)
        sd = _normalize_state_dict_keys(sd)

        ckpt_roi = _ckpt_tuple3(state, "patch_size", fallback=None, cast=int)
        ckpt_spacing = _ckpt_tuple3(state, "spacing", fallback=None, cast=float)

        roi_use = tuple(int(x) for x in (ckpt_roi if ckpt_roi is not None else roi_size_zyx_fallback))
        spacing_use = ckpt_spacing if ckpt_spacing is not None else spacing_zyx_fallback

        if roi_ref is None:
            roi_ref = roi_use
        elif tuple(int(x) for x in roi_use) != tuple(int(x) for x in roi_ref):
            raise RuntimeError(
                "Checkpoint ROI mismatch across ensemble folds. "
                f"First ROI={roi_ref}, but {os.path.basename(p)} has ROI={roi_use}. "
                "All checkpoints in one ensemble must have the same training patch_size."
            )

        if spacing_ref is None:
            spacing_ref = spacing_use

        print(
            f"[load] {os.path.basename(p)} "
            f"roi_from_ckpt={ckpt_roi is not None} roi_use={roi_use} "
            f"spacing_from_ckpt={ckpt_spacing is not None} spacing_use={spacing_use}"
        )

        m = build_model(
            name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            roi_size_zyx=roi_use,
            ckpt_config=cfg,
            spacing_zyx=spacing_use,
            swin_anisotropy_aware=swin_anisotropy_aware,
        )

        n_bad = _count_state_dict_mismatches(sd, m)
        pruned = _prune_sd_for_model(sd, m)

        if model_name.lower().strip() == "swinunetr" and n_bad > 0:
            raise RuntimeError(
                f"{os.path.basename(p)} would drop {n_bad} mismatched state_dict key(s) for SwinUNETR. "
                "This indicates the prediction model was not rebuilt exactly like training."
            )

        missing, unexpected = m.load_state_dict(pruned, strict=False)

        if missing:
            print(f"[load] {os.path.basename(p)} missing keys: {len(missing)} (showing up to 8)")
            for k in missing[:8]:
                print("   -", k)
        if unexpected:
            print(f"[load] {os.path.basename(p)} unexpected keys: {len(unexpected)} (showing up to 8)")
            for k in unexpected[:8]:
                print("   -", k)

        if model_name.lower().strip() != "swinunetr" and n_bad > 0:
            print(
                f"[warn] {os.path.basename(p)} had {n_bad} dropped/mismatched state_dict key(s). "
                "For CNN-family models this is less likely to be catastrophic, but investigate if unexpected."
            )

        m = MainHeadPredictor(m).to(device).eval()
        models.append(m)

    if roi_ref is None:
        raise RuntimeError("No checkpoints were loaded.")

    return models, roi_ref, spacing_ref


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Predict with MONAI checkpoints using nnU-Net v2.6.2 preprocessing/export (multi-fold ensemble)."
    )

    ap.add_argument("--input-dir", type=str, required=True, help="Directory with raw NIfTI imagesTs (single-channel).")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory for segmentations.")
    ap.add_argument("--model", type=str, required=True, choices=["unet3d", "unetr", "dynunet", "segresnet", "swinunetr"])
    ap.add_argument("--task", type=str, default="multiclass", choices=["binary", "multiclass"])
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=None, help="For multiclass; if omitted, inferred from dataset.json labels.")
    ap.add_argument("--lesion-class-id", type=int, default=3, help="Only used for binary training (informational).")

    ap.add_argument("--plans", type=str, required=True, help="nnU-Net v2 nnUNetPlans.json")
    ap.add_argument("--dataset-json", type=str, required=True, help="nnU-Net dataset.json")
    ap.add_argument("--configuration", type=str, default="3d_fullres")

    ap.add_argument("--runs-root", type=str, default=None, help="Base runs dir like runs/umd_multiclass (contains <model>/fold_XX/...).")
    ap.add_argument("--folds", type=str, default="all", help="Comma-separated folds (e.g. 0,1,2,3,4) or 'all'. Used with --runs-root.")
    ap.add_argument("--prefer", type=str, default="best", choices=["best", "final"])
    ap.add_argument("--checkpoints", type=str, nargs="+", default=None, help="Explicit checkpoint list (overrides --runs-root).")

    ap.add_argument("--sw-batch-size", type=int, default=1)
    ap.add_argument("--sw-overlap", type=float, default=0.5)
    ap.add_argument("--no-gaussian", action="store_true", help="Use constant blending instead of gaussian.")
    ap.add_argument("--amp", action="store_true", help="Enable AMP for inference (off by default).")

    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-probabilities", action="store_true", help="Also save probability maps (nnU-Net export).")
    ap.add_argument(
        "--probs-suffix",
        type=str,
        default="_probs.npz",
        help="If --save-probabilities: copy nnU-Net's generated .npz to output_dir/<case><suffix> (best-effort).",
    )

    ap.add_argument(
        "--require-nnunetv2",
        action="store_true",
        default=True,
        help="If set (default), error if nnunetv2 is not importable.",
    )
    ap.add_argument(
        "--swin-anisotropy-aware",
        action="store_true",
        default=True,
        help="If set (default), adjust Swin patch/window/roi to respect anisotropy + /32 constraints.",
    )

    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    if not require_nnunetv2():
        raise RuntimeError(
            "nnunetv2 is not importable, but this script is designed to use nnU-Net v2.6.2 preprocessing/export.\n"
            "Install:  pip install nnunetv2\n"
            "Then re-run."
        )

    # Load dataset.json to infer num classes if needed
    with open(args.dataset_json, "r") as f:
        dset = json.load(f)

    def infer_num_classes_from_dataset_json(d) -> int:
        labels = d.get("labels", {})
        if isinstance(labels, dict) and len(labels) > 0:
            ids = []
            for _, v in labels.items():
                try:
                    ids.append(int(v))
                except Exception:
                    pass
            if ids:
                return int(max(ids)) + 1
        raise ValueError("Could not infer num_classes from dataset.json labels.")

    if args.task == "binary":
        out_channels = 2
    else:
        if args.num_classes is None:
            args.num_classes = infer_num_classes_from_dataset_json(dset)
            print(f"[dataset.json] inferred num_classes={args.num_classes}")
        out_channels = int(args.num_classes)

    # Preproc/export manager (nnU-Net v2.6.2)
    nnx = NnUNetV2PreprocessorExporter(
        plans_path=args.plans,
        dataset_json=args.dataset_json,
        configuration=args.configuration,
    )
    spacing_zyx = nnx.get_spacing_zyx()
    patch_size_zyx = nnx.get_patch_size_zyx()
    print(f"[nnunetv2] configuration={args.configuration} spacing_zyx={spacing_zyx} patch_size_zyx={patch_size_zyx}")

    # Determine checkpoints (multi-fold)
    if args.checkpoints:
        ckpts = list(args.checkpoints)
    elif args.runs_root:
        ckpts = discover_fold_checkpoints(args.runs_root, args.model, prefer=args.prefer)
        folds = parse_folds_arg(args.folds)
        ckpts = filter_ckpts_by_folds(ckpts, folds)
    else:
        raise ValueError("Provide --checkpoints or --runs-root")

    if not ckpts:
        raise RuntimeError("No checkpoints found after discovery/filtering.")

    print("[predict] Using checkpoints:")
    for p in ckpts:
        print(" -", p)

    # Build ensemble models from checkpoint metadata first.
    # Fallback to plans only when old checkpoints do not contain patch_size / spacing.
    roi_size_zyx_fallback = tuple(int(x) for x in patch_size_zyx)
    models, roi_size_zyx, spacing_used = load_models_from_checkpoints(
        ckpts=ckpts,
        model_name=args.model,
        in_channels=int(args.in_channels),
        out_channels=int(out_channels),
        roi_size_zyx_fallback=roi_size_zyx_fallback,
        spacing_zyx_fallback=spacing_zyx,
        device=device,
        swin_anisotropy_aware=bool(args.swin_anisotropy_aware),
    )
    print(f"[predict] ensemble inference roi_size_zyx={roi_size_zyx} spacing_used={spacing_used}")

    # Collect input files
    in_dir = os.path.abspath(args.input_dir)
    cands = sorted(glob.glob(os.path.join(in_dir, "*_0000.nii.gz")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(in_dir, "*.nii.gz")))
    if not cands:
        raise RuntimeError(f"No NIfTI files found in {in_dir}")

    print(f"[predict] Found {len(cands)} case file(s)")

    # Predict each case
    for i, img_path in enumerate(cands, 1):
        try:
            base = os.path.basename(img_path)
            case_id = base.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            out_seg = os.path.join(args.output_dir, f"{case_id}{nnx.file_ending}")
            out_prob = os.path.join(args.output_dir, f"{case_id}{args.probs_suffix}")

            # nnU-Net preprocess expects list of modalities. Single channel => [img_path]
            data_czyx, props = nnx.preprocess_case([img_path])

            x = torch.from_numpy(data_czyx[None]).to(device=device, dtype=torch.float32, non_blocking=False)

            logits = infer_ensemble_logits(
                models=models,
                x_bczyx=x,
                roi_size_zyx=tuple(int(v) for v in roi_size_zyx),
                sw_batch_size=int(args.sw_batch_size),
                overlap=float(args.sw_overlap),
                gaussian=(not args.no_gaussian),
                amp=bool(args.amp),
            )
            logits_czyx = logits[0].detach().cpu().numpy().astype(np.float32)

            nnx.export_logits(
                logits_czyx=logits_czyx,
                props=props,
                out_seg_path=out_seg,
                save_probabilities=bool(args.save_probabilities),
                out_prob_path=(out_prob if args.save_probabilities else None),
            )

            print(f"[{i:4d}/{len(cands)}] wrote {os.path.basename(out_seg)}")

        except Exception as e:
            print(f"[{i:4d}/{len(cands)}] ERROR on {img_path}: {type(e).__name__}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()