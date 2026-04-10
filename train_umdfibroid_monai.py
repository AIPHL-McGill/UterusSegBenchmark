#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MONAI 3D segmentation models with an nnU-Net-*style* pipeline, **without importing nnunetv2**.

Key fixes to recover Dice when using nnU-Net preprocessed .b2nd:
- Preserve ignore voxels (-1) THROUGH cropping/padding and spatial transforms:
  * pad_value_seg = ignore_index
  * border_cval_seg = ignore_index
- Implement TRUE ignore-masked soft Dice (ignored voxels do not contribute).
- Default Dice excludes background (nnU-Net typical for multiclass Dice), CE still includes it.

Compatibility fixes (launcher):
- Re-introduce --val-mode {patch,fullcase} and --val-max-batches so older launchers don't crash.

SPEED fixes (GPU util low / I/O bound):
- Prevent CPU thread oversubscription in augmenter workers (OMP/MKL/OPENBLAS/NUMEXPR = 1).
- Add per-process LRU cache for loaded cases (data+seg) to avoid re-reading/decompressing .b2nd for every patch.
- Add per-process cache for foreground coords (FG sampling) keyed on cid.
- Optional: avoid loading the same cid multiple times within a batch by reusing cached arrays.

SwinUNETR anisotropy-aware updates (nnU-Net-ish):
- Auto-detect anisotropy from nnU-Net plans spacing.
- For anisotropic spacing: prefer patch_size=(1,2,2) and window_size=(wz,wy,wx) with small wz (e.g. 4),
  when the installed MONAI SwinUNETR supports tuple-valued patch_size/window_size.
- Always pass img_size=roi_size when supported.
- You can force int-valued params if your MONAI build rejects tuples (--swin-force-int),
  and/or override patch/window explicitly (--swin-patch-size / --swin-window-size).

Tested for MONAI 1.4.0 API compatibility (imports/signatures used here are stable in 1.4.0).
"""

import os

# -----------------------------------------------------------------------------
# Critical performance env guards (must be set before numpy/torch imports)
# -----------------------------------------------------------------------------
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCH_COMPILE", "0")

# Avoid BLAS/OMP thread explosion inside MultiThreadedAugmenter processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import inspect
import warnings
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, UNETR, DynUNet, SegResNet, SwinUNETR
from monai.utils import set_determinism

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform

try:
    from batchgenerators.transforms.compose import Compose  # type: ignore
except Exception:  # pragma: no cover
    try:
        from batchgenerators.transforms.abstract_transforms import Compose  # type: ignore
    except Exception:  # pragma: no cover
        class Compose(AbstractTransform):
            def __init__(self, transforms):
                self.transforms = list(transforms)

            def __call__(self, **data_dict):
                for t in self.transforms:
                    data_dict = t(**data_dict)
                return data_dict

from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

warnings.filterwarnings("ignore", message=r"the prediction of class .* is all 0")
warnings.filterwarnings("ignore", message=r"the ground truth of class .* is all 0")

# Ensure logs appear immediately even when stdout is piped via a launcher
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# -----------------------------------------------------------------------------
# Foreground oversampling probability quantization helper (nnU-Net-ish)
# -----------------------------------------------------------------------------
def get_effective_fg_prob(batch_size: int, fg_prob: float) -> float:
    """nnU-Net-like quantization: with batch_size==2 force 50% foreground sampling (1/2)."""
    try:
        bs = int(batch_size)
    except Exception:
        bs = batch_size
    if bs == 2:
        return 0.5
    return float(fg_prob)


# -----------------------------------------------------------------------------
# Plans / splits / dataset.json helpers
# -----------------------------------------------------------------------------
def load_splits(splits_json: str, fold: int) -> Tuple[List[str], List[str]]:
    with open(splits_json, "r") as f:
        splits = json.load(f)
    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError(f"splits_final.json invalid: expected list, got {type(splits)}")
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold={fold} out of range for {len(splits)} folds")
    tr = splits[fold].get("train", [])
    va = splits[fold].get("val", [])
    if not tr or not va:
        raise ValueError(f"fold {fold} has empty train/val in splits_final.json")
    return list(tr), list(va)


def load_plans(plans_json: str, configuration: str = "3d_fullres") -> Dict:
    with open(plans_json, "r") as f:
        plans = json.load(f)
    cfg_key = configuration
    cfgs = plans.get("configurations", {})
    if cfg_key not in cfgs:
        cfg_key = plans.get("default_configuration", cfg_key)
    if cfg_key not in cfgs:
        raise KeyError(f"Could not find configuration '{configuration}' (or default) in plans.")
    return {"cfg_key": cfg_key, "cfg": cfgs[cfg_key], "plans": plans}


def derive_nnunet_patch_and_bs(plans_cfg: Dict) -> Tuple[Tuple[int, int, int], int]:
    cfg = plans_cfg["cfg"]
    patch = tuple(int(x) for x in cfg["patch_size"])
    bs = int(cfg["batch_size"])
    return patch, bs


def load_dataset_json(dataset_json: str) -> Dict:
    with open(dataset_json, "r") as f:
        d = json.load(f)
    return d


def infer_num_classes_from_dataset_json(dataset_json: str) -> int:
    d = load_dataset_json(dataset_json)
    labels = d.get("labels", {})
    if isinstance(labels, dict) and len(labels) > 0:
        ids: List[int] = []
        for _, v in labels.items():
            try:
                ids.append(int(v))
            except Exception:
                pass
        if ids:
            return int(max(ids)) + 1
    raise ValueError(f"Could not infer num_classes from {dataset_json} (labels missing/unexpected).")


# -----------------------------------------------------------------------------
# Small helpers
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


def nnunet_deep_supervision_weights(n_outputs: int) -> np.ndarray:
    w = np.asarray([1.0 / (2 ** i) for i in range(n_outputs)], dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    return w


def is_anisotropic_spacing(spacing_zyx: Tuple[float, float, float], ratio: float = 3.0) -> bool:
    z, y, x = [float(s) for s in spacing_zyx]
    return z > ratio * max(y, x)


# -----------------------------------------------------------------------------
# SwinUNETR anisotropy-aware helpers (nnU-Net-ish)
# -----------------------------------------------------------------------------
def _as_tuple3(x):
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    return (int(x), int(x), int(x))


def ceil_to_multiple(v: int, m: int) -> int:
    v = int(v)
    m = int(m)
    return ((v + m - 1) // m) * m


def ceil_tuple3_to_multiple(t3: Tuple[int, int, int], m: int) -> Tuple[int, int, int]:
    return tuple(ceil_to_multiple(int(v), m) for v in t3)  # type: ignore



def choose_swin_params_from_plans(
    roi_size_zyx: Tuple[int, int, int],
    spacing_zyx: Tuple[float, float, float],
    base_window: int = 7,
    aniso_ratio: float = 3.0,
    aniso_wz: int = 4,
    force_int: bool = False,
) -> Dict[str, object]:
    """
    nnU-Net-like anisotropy adaptation for SwinUNETR:
      - If anisotropic (thick slices): avoid early Z downsampling & reduce attention window in Z.
      - Else: standard (2,2,2) patch embedding and cubic-ish window.
    Returns values suitable for SwinUNETR kwargs (patch_size, window_size) when available.
    """
    rz, ry, rx = [int(v) for v in roi_size_zyx]
    zsp, ysp, xsp = [float(v) for v in spacing_zyx]
    anisotropic = (zsp > float(aniso_ratio) * max(ysp, xsp))

    patch_size = (2, 2, 2)
    window_size = (int(base_window), int(base_window), int(base_window))

    if anisotropic:
        patch_size = (1, 2, 2)
        wz = max(1, min(int(aniso_wz), int(rz)))
        window_size = (int(wz), int(base_window), int(base_window))

    if force_int:
        # If your MONAI build only accepts int, fall back to scalar values.
        # (Patch embedding cannot be truly anisotropic in this fallback mode.)
        patch_size_out = 2
        window_size_out = int(window_size[0]) if anisotropic else int(base_window)
    else:
        patch_size_out = tuple(patch_size)
        window_size_out = tuple(window_size)

    return {
        "anisotropic": anisotropic,
        "patch_size": patch_size_out,
        "window_size": window_size_out,
    }


# -----------------------------------------------------------------------------
# Deterministic helpers (for reproducible coord subsampling)
# -----------------------------------------------------------------------------
import zlib

def _stable_seed_from_cid(cid: str, extra: int = 0) -> int:
    """Stable (cross-process) 31-bit seed derived from a case id."""
    s = zlib.adler32(cid.encode("utf-8"))
    return int((s + int(extra)) & 0x7fffffff)


# -----------------------------------------------------------------------------
# Ignore-label handling utilities
# -----------------------------------------------------------------------------
def _map_ignore_labels_to_index(
    seg: np.ndarray,
    ignore_labels: Tuple[int, ...],
    ignore_index: int = -1
) -> np.ndarray:
    seg = np.asarray(seg).astype(np.int16, copy=False)
    for ig in ignore_labels:
        seg[seg == int(ig)] = int(ignore_index)
    return seg


def _clamp_valid_labels(seg: np.ndarray, num_classes: int, ignore_index: int = -1) -> np.ndarray:
    seg = np.asarray(seg).astype(np.int16, copy=False)
    m = (seg != int(ignore_index))
    if np.any(m):
        seg[m] = np.clip(seg[m], 0, int(num_classes) - 1).astype(np.int16, copy=False)
    return seg


# -----------------------------------------------------------------------------
# A) Dataset reader (nnU-Net v2 Blosc2 .b2nd) with per-process LRU cache
# -----------------------------------------------------------------------------
class NnUNetPreprocessedDataset:
    """
    SPEED: Caches loaded (data, seg) per process to avoid re-reading/decompressing .b2nd every patch.
    This cache lives inside each MultiThreadedAugmenter worker process.
    """
    def __init__(
        self,
        preproc_dir: str,
        case_ids: List[str],
        num_classes: int,
        strict: bool = True,
        ignore_labels: Tuple[int, ...] = (255, -1),
        ignore_index: int = -1,
        clamp_labels: bool = True,
        case_cache_size: int = 4,
        fg_cache_size: int = 128,
    ):
        self.preproc_dir = os.path.abspath(preproc_dir)
        if not os.path.isdir(self.preproc_dir):
            raise FileNotFoundError(f"preproc_dir does not exist: {self.preproc_dir}")

        self.num_classes = int(num_classes)
        self.ignore_labels = tuple(int(x) for x in ignore_labels)
        self.ignore_index = int(ignore_index)
        self.clamp_labels = bool(clamp_labels)

        self.case_cache_size = int(max(0, case_cache_size))
        self.fg_cache_size = int(max(0, fg_cache_size))

        self._index: Dict[str, Tuple[str, str, Optional[str]]] = self._build_index(self.preproc_dir)

        orig = list(case_ids)
        keep = [cid for cid in orig if cid in self._index]
        missing = [cid for cid in orig if cid not in self._index]

        if missing:
            print(f"[preproc] WARNING: {len(missing)}/{len(orig)} case_ids missing in {self.preproc_dir}")
            print(f"[preproc] Missing sample: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

        if strict and missing:
            raise FileNotFoundError(
                f"{len(missing)} case(s) from splits not found as .b2nd+_seg.b2nd in {self.preproc_dir}. "
                f"Sample missing: {missing[:10]}"
            )

        if len(keep) == 0:
            raise FileNotFoundError(
                f"No valid .b2nd cases found for provided IDs in {self.preproc_dir}. "
                f"Expected <cid>.b2nd and <cid>_seg.b2nd."
            )

        self.case_ids = keep

        # FG caches
        self._fg_cache: Dict[str, np.ndarray] = {}
        self._fg_order: List[str] = []

        self._class_fg_cache: Dict[str, Dict[int, np.ndarray]] = {}
        self._class_fg_order: List[str] = []

        self._case_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._case_order: List[str] = []

    @staticmethod
    def _build_index(preproc_dir: str) -> Dict[str, Tuple[str, str, Optional[str]]]:
        files = os.listdir(preproc_dir)
        data_map: Dict[str, str] = {}
        seg_map: Dict[str, str] = {}
        pkl_map: Dict[str, str] = {}

        for fn in files:
            full = os.path.join(preproc_dir, fn)
            if fn.endswith("_seg.b2nd"):
                cid = fn[:-len("_seg.b2nd")]
                seg_map[cid] = full
            elif fn.endswith(".b2nd"):
                cid = fn[:-len(".b2nd")]
                data_map[cid] = full
            elif fn.endswith(".pkl"):
                cid = fn[:-len(".pkl")]
                pkl_map[cid] = full

        out: Dict[str, Tuple[str, str, Optional[str]]] = {}
        for cid, dp in data_map.items():
            sp = seg_map.get(cid, None)
            if sp is None:
                continue
            out[cid] = (dp, sp, pkl_map.get(cid, None))
        return out

    def __len__(self):
        return len(self.case_ids)

    def _cache_get_case(self, cid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.case_cache_size <= 0:
            return None
        item = self._case_cache.get(cid, None)
        if item is None:
            return None
        try:
            self._case_order.remove(cid)
        except ValueError:
            pass
        self._case_order.append(cid)
        return item

    def _cache_put_case(self, cid: str, data: np.ndarray, seg: np.ndarray) -> None:
        if self.case_cache_size <= 0:
            return
        if cid in self._case_cache:
            try:
                self._case_order.remove(cid)
            except ValueError:
                pass
        self._case_cache[cid] = (data, seg)
        self._case_order.append(cid)
        while len(self._case_order) > self.case_cache_size:
            old = self._case_order.pop(0)
            self._case_cache.pop(old, None)

    def load_case(self, cid: str) -> Tuple[np.ndarray, np.ndarray]:
        if cid not in self._index:
            raise FileNotFoundError(f"Missing case '{cid}' in {self.preproc_dir} (no .b2nd+_seg.b2nd pair)")

        cached = self._cache_get_case(cid)
        if cached is not None:
            return cached[0], cached[1]

        data_path, seg_path, _props_path = self._index[cid]

        try:
            import blosc2
        except Exception as e:
            raise ImportError("blosc2 is required to read nnU-Net .b2nd files. Install with: pip install blosc2") from e

        data = np.asarray(blosc2.open(data_path))
        seg = np.asarray(blosc2.open(seg_path))

        if data.ndim == 3:
            data = data[None]
        elif data.ndim != 4:
            raise ValueError(f"Unexpected data shape for '{cid}': {data.shape} from {data_path}")
        data = data.astype(np.float32, copy=False)

        if seg.ndim == 3:
            seg = seg[None]
        elif seg.ndim == 4 and seg.shape[0] != 1:
            seg = seg[:1]
        elif seg.ndim != 4:
            raise ValueError(f"Unexpected seg shape for '{cid}': {seg.shape} from {seg_path}")
        seg = seg.astype(np.int16, copy=False)

        seg = _map_ignore_labels_to_index(seg, self.ignore_labels, ignore_index=self.ignore_index)
        if self.clamp_labels:
            seg = _clamp_valid_labels(seg, num_classes=self.num_classes, ignore_index=self.ignore_index)

        self._cache_put_case(cid, data, seg)
        return data, seg

    def get_foreground_coords(self, cid: str, max_points: int = 200_000, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        if self.fg_cache_size > 0:
            fg = self._fg_cache.get(cid, None)
            if fg is not None:
                try:
                    self._fg_order.remove(cid)
                except ValueError:
                    pass
                self._fg_order.append(cid)
                return fg

        _, seg = self.load_case(cid)
        fg = np.argwhere((seg[0] > 0) & (seg[0] != self.ignore_index))

        if fg.shape[0] > max_points:
            rng_choice = rng if rng is not None else np.random.RandomState(_stable_seed_from_cid(cid, extra=11))
            idx = rng_choice.choice(fg.shape[0], size=max_points, replace=False)
            fg = fg[idx]

        if self.fg_cache_size > 0:
            if cid in self._fg_cache:
                try:
                    self._fg_order.remove(cid)
                except ValueError:
                    pass
            self._fg_cache[cid] = fg
            self._fg_order.append(cid)
            while len(self._fg_order) > self.fg_cache_size:
                old = self._fg_order.pop(0)
                self._fg_cache.pop(old, None)

        return fg

    def get_class_foreground_coords(
        self,
        cid: str,
        labels: Optional[List[int]] = None,
        max_points_per_class: int = 200_000,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[int, np.ndarray]:
        if labels is None:
            labels = [i for i in range(1, self.num_classes)]

        if self.fg_cache_size > 0:
            cached = self._class_fg_cache.get(cid, None)
            if cached is not None:
                try:
                    self._class_fg_order.remove(cid)
                except ValueError:
                    pass
                self._class_fg_order.append(cid)
                if set(cached.keys()) == set(labels):
                    return cached
                want = set(labels)
                return {k: v for k, v in cached.items() if k in want}

        _, seg = self.load_case(cid)
        seg0 = seg[0]

        out: Dict[int, np.ndarray] = {}
        for lb in labels:
            lb = int(lb)
            if lb <= 0:
                continue
            coords = np.argwhere(seg0 == lb)
            if coords.shape[0] > max_points_per_class:
                rng_choice = rng if rng is not None else np.random.RandomState(_stable_seed_from_cid(cid, extra=23 + lb))
                idx = rng_choice.choice(coords.shape[0], size=max_points_per_class, replace=False)
                coords = coords[idx]
            if coords.shape[0] > 0:
                out[lb] = coords

        if self.fg_cache_size > 0:
            if cid in self._class_fg_cache:
                try:
                    self._class_fg_order.remove(cid)
                except ValueError:
                    pass
            self._class_fg_cache[cid] = out
            self._class_fg_order.append(cid)
            while len(self._class_fg_order) > self.fg_cache_size:
                old = self._class_fg_order.pop(0)
                self._class_fg_cache.pop(old, None)

        return out

    def sample_foreground_center_per_class(
        self,
        cid: str,
        rng: np.random.RandomState,
        labels: Optional[List[int]] = None,
        max_points_per_class: int = 200_000,
    ) -> Optional[Tuple[int, int, int]]:
        per_class = self.get_class_foreground_coords(cid, labels=labels, max_points_per_class=max_points_per_class, rng=rng)
        if not per_class:
            return None
        present = list(per_class.keys())
        lb = int(present[int(rng.randint(0, len(present)))])
        coords = per_class[lb]
        if coords.shape[0] == 0:
            return None
        c = coords[int(rng.randint(0, coords.shape[0]))]
        return int(c[0]), int(c[1]), int(c[2])


# -----------------------------------------------------------------------------
# B) Patch sampler DataLoader for MultiThreadedAugmenter
# -----------------------------------------------------------------------------
class NnUNetPatchDataLoader:
    def __init__(
        self,
        dataset: NnUNetPreprocessedDataset,
        patch_size: Tuple[int, int, int],
        batch_size: int,
        oversample_foreground_percent: float,
        seed: int = 42,
        pad_value_data: float = 0.0,
        pad_value_seg: int = -1,
        fg_sampling_mode: str = "perclass",
        fg_max_points_per_class: int = 200_000,
        reuse_within_batch: bool = True,
    ):
        self.dataset = dataset
        self.patch_size = tuple(int(x) for x in patch_size)
        self.batch_size = int(batch_size)
        self.oversample_fg = float(oversample_foreground_percent)
        self.pad_value_data = float(pad_value_data)
        self.pad_value_seg = int(pad_value_seg)
        self.fg_sampling_mode = str(fg_sampling_mode).lower().strip()
        if self.fg_sampling_mode not in {"any", "perclass"}:
            raise ValueError(f"fg_sampling_mode must be 'any' or 'perclass' (got {self.fg_sampling_mode})")
        self.fg_max_points_per_class = int(max(0, fg_max_points_per_class))
        self.reuse_within_batch = bool(reuse_within_batch)

        self._base_seed = int(seed)
        self._thread_id = 0
        self.rng = np.random.RandomState(self._base_seed)

    def set_thread_id(self, thread_id: int):
        self._thread_id = int(thread_id)
        self.rng = np.random.RandomState(self._base_seed + 1000 * self._thread_id + 17)

    def reset(self):
        return

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def _random_case_id(self) -> str:
        idx = int(self.rng.randint(0, len(self.dataset.case_ids)))
        return self.dataset.case_ids[idx]

    def _crop_with_pad(
        self,
        arr: np.ndarray,
        start_zyx: Tuple[int, int, int],
        patch_zyx: Tuple[int, int, int],
        pad_value: float
    ) -> np.ndarray:
        z, y, x = arr.shape[1:]
        pz, py, px = patch_zyx
        sz, sy, sx = start_zyx
        ez, ey, ex = sz + pz, sy + py, sx + px

        pad_before = [max(0, -sz), max(0, -sy), max(0, -sx)]
        pad_after  = [max(0, ez - z), max(0, ey - y), max(0, ex - x)]

        vz0, vy0, vx0 = max(0, sz), max(0, sy), max(0, sx)
        vz1, vy1, vx1 = min(z, ez), min(y, ey), min(x, ex)

        cropped = arr[:, vz0:vz1, vy0:vy1, vx0:vx1]

        if any(pad_before) or any(pad_after):
            pad_width = (
                (0, 0),
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            )
            cropped = np.pad(cropped, pad_width, mode="constant", constant_values=pad_value)

        if cropped.shape[1:] != patch_zyx:
            cropped = cropped[:, :pz, :py, :px]
        return cropped

    def _sample_start(
        self,
        img_shape_zyx: Tuple[int, int, int],
        center_zyx: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[int, int, int]:
        z, y, x = img_shape_zyx
        pz, py, px = self.patch_size
        if center_zyx is None:
            sz = self.rng.randint(-pz // 2, max(1, z - pz + pz // 2 + 1))
            sy = self.rng.randint(-py // 2, max(1, y - py + py // 2 + 1))
            sx = self.rng.randint(-px // 2, max(1, x - px + px // 2 + 1))
            return int(sz), int(sy), int(sx)
        cz, cy, cx = center_zyx
        return int(cz - pz // 2), int(cy - py // 2), int(cx - px // 2)

    def generate_train_batch(self) -> Dict[str, np.ndarray]:
        data_list, seg_list = [], []

        bs = int(self.batch_size)

        # Deterministic nnU-Net-like foreground oversampling per batch:
        # choose exactly n_fg items to be forced-foreground.
        if bs == 2:
            n_fg = 1
        else:
            n_fg = int(round(bs * float(self.oversample_fg)))
            n_fg = max(0, min(bs, n_fg))
        fg_indices = set(self.rng.choice(bs, size=n_fg, replace=False).tolist()) if n_fg > 0 else set()

        inbatch: Dict[str, Tuple[np.ndarray, np.ndarray]] = {} if self.reuse_within_batch else {}

        for bi in range(bs):
            cid = self._random_case_id()
            if self.reuse_within_batch and cid in inbatch:
                data, seg = inbatch[cid]
            else:
                data, seg = self.dataset.load_case(cid)
                if self.reuse_within_batch:
                    inbatch[cid] = (data, seg)

            zyx = data.shape[1:]
            center = None

            if bi in fg_indices:
                if self.fg_sampling_mode == "perclass":
                    center = self.dataset.sample_foreground_center_per_class(
                        cid, rng=self.rng, max_points_per_class=self.fg_max_points_per_class
                    )
                else:
                    fg = self.dataset.get_foreground_coords(cid, rng=self.rng)
                    if fg.shape[0] > 0:
                        center = tuple(int(v) for v in fg[int(self.rng.randint(0, fg.shape[0]))])

            start = self._sample_start(zyx, center_zyx=center)
            crop_d = self._crop_with_pad(data, start, self.patch_size, pad_value=self.pad_value_data)
            crop_s = self._crop_with_pad(seg, start, self.patch_size, pad_value=self.pad_value_seg)

            data_list.append(crop_d)
            seg_list.append(crop_s)

        batch_data = np.ascontiguousarray(np.stack(data_list, axis=0).astype(np.float32, copy=False))
        batch_seg  = np.ascontiguousarray(np.stack(seg_list, axis=0).astype(np.int16, copy=False))
        return {"data": batch_data, "seg": batch_seg}


# -----------------------------------------------------------------------------
# C) Augmentation graph (anisotropy-aware) with correct ignore border
# -----------------------------------------------------------------------------
def build_nnunet_like_augmentation(
    patch_size_zyx: Tuple[int, int, int],
    spacing_zyx: Optional[Tuple[float, float, float]] = None,
    seg_border_value: int = -1,
) -> AbstractTransform:
    rot_z  = 30.0 / 180.0 * np.pi
    rot_xy = 30.0 / 180.0 * np.pi

    anisotropic = False
    if spacing_zyx is not None:
        anisotropic = is_anisotropic_spacing(tuple(float(x) for x in spacing_zyx), ratio=3.0)

    if anisotropic:
        rot_xy = 0.0
        p_el, p_rot, p_scale = 0.10, 0.20, 0.20
    else:
        p_el, p_rot, p_scale = 0.20, 0.20, 0.20

    simlr_ignore_axes = (0,) if anisotropic else None
    do_elastic = (not anisotropic)
    print(
        f"[aug] spacing_zyx={spacing_zyx} anisotropic={anisotropic} "
        f"do_elastic={do_elastic} rot_xy={'0' if anisotropic else '30deg'} "
        f"order_seg=0 simlr_ignore_axes={simlr_ignore_axes}",
        flush=True,
    )

    tr = [
        SpatialTransform(
            patch_size_zyx,
            do_elastic_deform=do_elastic,
            alpha=(0.0, 900.0),
            sigma=(9.0, 13.0),
            do_rotation=True,
            angle_x=(-rot_xy, rot_xy),
            angle_y=(-rot_xy, rot_xy),
            angle_z=(-rot_z, rot_z),
            do_scale=True,
            scale=(0.7, 1.4),
            border_mode_data="constant",
            border_cval_data=0.0,
            border_mode_seg="constant",
            border_cval_seg=int(seg_border_value),
            order_data=3,
            order_seg=0,
            random_crop=False,
            p_el_per_sample=p_el,
            p_rot_per_sample=p_rot,
            p_scale_per_sample=p_scale,
        ),
        MirrorTransform(axes=(0, 1, 2)),
        GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.15),
        GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        ContrastAugmentationTransform(contrast_range=(0.75, 1.25), p_per_sample=0.15),
        GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True, p_per_sample=0.15),
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1.0),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=simlr_ignore_axes,
        ),
    ]
    return Compose(tr)


# -----------------------------------------------------------------------------
# D) MultiThreadedAugmenter wrapper (signature-robust)
# -----------------------------------------------------------------------------
def make_mta(loader, transform, num_processes: int, seed: int, pin_memory: bool = True, cache: int = 6):
    sig = inspect.signature(MultiThreadedAugmenter.__init__).parameters
    kwargs = {}

    if "num_cached_per_thread" in sig:
        kwargs["num_cached_per_thread"] = int(cache)
    elif "num_cached_per_process" in sig:
        kwargs["num_cached_per_process"] = int(cache)
    elif "num_cached_per_worker" in sig:
        kwargs["num_cached_per_worker"] = int(cache)

    if "seeds" in sig:
        kwargs["seeds"] = [int(seed) + i for i in range(int(max(1, num_processes)))]
    elif "seed" in sig:
        kwargs["seed"] = int(seed)

    if "pin_memory" in sig:
        kwargs["pin_memory"] = bool(pin_memory)

    if "wait_time" in sig:
        kwargs["wait_time"] = 0.02

    base_args = [loader, transform, int(max(1, num_processes))]
    return MultiThreadedAugmenter(*base_args, **kwargs)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class MainHeadPredictor(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        y = self.base(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def build_model(name: str, in_channels: int, out_channels: int,
                roi_size: Tuple[int, int, int], deep_supervision: bool, args=None):
    name = name.lower()

    if name == "unet3d":
        return UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            num_res_units=2, act="LEAKYRELU", norm="INSTANCE", dropout=0.0
        )

    if name == "unetr":
        return UNETR(
            in_channels=in_channels, out_channels=out_channels,
            img_size=tuple(roi_size), feature_size=16, hidden_size=768,
            mlp_dim=3072, num_heads=12, norm_name="instance",
            res_block=True, dropout_rate=0.0
        )

    if name == "dynunet":
        kernels = getattr(args, "dynunet_kernels", None)
        strides = getattr(args, "dynunet_strides", None)
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
            "deep_supervision": bool(deep_supervision),
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
        # --- user knobs ---
        fs = int(getattr(args, "swin_feature_size", 24))
        heads = tuple(getattr(args, "swin_heads", [3, 6, 12, 24]))
        depths = tuple(getattr(args, "swin_depths", [2, 2, 2, 1]))
        base_win = int(getattr(args, "swin_window", 7))
        aniso_wz = int(getattr(args, "swin_aniso_window_z", 4))
        use_ckpt = bool(getattr(args, "swin_use_checkpoint", True))
        drop_path = float(getattr(args, "swin_drop_path", 0.1))
        force_int = bool(getattr(args, "swin_force_int", False))
        auto_aniso = bool(getattr(args, "swin_auto_aniso", True))
        aniso_ratio = float(getattr(args, "aniso_ratio", 3.0))

        # explicit override values (already normalized in main())
        patch_size_user = getattr(args, "swin_patch_size", None)   # None | int | tuple3
        window_user = getattr(args, "swin_window_size", None)      # None | int | tuple3

        if fs % 12 != 0:
            raise ValueError(f"--swin-feature-size must be divisible by 12; got {fs}")

        # --- auto anisotropy adaptation (nnU-Net-like) ---
        patch_size = 2
        window_size = base_win
        if auto_aniso:
            spacing_zyx = tuple(getattr(args, "_spacing_zyx_for_model", (1.0, 1.0, 1.0)))
            auto = choose_swin_params_from_plans(
                roi_size_zyx=tuple(roi_size),
                spacing_zyx=tuple(spacing_zyx),
                base_window=base_win,
                aniso_ratio=aniso_ratio,
                aniso_wz=aniso_wz,
                force_int=force_int,
            )
            patch_size = auto["patch_size"]
            window_size = auto["window_size"]
            print(
                f"[swin] auto_aniso={auto_aniso} anisotropic={auto['anisotropic']} "
                f"patch_size={patch_size} window_size={window_size} roi={tuple(roi_size)} spacing={spacing_zyx}",
                flush=True,
            )

        # --- user overrides win over auto ---
        if patch_size_user is not None:
            patch_size = patch_size_user
        if window_user is not None:
            window_size = window_user

        base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=fs,
            depths=depths,
            num_heads=heads,
            norm_name="instance",
            spatial_dims=3,
        )

        # Important: tie architecture to actual roi_size (nnU-Net-ish reality)
        if _has_param(SwinUNETR, "img_size"):
            base["img_size"] = tuple(roi_size)

        # patch_size/window_size are version dependent; only pass if supported
        if _has_param(SwinUNETR, "patch_size"):
            base["patch_size"] = patch_size
        if _has_param(SwinUNETR, "window_size"):
            base["window_size"] = window_size

        optional = {
            "qkv_bias": True,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": float(drop_path),
            "normalize": True,
            # keep patch_norm only if your MONAI has it
            "patch_norm": False,
            "use_checkpoint": use_ckpt,
            "downsample": "merging",
            "use_v2": False,
        }
        base.update({k: v for k, v in optional.items() if _has_param(SwinUNETR, k)})

        base = _prune_kwargs(SwinUNETR, base)
        return SwinUNETR(**base)

    raise ValueError(f"Unknown model: {name}")


# -----------------------------------------------------------------------------
# TRUE ignore-masked Dice + CE
# -----------------------------------------------------------------------------
def masked_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    num_classes: int,
    include_background: bool,
    smooth_nr: float,
    smooth_dr: float,
    batch_dice: bool = True,
    ignore_empty: bool = True,
) -> torch.Tensor:
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError(f"Expected target (B,1,...) got {tuple(target.shape)}")

    t = target.squeeze(1).long()
    valid = (t != ignore_index)

    t0 = t.clone()
    t0[~valid] = 0

    p = torch.softmax(logits, dim=1)
    y = F.one_hot(t0, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    m = valid.unsqueeze(1).float()
    p = p * m
    y = y * m

    if not include_background:
        p = p[:, 1:]
        y = y[:, 1:]

    spatial_dims = tuple(range(2, p.ndim))
    intersect = (p * y).sum(dim=spatial_dims)
    denom = (p.sum(dim=spatial_dims) + y.sum(dim=spatial_dims))
    gt_sum = y.sum(dim=spatial_dims)

    if batch_dice:
        intersect = intersect.sum(dim=0)
        denom = denom.sum(dim=0)
        gt_sum = gt_sum.sum(dim=0)

    dice = (2.0 * intersect + smooth_nr) / (denom + smooth_dr)

    if ignore_empty:
        present = gt_sum > 0
        if present.any():
            dice = dice[present]
        else:
            return logits.new_tensor(0.0)

    return 1.0 - dice.mean()


class DiceCEIgnoreLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -1,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch_dice: bool = True,
        ignore_empty: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.include_background = bool(include_background)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch_dice = bool(batch_dice)
        self.ignore_empty = bool(ignore_empty)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.squeeze(1).long()
        ce = self.ce(logits, t)
        dice = masked_soft_dice_loss(
            logits=logits,
            target=target,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
            include_background=self.include_background,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
            batch_dice=self.batch_dice,
            ignore_empty=self.ignore_empty,
        )
        return self.dice_weight * dice + self.ce_weight * ce


def compute_loss_with_ds(loss_fn: nn.Module, logits, target: torch.Tensor, ds_weights: Optional[np.ndarray] = None) -> torch.Tensor:
    if not isinstance(logits, (list, tuple)):
        return loss_fn(logits, target)

    outs = list(logits)
    if ds_weights is None:
        ds_weights = nnunet_deep_supervision_weights(len(outs))

    total = 0.0
    for i, out in enumerate(outs):
        if out.shape[2:] != target.shape[2:]:
            tgt_i = F.interpolate(target.float(), size=out.shape[2:], mode="nearest").long()
        else:
            tgt_i = target
        total = total + float(ds_weights[i]) * loss_fn(out, tgt_i)
    return total


# -----------------------------------------------------------------------------
# nnU-Net-like Dice evaluation helpers (ignore-aware)
# -----------------------------------------------------------------------------
def _dice_stats_ignore(
    gt: torch.Tensor,
    pred: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    eps: float = 1e-8,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    if gt.shape != pred.shape:
        raise ValueError(f"gt/pred shape mismatch: gt={tuple(gt.shape)} pred={tuple(pred.shape)}")

    valid = (gt != int(ignore_index))
    dices: Dict[int, float] = {}
    present: Dict[int, int] = {}

    for c in range(1, int(num_classes)):
        gt_c = (gt == c) & valid
        n_gt = int(gt_c.sum().item())
        if n_gt <= 0:
            continue
        pred_c = (pred == c) & valid
        tp = int((gt_c & pred_c).sum().item())
        fp = int((~gt_c & pred_c & valid).sum().item())
        fn = int((gt_c & ~pred_c).sum().item())
        denom = (2 * tp + fp + fn)
        d = (2.0 * tp) / (denom + eps)
        dices[c] = float(d)
        present[c] = n_gt

    return dices, present


def _aggregate_nnunet_style(
    per_case_dices: List[Dict[int, float]],
    num_classes: int,
) -> Tuple[float, Dict[int, float], Dict[int, int]]:
    by_class: Dict[int, List[float]] = {c: [] for c in range(1, int(num_classes))}
    for d in per_case_dices:
        for c, v in d.items():
            if c in by_class:
                by_class[c].append(float(v))

    class_means: Dict[int, float] = {}
    n_present: Dict[int, int] = {}
    for c in range(1, int(num_classes)):
        vals = by_class[c]
        if len(vals) == 0:
            continue
        class_means[c] = float(np.mean(vals))
        n_present[c] = int(len(vals))

    mean_fg = float(np.mean(list(class_means.values()))) if len(class_means) else 0.0
    return mean_fg, class_means, n_present


# -----------------------------------------------------------------------------
# Validation (launcher-compatible)
# -----------------------------------------------------------------------------
def validate_patches(
    model: nn.Module,
    val_gen,
    device,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    ignore_index: int,
    max_batches: int = 32,
    debug_label4: bool = False,
    debug_class: int = 4,
    print_per_class: bool = True,
) -> float:
    model.eval()
    per_case_dices: List[Dict[int, float]] = []
    with torch.no_grad():
        for _ in range(int(max_batches)):
            batch = next(val_gen)
            x = torch.as_tensor(batch["data"]).to(device, non_blocking=True)
            y = torch.as_tensor(batch["seg"]).to(device, non_blocking=True).long()

            logits = sliding_window_inference(
                x, patch_size, sw_batch_size=1, predictor=MainHeadPredictor(model),
                overlap=0.5, mode="gaussian"
            )
            pred = torch.argmax(logits, dim=1, keepdim=True)

            if debug_label4:
                valid = (y != ignore_index)
                y_valid = y[valid].view(-1)
                p_valid = pred[valid].view(-1)
                gt_counts = torch.bincount(y_valid.to(torch.long), minlength=num_classes).detach().cpu().tolist()
                pr_counts = torch.bincount(p_valid.to(torch.long), minlength=num_classes).detach().cpu().tolist()
                dc = int(debug_class)
                cid_str = str(batch.get("cid", "patch"))
                print(
                    f"[val][{cid_str}] gt_c{dc}={gt_counts[dc] if dc < len(gt_counts) else -1} "
                    f"pred_c{dc}={pr_counts[dc] if dc < len(pr_counts) else -1} "
                    f"gt_hist={gt_counts} pred_hist={pr_counts}",
                    flush=True
                )

            gt_b = y[:, 0]
            pr_b = pred[:, 0]
            for bi in range(gt_b.shape[0]):
                d_i, _ = _dice_stats_ignore(
                    gt=gt_b[bi], pred=pr_b[bi],
                    num_classes=num_classes, ignore_index=ignore_index
                )
                per_case_dices.append(d_i)

    mean_fg, class_means, n_present = _aggregate_nnunet_style(per_case_dices, num_classes=num_classes)

    if print_per_class:
        parts = []
        for c in range(1, int(num_classes)):
            if c in class_means:
                parts.append(f"c{c}:{class_means[c]:.4f} (n={n_present.get(c,0)})")
            else:
                parts.append(f"c{c}:NA (n=0)")
        print("[val] patch per-class dice: " + " | ".join(parts), flush=True)
        print(f"[val] patch mean_fg_dice: {mean_fg:.4f}", flush=True)

    if not np.isfinite(mean_fg):
        raise RuntimeError("[val] No valid validation patches were evaluated (all dice NaN / no GT fg present).")
    return float(mean_fg)


def validate_full_cases(
    model: nn.Module,
    dataset: "NnUNetPreprocessedDataset",
    case_ids: List[str],
    device,
    roi_size: Tuple[int, int, int],
    num_classes: int,
    ignore_index: int,
    max_cases: int = 8,
    debug_label4: bool = False,
    debug_class: int = 4,
    print_per_class: bool = True,
) -> float:
    model.eval()
    per_case_dices: List[Dict[int, float]] = []
    with torch.no_grad():
        sel = list(case_ids) if int(max_cases) <= 0 else list(case_ids)[: int(max_cases)]
        if len(sel) == 0:
            raise RuntimeError("[val] No validation cases selected (empty case list).")
        print(f"[val] fullcase evaluating n_cases={len(sel)} (requested max_cases={max_cases})", flush=True)

        for cid in sel:
            x_np, y_np = dataset.load_case(cid)
            x = torch.from_numpy(x_np[None]).to(device)
            y = torch.from_numpy(y_np[None]).to(device).long()

            logits = sliding_window_inference(
                x, roi_size, sw_batch_size=1, predictor=MainHeadPredictor(model),
                overlap=0.5, mode="gaussian"
            )
            pred = torch.argmax(logits, dim=1, keepdim=True)

            if debug_label4:
                valid = (y != ignore_index)
                y_valid = y[valid].view(-1)
                p_valid = pred[valid].view(-1)
                gt_counts = torch.bincount(y_valid.to(torch.long), minlength=num_classes).detach().cpu().tolist()
                pr_counts = torch.bincount(p_valid.to(torch.long), minlength=num_classes).detach().cpu().tolist()
                dc = int(debug_class)
                gt_c = int(gt_counts[dc]) if dc < len(gt_counts) else -1
                pr_c = int(pr_counts[dc]) if dc < len(pr_counts) else -1
                print(f"[val][{cid}] gt_c{dc}={gt_c} pred_c{dc}={pr_c} gt_hist={gt_counts} pred_hist={pr_counts}", flush=True)

            d_i, _ = _dice_stats_ignore(
                gt=y[0, 0], pred=pred[0, 0],
                num_classes=num_classes, ignore_index=ignore_index
            )
            per_case_dices.append(d_i)

    mean_fg, class_means, n_present = _aggregate_nnunet_style(per_case_dices, num_classes=num_classes)

    if print_per_class:
        parts = []
        for c in range(1, int(num_classes)):
            if c in class_means:
                parts.append(f"c{c}:{class_means[c]:.4f} (n={n_present.get(c,0)})")
            else:
                parts.append(f"c{c}:NA (n=0)")
        print(f"[val] fullcase per-class dice: " + " | ".join(parts), flush=True)
        print(f"[val] fullcase mean_fg_dice: {mean_fg:.4f}", flush=True)

    if not np.isfinite(mean_fg):
        raise RuntimeError("[val] No valid validation cases were evaluated (all dice NaN; likely no GT foreground).")
    return float(mean_fg)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    set_determinism(seed=args.seed)

    try:
        torch.set_num_threads(int(max(1, args.torch_num_threads)))
    except Exception:
        pass

    plans_cfg = load_plans(args.plans_json, configuration=args.nnunet_config)
    patch_size_plans, bs_plans = derive_nnunet_patch_and_bs(plans_cfg)

    cfg = plans_cfg["cfg"]
    spacing = tuple(float(x) for x in cfg.get("spacing", [1.0, 1.0, 1.0]))

    # Make spacing visible to build_model() for anisotropy-aware Swin config
    args._spacing_zyx_for_model = spacing

    if args.dataset_json is not None and args.num_classes is None:
        args.num_classes = infer_num_classes_from_dataset_json(args.dataset_json)
        print(f"[dataset.json] inferred num_classes={args.num_classes} from {args.dataset_json}")

    if args.num_classes is None:
        raise ValueError("num_classes is None. Provide --num-classes or --dataset-json to infer it.")

    patch_size = tuple(args.patch_size) if args.patch_size is not None else tuple(patch_size_plans)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(bs_plans)


    # ---- SwinUNETR ROI divisibility override ----
    # MONAI SwinUNETR requires each spatial dim to be divisible by 2**5 (=32) due to 5-stage downsampling.
    # nnU-Net plans can produce anisotropic Z (e.g., 16) which violates this. For SwinUNETR only, we
    # ceil the ROI to /32 in each dim so the same ROI is used consistently by:
    #   - patch sampler / crops
    #   - model img_size (init check)
    #   - sliding-window inference ROI
    if args.model.lower() == "swinunetr":
        # SwinUNETR requires each spatial dim divisible by 32. The previous behavior was to CEIL the
        # nnU-Net ROI to /32 which can *increase* memory and cause OOM. For benchmarking across GPUs,
        # we instead FLOOR to the nearest /32 (min 32) to reduce memory while staying valid.
        orig_roi = tuple(int(v) for v in patch_size)
        floored = tuple(max(32, (int(v) // 32) * 32) for v in orig_roi)
        patch_size = tuple(int(v) for v in floored)
        if patch_size != orig_roi:
            print(f"[swin] ROI override for /32 divisibility (floor): {orig_roi} -> {patch_size}", flush=True)
            if args.batch_size is None and int(bs_plans) > 1:
                print("[swin] NOTE: ROI changed for divisibility; if you OOM, try --batch-size 1.", flush=True)


    if args.model.lower() == "dynunet":
        if "conv_kernel_sizes" in cfg:
            args.dynunet_kernels = cfg["conv_kernel_sizes"]
        if "pool_op_kernel_sizes" in cfg:
            args.dynunet_strides = cfg["pool_op_kernel_sizes"]

    train_ids, val_ids = load_splits(args.splits_json, fold=args.fold)

    seg_num_classes = int(args.num_classes)
    ignore_index = int(args.ignore_index)
    ignore_labels = tuple(int(x) for x in args.ignore_labels)

    ds_tr = NnUNetPreprocessedDataset(
        args.preproc_dir, train_ids, num_classes=seg_num_classes,
        strict=(not args.allow_missing_cases),
        ignore_labels=ignore_labels, ignore_index=ignore_index, clamp_labels=True,
        case_cache_size=int(args.case_cache_size),
        fg_cache_size=int(args.fg_cache_size),
    )
    ds_va = NnUNetPreprocessedDataset(
        args.preproc_dir, val_ids, num_classes=seg_num_classes,
        strict=(not args.allow_missing_cases),
        ignore_labels=ignore_labels, ignore_index=ignore_index, clamp_labels=True,
        case_cache_size=int(args.case_cache_size),
        fg_cache_size=int(args.fg_cache_size),
    )

    # Patch sampler
    eff_fg = get_effective_fg_prob(batch_size, args.oversample_foreground_percent)

    if args.sampler == "nnunet":
        from vendored_nnunet_sampler import VendoredNnUNetPatchDataLoader as _Sampler
        dl_tr = _Sampler(
            ds_tr, patch_size=patch_size, batch_size=batch_size,
            oversample_foreground_percent=eff_fg,
            seed=args.seed + 1000 * args.fold,
            pad_value_data=0.0, pad_value_seg=ignore_index,
            fg_sampling_mode=args.fg_sampling,
            fg_perclass_selection=args.fg_perclass_selection,
            fg_max_points_per_class=args.fg_max_points_per_class,
            min_fg_voxels_in_patch=args.fg_min_voxels_in_patch,
            fg_resample_max_tries=args.fg_resample_max_tries,
            reuse_within_batch=(not args.no_reuse_within_batch),
            debug_every_n_batches=int(getattr(args, "debug_sampler_every", 0)),
            debug_class=4,
        )
    else:
        dl_tr = NnUNetPatchDataLoader(
            ds_tr, patch_size=patch_size, batch_size=batch_size,
            oversample_foreground_percent=eff_fg,
            seed=args.seed + 1000 * args.fold,
            pad_value_data=0.0, pad_value_seg=ignore_index,
            fg_sampling_mode=args.fg_sampling,
            fg_max_points_per_class=args.fg_max_points_per_class,
            reuse_within_batch=(not args.no_reuse_within_batch),
        )

    dl_va = NnUNetPatchDataLoader(
        ds_va, patch_size=patch_size, batch_size=1,
        oversample_foreground_percent=0.0,
        seed=args.seed + 1000 * args.fold + 123,
        pad_value_data=0.0, pad_value_seg=ignore_index,
        fg_sampling_mode=args.fg_sampling,
        fg_max_points_per_class=args.fg_max_points_per_class,
        reuse_within_batch=True,
    )

    aug_tr = build_nnunet_like_augmentation(patch_size, spacing_zyx=spacing, seg_border_value=ignore_index)
    aug_va = Compose([])

    print(f"[sampler] b2nd sampler={args.sampler} eff_fg_prob={eff_fg} fg_perclass_selection={getattr(args,'fg_perclass_selection','uniform')} fg_min_voxels_in_patch={getattr(args,'fg_min_voxels_in_patch',0)}")

    train_gen = make_mta(
        dl_tr, aug_tr,
        num_processes=args.num_workers,
        seed=args.seed + 17,
        pin_memory=True,
        cache=int(args.mta_cache),
    )
    val_gen = make_mta(
        dl_va, aug_va,
        num_processes=max(1, args.num_workers // 2),
        seed=args.seed + 999,
        pin_memory=True,
        cache=max(1, int(args.mta_cache // 2)),
    )

    for g in (train_gen, val_gen):
        for m in ("_start", "start"):
            if hasattr(g, m):
                try:
                    getattr(g, m)()
                except Exception:
                    pass

    out_channels = 2 if args.task == "binary" else int(args.num_classes)

    model = build_model(
        args.model, in_channels=int(args.in_channels), out_channels=out_channels,
        roi_size=patch_size, deep_supervision=args.deep_supervision, args=args
    ).to(device)

    # -------------------------
    # Optimizer / schedule
    # -------------------------
    # This trainer historically defaulted to nnU-Net-ish SGD+poly. That is a good fit for
    # convolutional architectures, but SwinUNETR generally optimizes better with AdamW + warmup/cosine.
    # We keep the existing behavior for conv models, and switch to transformer-friendly defaults
    # for SwinUNETR unless the user explicitly opts out.

    base_lr = float(args.lr)

    if args.auto_lr_scale and args.model.lower() == "segresnet":
        base_lr = base_lr * float(args.segresnet_lr_scale)
        print(f"[lr] segresnet auto scale: lr -> {base_lr:g}")

    is_swin = (args.model.lower() == "swinunetr")

    # If user kept old defaults (lr=1e-2 / weight_decay=3e-5), override for Swin unless --respect-lr.
    if is_swin and (not bool(getattr(args, "respect_lr", False))):
        if abs(base_lr - 1e-2) < 1e-12:
            base_lr = float(getattr(args, "swin_lr", 2e-4))
            print(f"[swin][opt] overriding lr default -> {base_lr:g} (use --respect-lr to disable)", flush=True)
        if abs(float(args.weight_decay) - 3e-5) < 1e-12:
            args.weight_decay = float(getattr(args, "swin_weight_decay", 5e-2))
            print(f"[swin][opt] overriding weight_decay default -> {float(args.weight_decay):g} (AdamW)", flush=True)

    if is_swin:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(getattr(args, "swin_weight_decay", args.weight_decay)),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr, momentum=0.99, nesterov=True,
            weight_decay=float(args.weight_decay),
        )

    loss_fn = DiceCEIgnoreLoss(
        num_classes=out_channels,
        ignore_index=ignore_index,
        dice_weight=float(args.loss_dice_weight),
        ce_weight=float(args.loss_ce_weight),
        include_background=bool(args.dice_include_background),
        smooth_nr=float(args.dice_smooth_nr),
        smooth_dr=float(args.dice_smooth_dr),
    )
    print(f"[loss] Dice include_background={bool(args.dice_include_background)} (default False).")

    ds_w = None
    if args.deep_supervision:
        with torch.no_grad():
            dummy = torch.zeros((1, int(args.in_channels), *patch_size), device=device)
            y_hat = model(dummy)
            if isinstance(y_hat, (list, tuple)):
                ds_w = nnunet_deep_supervision_weights(len(y_hat))
                print(f"[ds] Using nnU-Net DS weights: {ds_w.tolist()}")

    use_amp = (torch.cuda.is_available() and not args.no_amp)
    if args.model.lower() == "segresnet" and not args.force_amp_segresnet:
        use_amp = False
        print("[amp] segresnet: AMP disabled for stability (use --force-amp-segresnet to override).")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    iters_per_epoch = int(args.num_iterations_per_epoch)
    max_epochs = int(args.max_epochs)
    max_iter = max_epochs * iters_per_epoch

    # Swin schedule: warmup + cosine
    swin_warmup_epochs = int(getattr(args, "swin_warmup_epochs", 10))
    swin_min_lr = float(getattr(args, "swin_min_lr", 1e-6))

    # Optional grad accumulation (useful for Swin when batch_size must be 1)
    grad_accum_steps = int(max(1, getattr(args, "grad_accum_steps", 1)))
    if grad_accum_steps > 1:
        print(f"[opt] grad_accum_steps={grad_accum_steps}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    best_path = os.path.join(args.outdir, f"best_{args.model}_fold{args.fold:02d}.pt")
    final_path = os.path.join(args.outdir, f"final_{args.model}_fold{args.fold:02d}.pt")

    best_val = -1.0
    global_iter = 0

    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0

            for _ in range(iters_per_epoch):
                batch = next(train_gen)
                x = torch.as_tensor(batch["data"]).to(device, non_blocking=True)
                y = torch.as_tensor(batch["seg"]).to(device, non_blocking=True).long()

                if args.task == "binary":
                    lesion_id = int(args.lesion_class_id)

                    y_bin = torch.full_like(y, fill_value=ignore_index)
                    m = (y != ignore_index)

                    with torch.no_grad():
                        present = bool(torch.any((y[m] == lesion_id)).item())

                    if present:
                        y_bin[m] = (y[m] == lesion_id).long()
                    else:
                        y_bin[m] = (y[m] > 0).long()
                        if not hasattr(train, "_warned_missing_lesion_id"):
                            setattr(train, "_warned_missing_lesion_id", True)
                            y_valid = y[m].view(-1)
                            uniq = torch.unique(y_valid).detach().cpu().tolist()
                            print(
                                f"[warn][binary] lesion_class_id={lesion_id} not present in batch; "
                                f"falling back to (label>0) foreground collapse. present_labels={uniq}",
                                flush=True,
                            )
                    y = y_bin

                if int(getattr(args, "debug_sampler_every", 0)) > 0 and (global_iter % int(args.debug_sampler_every) == 0):
                    with torch.no_grad():
                        y0 = y[:, 0]
                        valid = (y0 != ignore_index)
                        y_valid = y0[valid].view(-1)
                        hist = torch.bincount(y_valid.to(torch.long), minlength=out_channels).detach().cpu().tolist()
                        dc = 4
                        c4_per = [int(torch.sum((y0[i] == dc) & valid[i]).item()) for i in range(y0.shape[0])] if dc < out_channels else []
                        print(f"[dbg][iter {global_iter}] valid_vox={int(valid.sum().item())} ignore_vox={int((~valid).sum().item())} hist={hist} label4_per_sample={c4_per}", flush=True)

                # ---- LR schedule ----
                if is_swin:
                    # Warmup in epochs, then cosine decay to swin_min_lr
                    cur_epoch_f = float(epoch - 1) + float(_ / max(1, iters_per_epoch))
                    if cur_epoch_f < float(swin_warmup_epochs):
                        lr_now = base_lr * (cur_epoch_f / max(1.0, float(swin_warmup_epochs)))
                    else:
                        t = (cur_epoch_f - float(swin_warmup_epochs)) / max(1.0, float(max_epochs - swin_warmup_epochs))
                        t = min(max(t, 0.0), 1.0)
                        lr_now = swin_min_lr + 0.5 * (base_lr - swin_min_lr) * (1.0 + float(np.cos(np.pi * t)))
                else:
                    frac = 1.0 - (float(global_iter) / float(max_iter))
                    frac = max(0.0, frac)
                    lr_now = float(base_lr) * (frac ** float(args.poly_power))

                for pg in optimizer.param_groups:
                    pg["lr"] = float(lr_now)

                # ---- gradient accumulation ----
                if (global_iter % grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = compute_loss_with_ds(loss_fn, logits, y, ds_weights=ds_w)
                    loss = loss / float(grad_accum_steps)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at iter={global_iter} (loss={loss.item()}).")

                scaler.scale(loss).backward()

                do_step = ((global_iter + 1) % grad_accum_steps) == 0
                if do_step:
                    clip_norm = float(args.grad_clip_norm)
                    if is_swin and abs(clip_norm - 12.0) < 1e-12:
                        # 12 is nnU-Net-ish SGD default; for transformers a smaller clip is safer.
                        clip_norm = float(getattr(args, "swin_grad_clip_norm", 1.0))
                    if clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

                    scaler.step(optimizer)
                    scaler.update()

                running += float(loss.detach().item()) * float(grad_accum_steps)
                global_iter += 1

            mean_loss = running / float(iters_per_epoch)

            do_val = (epoch % int(args.val_every) == 0) or (epoch == 1) or (epoch == max_epochs)
            if do_val:
                if args.val_mode == "patch":
                    val_dice = validate_patches(
                        model=model,
                        val_gen=val_gen,
                        device=device,
                        patch_size=patch_size,
                        num_classes=out_channels,
                        ignore_index=ignore_index,
                        max_batches=int(args.val_max_batches),
                        debug_label4=bool(args.debug_label4),
                    )
                    mode_str = "patch"
                else:
                    val_dice = validate_full_cases(
                        model=model,
                        dataset=ds_va,
                        case_ids=ds_va.case_ids,
                        device=device,
                        roi_size=patch_size,
                        num_classes=out_channels,
                        ignore_index=ignore_index,
                        max_cases=int(args.val_num_cases),
                        debug_label4=bool(args.debug_label4),
                    )
                    mode_str = "fullcase"

                print(f"[fold {args.fold}] epoch {epoch:04d}/{max_epochs} | loss {mean_loss:.4f} | val Dice {val_dice:.4f} ({mode_str})", flush=True)

                if val_dice > best_val:
                    best_val = val_dice
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_dice": best_val,
                            "config": vars(args),
                            "patch_size": patch_size,
                            "spacing": spacing,
                            "plans_config": plans_cfg["cfg_key"],
                        },
                        best_path,
                    )
                    print(f"[fold {args.fold}] saved new best -> {best_path}", flush=True)
            else:
                print(f"[fold {args.fold}] epoch {epoch:04d}/{max_epochs} | loss {mean_loss:.4f}", flush=True)

        torch.save(
            {
                "epoch": max_epochs,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": best_val,
                "config": vars(args),
                "patch_size": patch_size,
                "spacing": spacing,
                "plans_config": plans_cfg["cfg_key"],
            },
            final_path,
        )
        print(f"[fold {args.fold}] saved final -> {final_path}", flush=True)

    finally:
        for g in (train_gen, val_gen):
            for meth in ("_finish", "finish", "stop", "shutdown"):
                if hasattr(g, meth):
                    try:
                        getattr(g, meth)()
                    except Exception:
                        pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--plans-json", type=str, required=True)
    p.add_argument("--dataset-json", type=str, default=None)
    p.add_argument("--preproc-dir", type=str, required=True)
    p.add_argument("--splits-json", type=str, required=True)
    p.add_argument("--nnunet-config", type=str, default="3d_fullres")
    p.add_argument("--fold", type=int, default=0)

    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model", type=str, choices=["unet3d", "unetr", "dynunet", "segresnet", "swinunetr"], required=True)
    p.add_argument("--task", type=str, choices=["multiclass", "binary"], default="multiclass")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--lesion-class-id", type=int, default=3)

    p.add_argument("--oversample-foreground-percent", type=float, default=0.33)
    p.add_argument("--fg-sampling", type=str, default="perclass", choices=["perclass", "any"])
    p.add_argument("--fg-max-points-per-class", type=int, default=200000)

    p.add_argument("--sampler", type=str, default="nnunet", choices=["native", "nnunet"])
    p.add_argument("--fg-perclass-selection", type=str, default="uniform", choices=["uniform", "inverse_frequency"])
    p.add_argument("--fg-min-voxels-in-patch", type=int, default=0)
    p.add_argument("--fg-resample-max-tries", type=int, default=8)

    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--num-iterations-per-epoch", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=3e-5)

    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--val-num-cases", type=int, default=8)

    p.add_argument("--val-mode", type=str, choices=["patch", "fullcase"], default="fullcase")
    p.add_argument("--val-max-batches", type=int, default=32)

    p.add_argument("--patch-size", type=int, nargs=3, default=None)
    p.add_argument("--batch-size", type=int, default=None)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deep-supervision", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--force-amp-segresnet", action="store_true")

    p.add_argument("--ignore-index", type=int, default=-1)
    p.add_argument("--ignore-labels", type=int, nargs="*", default=[255, -1])

    p.add_argument("--loss-dice-weight", type=float, default=1.0)
    p.add_argument("--loss-ce-weight", type=float, default=1.0)
    p.add_argument("--dice-smooth-nr", type=float, default=1e-5)
    p.add_argument("--dice-smooth-dr", type=float, default=1e-5)
    p.add_argument("--dice-include-background", action="store_true")

    p.add_argument("--auto-lr-scale", action="store_true")
    p.add_argument("--segresnet-lr-scale", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=12.0)

    # -------------------------
    # Optimizer/schedule overrides (esp. for SwinUNETR)
    # -------------------------
    p.add_argument("--respect-lr", action="store_true",
                   help="Do not auto-adjust lr/weight_decay for SwinUNETR when using legacy defaults.")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over N iterations before optimizer step (useful when batch_size=1).")

    # Swin-friendly defaults (used when SwinUNETR + legacy defaults and --respect-lr is NOT set)
    p.add_argument("--swin-lr", type=float, default=2e-4)
    p.add_argument("--swin-weight-decay", type=float, default=5e-2)
    p.add_argument("--swin-warmup-epochs", type=int, default=10)
    p.add_argument("--swin-min-lr", type=float, default=1e-6)
    p.add_argument("--swin-grad-clip-norm", type=float, default=1.0)

    p.add_argument("--allow-missing-cases", action="store_true")

    p.add_argument("--case-cache-size", type=int, default=16)
    p.add_argument("--fg-cache-size", type=int, default=512)
    p.add_argument("--mta-cache", type=int, default=6)
    p.add_argument("--no-reuse-within-batch", action="store_true")
    p.add_argument("--torch-num-threads", type=int, default=1)

    p.add_argument("--debug-sampler-every", type=int, default=0)
    p.add_argument("--debug-label4", action="store_true")

    # -------------------------
    # SwinUNETR anisotropy knobs
    # -------------------------
    p.add_argument("--swin-auto-aniso", action="store_true", default=True,
                   help="Auto-adjust SwinUNETR patch/window for anisotropic spacing (nnU-Net-like).")
    p.add_argument("--aniso-ratio", type=float, default=3.0,
                   help="Anisotropy threshold: z_spacing > ratio * max(x,y).")
    p.add_argument("--swin-aniso-window-z", type=int, default=4,
                   help="Z window size to use when anisotropic (if tuple window_size supported).")
    p.add_argument("--swin-force-int", action="store_true",
                   help="Force int-valued patch/window if your SwinUNETR build rejects tuples.")

    # Optional explicit overrides (accept 1 or 3 ints; normalized below)
    p.add_argument("--swin-window-size", type=int, nargs="*", default=None,
                   help="Override window_size (either 1 int, or 3 ints).")
    p.add_argument("--swin-patch-size", type=int, nargs="*", default=None,
                   help="Override patch_size (either 1 int, or 3 ints).")

    # Existing swin params (kept)
    p.add_argument("--swin-window", type=int, default=7)
    # Smaller default to fit more GPUs; old default was 24
    p.add_argument("--swin-feature-size", type=int, default=12)
    p.add_argument("--swin-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--swin-depths", type=int, nargs=4, default=[2, 2, 2, 1])
    p.add_argument("--swin-drop-path", type=float, default=0.1,
                   help="Stochastic depth (drop path) rate for SwinUNETR (regularization).")
    p.add_argument("--swin-use-checkpoint", action="store_true", default=True)

    args = p.parse_args()

    # Normalize optional 1-or-3 int overrides to int or tuple3
    if args.swin_window_size is not None:
        if len(args.swin_window_size) == 0:
            args.swin_window_size = None
        elif len(args.swin_window_size) == 1:
            args.swin_window_size = int(args.swin_window_size[0])
        elif len(args.swin_window_size) == 3:
            args.swin_window_size = tuple(int(x) for x in args.swin_window_size)
        else:
            raise ValueError("--swin-window-size expects 1 or 3 ints.")

    if args.swin_patch_size is not None:
        if len(args.swin_patch_size) == 0:
            args.swin_patch_size = None
        elif len(args.swin_patch_size) == 1:
            args.swin_patch_size = int(args.swin_patch_size[0])
        elif len(args.swin_patch_size) == 3:
            args.swin_patch_size = tuple(int(x) for x in args.swin_patch_size)
        else:
            raise ValueError("--swin-patch-size expects 1 or 3 ints.")

    if not args.auto_lr_scale:
        args.auto_lr_scale = True

    # normalize to attribute used in train()
    args.grad_accum_steps = int(max(1, args.grad_accum_steps))

    print(
        f"[speed] case_cache_size={args.case_cache_size} fg_cache_size={args.fg_cache_size} "
        f"mta_cache={args.mta_cache} OMP/MKL/OPENBLAS/NUMEXPR=1",
        flush=True,
    )
    train(args)


# -----------------------------------------------------------------------------
# Deterministic collate for raw-NIfTI MONAI backend (kept as-is from your snippet)
# -----------------------------------------------------------------------------
class DeterministicNnUNetBatchCropCollate:
    """Deterministic per-batch foreground oversampling + center cropping for MONAI raw-NIfTI backend."""
    def __init__(
        self,
        roi_size_zyx,
        fg_prob: float,
        num_classes: int,
        ignore_index: int = -1,
        fg_sampling: str = "perclass",
        fg_max_points_per_class: int = 200_000,
        seed: int = 42,
        pad_value_image: float = 0.0,
        pad_value_label: int = -1,
    ):
        self.roi = tuple(int(x) for x in roi_size_zyx)
        self.fg_prob = float(fg_prob)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.fg_sampling = str(fg_sampling).lower().strip()
        if self.fg_sampling not in {"any", "perclass"}:
            raise ValueError(f"fg_sampling must be 'any' or 'perclass' (got {self.fg_sampling})")
        self.fg_max_points_per_class = int(fg_max_points_per_class)
        self.rng = np.random.default_rng(int(seed))
        self.pad_value_image = float(pad_value_image)
        self.pad_value_label = int(pad_value_label)

    def _crop_with_pad(self, t, start_zyx, roi_zyx, pad_value):
        import torch
        import torch.nn.functional as F
        if t.dim() == 3:
            t = t.unsqueeze(0)
        assert t.dim() == 4
        c, z, y, x = t.shape
        pz, py, px = roi_zyx
        sz, sy, sx = start_zyx
        ez, ey, ex = sz + pz, sy + py, sx + px

        vz0, vy0, vx0 = max(0, sz), max(0, sy), max(0, sx)
        vz1, vy1, vx1 = min(z, ez), min(y, ey), min(x, ex)
        cropped = t[:, vz0:vz1, vy0:vy1, vx0:vx1]

        pad_before = [max(0, -sx), max(0, -sy), max(0, -sz)]
        pad_after  = [max(0, ex - x), max(0, ey - y), max(0, ez - z)]
        if any(v > 0 for v in pad_before + pad_after):
            pads = (pad_before[0], pad_after[0], pad_before[1], pad_after[1], pad_before[2], pad_after[2])
            cropped = F.pad(cropped, pads, mode="constant", value=float(pad_value))

        cropped = cropped[:, :pz, :py, :px]
        return cropped

    def _sample_random_center(self, shape_zyx):
        z, y, x = shape_zyx
        cz = int(self.rng.integers(0, max(1, z)))
        cy = int(self.rng.integers(0, max(1, y)))
        cx = int(self.rng.integers(0, max(1, x)))
        return (cz, cy, cx)

    def _sample_fg_center_any(self, lab):
        import torch
        lab3 = lab[0] if lab.dim() == 4 else lab
        mask = (lab3 != self.ignore_index) & (lab3 > 0)
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0:
            return None
        if idx.shape[0] > self.fg_max_points_per_class:
            sel = self.rng.choice(idx.shape[0], size=self.fg_max_points_per_class, replace=False)
            idx = idx[torch.as_tensor(sel, dtype=torch.long)]
        j = int(self.rng.integers(0, idx.shape[0]))
        cz, cy, cx = (int(idx[j, 0]), int(idx[j, 1]), int(idx[j, 2]))
        return (cz, cy, cx)

    def _sample_fg_center_perclass(self, lab):
        import torch
        lab3 = lab[0] if lab.dim() == 4 else lab
        present = []
        coords_by_c = {}
        for c in range(1, self.num_classes):
            m = (lab3 == c)
            if torch.any(m):
                idx = torch.nonzero(m, as_tuple=False)
                if idx.shape[0] > self.fg_max_points_per_class:
                    sel = self.rng.choice(idx.shape[0], size=self.fg_max_points_per_class, replace=False)
                    idx = idx[torch.as_tensor(sel, dtype=torch.long)]
                present.append(c)
                coords_by_c[c] = idx
        if not present:
            return None
        csel = int(present[int(self.rng.integers(0, len(present)))])
        idx = coords_by_c[csel]
        j = int(self.rng.integers(0, idx.shape[0]))
        cz, cy, cx = (int(idx[j, 0]), int(idx[j, 1]), int(idx[j, 2]))
        return (cz, cy, cx)

    def __call__(self, batch):
        import torch
        bs = len(batch)
        eff_fg = get_effective_fg_prob(bs, self.fg_prob)
        if bs == 2:
            n_fg = 1
        else:
            n_fg = int(round(bs * float(eff_fg)))
            n_fg = max(0, min(bs, n_fg))
        fg_indices = set(self.rng.choice(bs, size=n_fg, replace=False).tolist()) if n_fg > 0 else set()

        out = []
        for bi, item in enumerate(batch):
            img = item["image"]
            lab = item.get("label", item.get("seg", None))
            if lab is None:
                raise KeyError("Expected 'label' (or 'seg') in batch item for MONAI nifti backend.")

            if img.dim() == 3:
                img = img.unsqueeze(0)
            if lab.dim() == 3:
                lab = lab.unsqueeze(0)

            _, z, y, x = img.shape
            center = None
            if bi in fg_indices:
                if self.fg_sampling == "perclass":
                    center = self._sample_fg_center_perclass(lab)
                else:
                    center = self._sample_fg_center_any(lab)
            if center is None:
                center = self._sample_random_center((z, y, x))

            pz, py, px = self.roi
            cz, cy, cx = center
            sz = int(cz - pz // 2)
            sy = int(cy - py // 2)
            sx = int(cx - px // 2)
            start = (sz, sy, sx)

            crop_img = self._crop_with_pad(img, start, self.roi, self.pad_value_image)
            crop_lab = self._crop_with_pad(lab, start, self.roi, self.pad_value_label)

            new_item = dict(item)
            new_item["image"] = crop_img
            new_item["label"] = crop_lab
            out.append(new_item)

        images = torch.stack([o["image"] for o in out], dim=0)
        labels = torch.stack([o["label"] for o in out], dim=0)
        return {"image": images, "label": labels}


if __name__ == "__main__":
    main()