#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vendored nnU-Net v2-style patch sampler (no nnunetv2 import).

PATCH GOALS (your request):
1) **Literal fidelity** to nnU-Net v2 dataloader sampling:
   - Deterministic FG oversampling: **last** ~X% of samples in the minibatch are forced-FG
     (NOT random indices).
   - Bounding-box sampling matches nnU-Net’s `get_bbox()` logic:
       * compute `need_to_pad`, allow negative lower bounds
       * random bbox_lbs ~ randint(lbs, ubs+1)
       * forced-FG bbox centered on a chosen FG voxel, clamped to [lb, ub]
   - Padding semantics: image pad=0 (configurable), seg pad=ignore_index (configurable)

2) Fix the “all ignore supervision” failure mode:
   - Guarantee that sampled bbox always intersects the image extent (per-axis overlap > 0).
   - If a bbox would not overlap (should be rare if bbox logic is correct), clamp to a safe bbox.

3) Keep your optional quality gate (`min_fg_voxels_in_patch`) as an *optional* deviation
   (default 0 -> disabled). It will never silently produce a completely invalid crop.

Compatibility patch:
- Restored `debug_class` in SamplerConfig and stored it in the loader (Option A),
  because your trainer passes `debug_class=...`.

Assumes dataset provides:
  - dataset.case_ids: List[str]
  - dataset.load_case(cid) -> (data[C,Z,Y,X], seg[1,Z,Y,X])
  - dataset.get_class_foreground_coords(cid, labels=None, max_points_per_class=...) -> Dict[int, coords[N,3]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SamplerConfig:
    patch_size: Tuple[int, int, int]
    batch_size: int
    oversample_foreground_percent: float
    seed: int
    pad_value_data: float
    pad_value_seg: int

    fg_sampling_mode: str = "perclass"         # {"perclass","any"}
    fg_perclass_selection: str = "uniform"     # {"uniform","inverse_frequency"}
    fg_max_points_per_class: int = 200_000

    min_fg_voxels_in_patch: int = 0            # 0 disables quality gate (nnU-Net default behavior)
    fg_resample_max_tries: int = 8             # only used if min_fg_voxels_in_patch > 0

    reuse_within_batch: bool = True
    debug_every_n_batches: int = 0

    # Option A: restore compatibility with trainer CLI/kwargs
    debug_class: int = 4


class VendoredNnUNetPatchDataLoader:
    """Drop-in replacement for an nnU-Net-like patch data loader."""

    def __init__(self, dataset, **kwargs):
        cfg = SamplerConfig(**kwargs)
        self.dataset = dataset

        self.patch_size = tuple(int(x) for x in cfg.patch_size)
        self.batch_size = int(cfg.batch_size)
        self.oversample_fg = float(cfg.oversample_foreground_percent)

        self.pad_value_data = float(cfg.pad_value_data)
        self.pad_value_seg = int(cfg.pad_value_seg)

        self.fg_sampling_mode = str(cfg.fg_sampling_mode).lower().strip()
        if self.fg_sampling_mode not in {"any", "perclass"}:
            raise ValueError(f"fg_sampling_mode must be 'any' or 'perclass' (got {self.fg_sampling_mode})")

        self.fg_perclass_selection = str(cfg.fg_perclass_selection).lower().strip()
        if self.fg_sampling_mode == "perclass" and self.fg_perclass_selection not in {"uniform", "inverse_frequency"}:
            raise ValueError(
                f"fg_perclass_selection must be 'uniform' or 'inverse_frequency' (got {self.fg_perclass_selection})"
            )

        self.fg_max_points_per_class = int(max(0, cfg.fg_max_points_per_class))
        self.min_fg_voxels_in_patch = int(max(0, cfg.min_fg_voxels_in_patch))
        self.fg_resample_max_tries = int(max(1, cfg.fg_resample_max_tries))
        self.reuse_within_batch = bool(cfg.reuse_within_batch)

        self.debug_every_n_batches = int(max(0, cfg.debug_every_n_batches))
        self.debug_class = int(cfg.debug_class)  # restored; may be unused but keeps API stable

        self._dbg_batches = 0
        self._dbg_all_ignore = 0

        self._base_seed = int(cfg.seed)
        self._thread_id = 0
        self.rng = np.random.RandomState(self._base_seed)

    def set_thread_id(self, thread_id: int):
        self._thread_id = int(thread_id)
        # mimic "different seed per worker" behavior
        self.rng = np.random.RandomState(self._base_seed + 1000 * self._thread_id + 17)

    def reset(self):
        return

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    # -------------------------
    # nnU-Net-like oversampling
    # -------------------------

    def _oversample_last_percent(self, sample_idx: int) -> bool:
        """
        nnU-Net v2 behavior:
          return not sample_idx < round(batch_size * (1 - oversample_foreground_percent))
        i.e., the last ~X% of samples in the batch are forced-FG.
        """
        bs = int(self.batch_size)
        cutoff = int(round(bs * (1.0 - float(self.oversample_fg))))
        return not (sample_idx < cutoff)

    # -------------------------
    # Helpers
    # -------------------------

    def _random_case_id(self) -> str:
        idx = int(self.rng.randint(0, len(self.dataset.case_ids)))
        return self.dataset.case_ids[idx]

    @staticmethod
    def _compute_lbs_ubs(
        data_shape_zyx: Tuple[int, int, int],
        patch_size_zyx: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        nnU-Net v2 style bbox sampling bounds:
          need_to_pad = max(0, patch_size - data_shape)
          lbs = - need_to_pad // 2
          ubs = data_shape + need_to_pad//2 + need_to_pad%2 - patch_size
        """
        data_shape = np.asarray(data_shape_zyx, dtype=np.int64)
        patch_size = np.asarray(patch_size_zyx, dtype=np.int64)

        need_to_pad = np.maximum(0, patch_size - data_shape)  # (3,)
        lbs = - (need_to_pad // 2)
        ubs = data_shape + (need_to_pad // 2) + (need_to_pad % 2) - patch_size
        return lbs, ubs, need_to_pad

    @staticmethod
    def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
        """length of overlap between [a0,a1) and [b0,b1)"""
        return max(0, min(a1, b1) - max(a0, b0))

    def _bbox_overlaps_image(self, bbox_lbs: np.ndarray, data_shape_zyx: Tuple[int, int, int]) -> bool:
        """
        Ensure bbox intersects image extent in all axes.
        bbox is [lb, lb+patch) in each axis.
        image is [0, shape).
        """
        patch = np.asarray(self.patch_size, dtype=np.int64)
        shape = np.asarray(data_shape_zyx, dtype=np.int64)
        bbox_ubs = bbox_lbs + patch
        for d in range(3):
            if self._interval_overlap(int(bbox_lbs[d]), int(bbox_ubs[d]), 0, int(shape[d])) <= 0:
                return False
        return True

    def _clamp_bbox_to_overlap(self, bbox_lbs: np.ndarray, lbs: np.ndarray, ubs: np.ndarray) -> np.ndarray:
        """
        If bbox would not overlap the image (should not happen in correct logic),
        clamp to [lbs, ubs] range (safe) and return.
        """
        bbox_lbs = np.asarray(bbox_lbs, dtype=np.int64)
        for d in range(3):
            bbox_lbs[d] = int(np.clip(bbox_lbs[d], int(lbs[d]), int(ubs[d])))
        return bbox_lbs

    def _randint_inclusive(self, low: int, high: int) -> int:
        """
        randint with inclusive upper bound (nnU-Net uses randint(lbs, ubs+1)).
        """
        if high < low:
            return int(low)
        return int(self.rng.randint(low, high + 1))

    # -------------------------
    # FG center selection
    # -------------------------

    def _pick_present_class(self, per_class: Dict[int, np.ndarray]) -> Optional[int]:
        """
        nnU-Net: pick eligible classes among those that have stored locations.
        We exclude background(0) and ignore/pad_value_seg if it ever appears.
        """
        labels = [int(k) for k, v in per_class.items() if v is not None and v.shape[0] > 0]
        labels = [lb for lb in labels if lb != 0 and lb != int(self.pad_value_seg)]
        if not labels:
            return None

        if len(labels) == 1:
            return int(labels[0])

        if self.fg_perclass_selection == "uniform":
            return int(labels[int(self.rng.randint(0, len(labels)))])

        # optional deviation: inverse_frequency weighting
        counts = np.asarray([max(1, per_class[lb].shape[0]) for lb in labels], dtype=np.float64)
        w = 1.0 / counts
        w = w / w.sum()
        return int(self.rng.choice(labels, p=w))

    def _sample_fg_voxel(self, cid: str) -> Tuple[Optional[Tuple[int, int, int]], Optional[int]]:
        """
        Returns (voxel_zyx, class_id) or (None, None) if no fg voxel exists.
        """
        if self.fg_sampling_mode == "any":
            _, seg = self.dataset.load_case(cid)
            seg0 = seg[0]
            coords = np.argwhere((seg0 > 0) & (seg0 != int(self.pad_value_seg)))
            if coords.shape[0] == 0:
                return None, None
            c = coords[int(self.rng.randint(0, coords.shape[0]))]
            return (int(c[0]), int(c[1]), int(c[2])), None

        per_class = self.dataset.get_class_foreground_coords(
            cid, labels=None, max_points_per_class=self.fg_max_points_per_class
        )
        if not per_class:
            return None, None

        lb = self._pick_present_class(per_class)
        if lb is None:
            return None, None
        coords = per_class.get(lb, None)
        if coords is None or coords.shape[0] == 0:
            return None, None
        c = coords[int(self.rng.randint(0, coords.shape[0]))]
        return (int(c[0]), int(c[1]), int(c[2])), int(lb)

    # -------------------------
    # Crop and pad (nnU-Net-like)
    # -------------------------

    def _crop_and_pad_nd(
        self,
        arr: np.ndarray,                     # (C,Z,Y,X) or (1,Z,Y,X)
        bbox_lbs: np.ndarray,                # (3,) lower bounds (can be negative)
        patch_zyx: Tuple[int, int, int],
        pad_value: float,
    ) -> np.ndarray:
        """
        Equivalent to nnU-Net crop_and_pad_nd for 3D.
        """
        z, y, x = arr.shape[1:]
        pz, py, px = patch_zyx
        sz, sy, sx = int(bbox_lbs[0]), int(bbox_lbs[1]), int(bbox_lbs[2])
        ez, ey, ex = sz + pz, sy + py, sx + px

        vz0, vy0, vx0 = max(0, sz), max(0, sy), max(0, sx)
        vz1, vy1, vx1 = min(z, ez), min(y, ey), min(x, ex)

        cropped = arr[:, vz0:vz1, vy0:vy1, vx0:vx1]

        pad_before = (max(0, -sz), max(0, -sy), max(0, -sx))
        pad_after = (max(0, ez - z), max(0, ey - y), max(0, ex - x))

        if any(pad_before) or any(pad_after):
            pad_width = (
                (0, 0),
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            )
            cropped = np.pad(cropped, pad_width, mode="constant", constant_values=pad_value)

        if cropped.shape[1:] != (pz, py, px):
            cropped = cropped[:, :pz, :py, :px]
        return cropped

    # -------------------------
    # Quality gate (optional)
    # -------------------------

    @staticmethod
    def _fg_count_in_patch(crop_seg: np.ndarray, target_class: Optional[int], ignore_value: int) -> int:
        seg0 = crop_seg[0]
        if target_class is None:
            return int(np.sum((seg0 > 0) & (seg0 != int(ignore_value))))
        return int(np.sum(seg0 == int(target_class)))

    # -------------------------
    # Core nnU-Net-like bbox selection
    # -------------------------

    def _get_bbox(
        self,
        data_shape_zyx: Tuple[int, int, int],
        force_fg: bool,
        fg_voxel_zyx: Optional[Tuple[int, int, int]],
    ) -> np.ndarray:
        lbs, ubs, _need = self._compute_lbs_ubs(data_shape_zyx, self.patch_size)

        if (not force_fg) or (fg_voxel_zyx is None):
            bbox_lbs = np.array(
                [self._randint_inclusive(int(lbs[d]), int(ubs[d])) for d in range(3)],
                dtype=np.int64,
            )
        else:
            voxel = np.asarray(fg_voxel_zyx, dtype=np.int64)
            patch = np.asarray(self.patch_size, dtype=np.int64)
            bbox_lbs = voxel - (patch // 2)
            for d in range(3):
                bbox_lbs[d] = max(int(lbs[d]), int(bbox_lbs[d]))
                bbox_lbs[d] = min(int(ubs[d]), int(bbox_lbs[d]))

        # Safety: guarantee overlap with image (prevents all-ignore supervision)
        if not self._bbox_overlaps_image(bbox_lbs, data_shape_zyx):
            bbox_lbs = self._clamp_bbox_to_overlap(bbox_lbs, lbs, ubs)

        return bbox_lbs

    # -------------------------
    # Batch generation
    # -------------------------

    def generate_train_batch(self) -> Dict[str, np.ndarray]:
        data_list: List[np.ndarray] = []
        seg_list: List[np.ndarray] = []
        cid_list: List[str] = []

        inbatch: Dict[str, Tuple[np.ndarray, np.ndarray]] = {} if self.reuse_within_batch else {}

        for bi in range(int(self.batch_size)):
            force_fg = self._oversample_last_percent(bi)

            do_quality = bool(force_fg and (self.min_fg_voxels_in_patch > 0))
            n_tries = int(self.fg_resample_max_tries) if do_quality else 1

            best: Optional[Tuple[int, np.ndarray, np.ndarray, str, Optional[int]]] = None
            # best = (fg_cnt, crop_d, crop_s, cid, target_class)

            for _attempt in range(n_tries):
                cid = self._random_case_id()

                if self.reuse_within_batch and cid in inbatch:
                    data, seg = inbatch[cid]
                else:
                    data, seg = self.dataset.load_case(cid)
                    if self.reuse_within_batch:
                        inbatch[cid] = (data, seg)

                shape_zyx = tuple(int(s) for s in data.shape[1:])

                fg_voxel = None
                target_class = None

                if force_fg:
                    fg_voxel, target_class = self._sample_fg_voxel(cid)
                    # nnU-Net behavior: if no fg exists, it falls back to random bbox

                bbox_lbs = self._get_bbox(shape_zyx, force_fg=force_fg, fg_voxel_zyx=fg_voxel)

                crop_d = self._crop_and_pad_nd(data, bbox_lbs, self.patch_size, pad_value=self.pad_value_data)
                crop_s = self._crop_and_pad_nd(seg,  bbox_lbs, self.patch_size, pad_value=self.pad_value_seg)

                # Detect "all ignore" seg patch (catastrophic for loss). Retry if possible.
                valid_vox = int(np.sum(crop_s[0] != int(self.pad_value_seg)))
                if valid_vox == 0:
                    self._dbg_all_ignore += 1
                    continue

                if not do_quality:
                    best = (0, crop_d, crop_s, cid, target_class)
                    break

                fg_cnt = self._fg_count_in_patch(
                    crop_s,
                    target_class if self.fg_sampling_mode == "perclass" else None,
                    self.pad_value_seg,
                )

                if (best is None) or (fg_cnt > best[0]):
                    best = (int(fg_cnt), crop_d, crop_s, cid, target_class)

                if fg_cnt >= int(self.min_fg_voxels_in_patch):
                    break

            # Hard fallback if attempts all produced all-ignore (or best None)
            if best is None:
                cid = self._random_case_id()
                data, seg = self.dataset.load_case(cid)
                shape_zyx = tuple(int(s) for s in data.shape[1:])
                bbox_lbs = self._get_bbox(shape_zyx, force_fg=False, fg_voxel_zyx=None)
                crop_d = self._crop_and_pad_nd(data, bbox_lbs, self.patch_size, pad_value=self.pad_value_data)
                crop_s = self._crop_and_pad_nd(seg,  bbox_lbs, self.patch_size, pad_value=self.pad_value_seg)
                best = (0, crop_d, crop_s, cid, None)

            _, crop_d, crop_s, best_cid, _ = best
            data_list.append(crop_d)
            seg_list.append(crop_s)
            cid_list.append(best_cid)

        batch_data = np.ascontiguousarray(np.stack(data_list, axis=0).astype(np.float32, copy=False))
        batch_seg = np.ascontiguousarray(np.stack(seg_list, axis=0).astype(np.int16, copy=False))

        if self.debug_every_n_batches > 0:
            self._dbg_batches += 1
            if (self._dbg_batches % self.debug_every_n_batches) == 0:
                print(
                    f"[sampler][tid={self._thread_id}] batches={self._dbg_batches} "
                    f"all_ignore_patches_seen={self._dbg_all_ignore} "
                    f"oversample_fg={self.oversample_fg:.3f} batch_size={self.batch_size} "
                    f"debug_class={self.debug_class}",
                    flush=True,
                )

        return {"data": batch_data, "seg": batch_seg, "cid": cid_list}
