#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid multi-dataset validator

Supported dataset modes
-----------------------
1) legacy_orig_dir
   For legacy UMD-style folders:
       orig_dir/<subject>/<subject>_seg.nii.gz, _seq.nii.gz, _label(s).nii.gz, _t2.nii.gz

2) nnunet_raw
   For nnU-Net raw-style datasets:
       raw_dataset_dir/imagesTs/<case>_0000.nii.gz
       raw_dataset_dir/labelsTs/<case>.nii.gz

Each dataset is analyzed independently into:
    output_dir/<dataset_name>/

Dataset is NOT used as an analysis factor.

Also writes top-level compiled CSVs across all datasets:
    output_dir/all_per_subject_detailed.csv
    output_dir/all_summary_per_label.csv
    output_dir/all_summary_overall.csv
    output_dir/all_pairwise_by_label.csv
    output_dir/all_pairwise_overall.csv
"""

import os, re, csv, json, glob, math
from collections import defaultdict, Counter
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG — EDIT THESE
# -----------------------------
CONFIG = {
    "output_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UM_nnUnet/results_cmig",

    "datasets": [
        {
            "name": "umd",
            "display_name": "UMD Internal Tesing",
            "source_mode": "legacy_orig_dir",
            "orig_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/UMD",
            "pred_suffix": "",
            "subjects": None,
            "labels": [1, 2, 3, 4],
            "label_names": {
                1: "Uterine muscular wall",
                2: "Uterine cavity",
                3: "Uterine myomas",
                4: "Nabothian cyst",
            },
        },
        {
            "name": "umd_external",
            "display_name": "UMD External Testing",
            "source_mode": "nnunet_raw",
            "raw_dataset_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_testprep_umd",
            "pred_suffix": "_ext",
            "subjects": None,
            "labels": [1, 2, 3, 4],
            "label_names": {
                1: "Uterine muscular wall",
                2: "Uterine cavity",
                3: "Uterine myomas",
                4: "Nabothian cyst",
            },
        },
        {
            "name": "emca",
            "display_name": "EMCA",
            "source_mode": "nnunet_raw",
            "raw_dataset_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_raw_data/Dataset101_emca",
            "pred_suffix": "_emca",
            "subjects": None,
        },
        {
            "name": "lms",
            "display_name": "LMS",
            "source_mode": "nnunet_raw",
            "raw_dataset_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_raw_data/Dataset102_LMS",
            "pred_suffix": "_lms",
            "subjects": None,
        },
    ],

    "models": [
        {"name": "UNet3D",     "pred_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_unet3d",    "pred_type": "auto"},
        {"name": "Swin-UNETR", "pred_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_swinunetr", "pred_type": "auto"},
        {"name": "nnU-Net",    "pred_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_nnunet",    "pred_type": "auto"},
    ],

    # global fallback only
    "labels": [1, 2, 3, 4],
    "label_names": {
        1: "Uterine muscular wall",
        2: "Uterine cavity",
        3: "Uterine myomas",
        4: "Nabothian cyst",
    },

    "label_map": {},
    "pred_type_global": None,
    "proba_thresh": 0.5,
    "subjects": None,
    "raw_split": "Ts",
    "gt_name_patterns": [
        "{id}_seg.nii.gz",
        "{id}_seq.nii.gz",
        "{id}_label.nii.gz",
        "{id}_labels.nii.gz",
    ],
    "make_overlays": False,
    "save_case_qc": True,
    "compare_metric": "dice",
    "n_boot": 5000,
    "seed": 1337,

    # Visualization tuning
    "overlay_alpha": 0.22,
    "jitter_points": True,
    "jitter_width": 0.08,
    "jitter_alpha": 0.55,
    "jitter_size": 18,
}
# -----------------------------


# -----------------------------
# Math helpers
# -----------------------------
def _z(u: float) -> float:
    u = float(np.clip(u, 1e-12, 1 - 1e-12))
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 0.97575
    if u < plow:
        q = math.sqrt(-2 * math.log(u))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return float(-num / den)
    if u > phigh:
        q = math.sqrt(-2 * math.log(1 - u))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return float(num / den)
    q = u - 0.5
    r = q*q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return float(num / den)

def paired_bootstrap(a, b, n_boot=5000, seed=1337):
    rng = np.random.default_rng(seed)
    x = np.asarray(a, float); y = np.asarray(b, float)
    assert x.shape == y.shape and x.ndim == 1 and x.size >= 2
    n = x.size
    diff_obs = float(np.mean(x - y))
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(x[idx] - y[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff_obs, (float(lo), float(hi)), boots

def paired_bca_ci(a, b, n_boot=5000, alpha=0.05, seed=1337):
    rng = np.random.default_rng(seed)
    x = np.asarray(a, float); y = np.asarray(b, float)
    assert x.shape == y.shape and x.ndim == 1 and x.size >= 2
    d = x - y
    theta_hat = float(np.mean(d))
    boots = np.empty(n_boot, float)
    n = d.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(d[idx]))
    prop = float(np.clip((boots < theta_hat).mean(), 1e-12, 1-1e-12))
    z0 = _z(prop)
    s = d.sum()
    jk = np.empty(n, float)
    for i in range(n):
        jk[i] = float((s - d[i]) / (n - 1))
    jk_bar = float(jk.mean())
    num = float(np.sum((jk_bar - jk)**3))
    den = float(6.0 * (np.sum((jk_bar - jk)**2) ** 1.5 + 1e-12))
    a_hat = num / (den + 1e-18)
    def bca_alpha(q):
        z = _z(q); adj = z0 + z; denom = 1 - a_hat * adj
        z_bca = z0 + adj / (denom + 1e-18)
        return float(np.clip(0.5*(1+math.erf(z_bca/math.sqrt(2))), 1e-12, 1-1e-12))
    lo = float(np.quantile(boots, bca_alpha(alpha/2)))
    hi = float(np.quantile(boots, bca_alpha(1 - alpha/2)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        lo, hi = np.percentile(boots, [100*(alpha/2), 100*(1-alpha/2)])
    return lo, hi

def cliffs_delta(x, y):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    y = np.asarray([v for v in y if np.isfinite(v)], float)
    nx, ny = x.size, y.size
    assert nx >= 1 and ny >= 1
    x = np.sort(x); y = np.sort(y)
    i = j = more = less = 0
    while i < nx and j < ny:
        if x[i] > y[j]:
            more += (nx - i); j += 1
        elif x[i] < y[j]:
            less += (ny - j); i += 1
        else:
            xv = x[i]; yv = y[j]
            cx = 0
            while i < nx and x[i] == xv:
                i += 1; cx += 1
            cy = 0
            while j < ny and y[j] == yv:
                j += 1; cy += 1
    return float((more - less) / (nx * ny))

def cliffs_delta_boot_ci(x, y, n_boot=5000, seed=1337, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    y = np.asarray([v for v in y if np.isfinite(v)], float)
    nx, ny = x.size, y.size
    d0 = cliffs_delta(x, y)
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        xb = x[rng.integers(0, nx, size=nx)]
        yb = y[rng.integers(0, ny, size=ny)]
        boots[i] = cliffs_delta(xb, yb)
    lo, hi = np.percentile(boots, [100*(alpha/2), 100*(1-alpha/2)])
    return d0, float(lo), float(hi)

# -----------------------------
# Metrics
# -----------------------------
def dice_jaccard(gt_arr, pr_arr):
    vg = float((gt_arr > 0).sum())
    vp = float((pr_arr > 0).sum())
    if vg == 0 and vp == 0:
        return 1.0, 1.0, True, True
    if vg == 0 or vp == 0:
        return 0.0, 0.0, (vg == 0), (vp == 0)
    inter = float(((gt_arr > 0) & (pr_arr > 0)).sum())
    dice = 2.0 * inter / (vg + vp)
    jac = inter / (vg + vp - inter + 1e-12)
    return dice, jac, False, False

def hd95_mm(gt_img, pr_img):
    gt = sitk.GetArrayFromImage(gt_img)
    pr = sitk.GetArrayFromImage(pr_img)
    if gt.sum() == 0 and pr.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pr.sum() == 0:
        return np.nan
    f = sitk.HausdorffDistanceImageFilter()
    try:
        f.Execute(gt_img, pr_img)
        return float(f.GetHausdorffDistancePercentile(95.0))
    except Exception:
        try:
            return float(f.GetHausdorffDistance())
        except Exception:
            return np.nan

def assd_mm(gt_img, pr_img):
    gt_arr = sitk.GetArrayFromImage(gt_img)
    pr_arr = sitk.GetArrayFromImage(pr_img)
    if gt_arr.sum() == 0 and pr_arr.sum() == 0:
        return 0.0
    if gt_arr.sum() == 0 or pr_arr.sum() == 0:
        return np.nan
    gt_surf = sitk.LabelContour(gt_img)
    pr_surf = sitk.LabelContour(pr_img)
    gt_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.BinaryNot(gt_img), insideIsPositive=False,
                                                  squaredDistance=False, useImageSpacing=True))
    pr_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.BinaryNot(pr_img), insideIsPositive=False,
                                                  squaredDistance=False, useImageSpacing=True))
    gt2pr = sitk.GetArrayFromImage(pr_dm)[sitk.GetArrayFromImage(gt_surf) > 0]
    pr2gt = sitk.GetArrayFromImage(gt_dm)[sitk.GetArrayFromImage(pr_surf) > 0]
    vals = []
    if gt2pr.size > 0:
        vals.append(float(gt2pr.mean()))
    if pr2gt.size > 0:
        vals.append(float(pr2gt.mean()))
    return float(np.mean(vals)) if vals else np.nan

def as_binary(img, label):
    return sitk.BinaryThreshold(img, lowerThreshold=label, upperThreshold=label,
                                insideValue=1, outsideValue=0)

def volume_ml(img):
    arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing()
    return float(arr.sum() * (sx*sy*sz) / 1000.0)

# -----------------------------
# Pred handling
# -----------------------------
def _is_prob_map(img: sitk.Image):
    is_vec = img.GetNumberOfComponentsPerPixel() > 1
    size = img.GetSize()
    is_4d_tail = (len(size) == 4 and size[3] in (5, 4))
    return is_vec or is_4d_tail

def _argmax_labelmap_from_prob(img: sitk.Image):
    if img.GetNumberOfComponentsPerPixel() > 1:
        comps = [sitk.VectorIndexSelectionCast(img, i) for i in range(img.GetNumberOfComponentsPerPixel())]
        arrs = [sitk.GetArrayFromImage(c) for c in comps]
        C = len(arrs)
        stack = np.stack(arrs, axis=0)
        if C == 5:
            lab_idx = np.argmax(stack[1:], axis=0) + 1
        else:
            lab_idx = np.argmax(stack, axis=0)
        out = sitk.GetImageFromArray(lab_idx.astype(np.int16))
        out.CopyInformation(img)
        return out
    else:
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim != 4:
            raise RuntimeError("Unexpected prob map shape.")
        C = arr.shape[0]
        if C == 5:
            lab_idx = np.argmax(arr[1:, ...], axis=0) + 1
        else:
            lab_idx = np.argmax(arr, axis=0)
        out = sitk.GetImageFromArray(lab_idx.astype(np.int16))
        out.CopyInformation(img)
        return out

def _threshold_labelmap_from_prob(img: sitk.Image, thresh=0.5):
    if img.GetNumberOfComponentsPerPixel() > 1:
        comps = [sitk.VectorIndexSelectionCast(img, i) for i in range(img.GetNumberOfComponentsPerPixel())]
        arrs = [sitk.GetArrayFromImage(c) for c in comps]
        C = len(arrs)
        start = 1 if C == 5 else 0
        out = np.zeros_like(arrs[0], dtype=np.int16)
        for lab in range(start, C):
            out[arrs[lab] >= thresh] = lab
        img_out = sitk.GetImageFromArray(out)
        img_out.CopyInformation(img)
        return img_out
    else:
        arr = sitk.GetArrayFromImage(img)
        C = arr.shape[0]
        start = 1 if C == 5 else 0
        out = np.zeros_like(arr[0], dtype=np.int16)
        for lab in range(start, C):
            out[arr[lab] >= thresh] = lab
        img_out = sitk.GetImageFromArray(out)
        img_out.CopyInformation(img)
        return img_out

def _clamp_labels(img: sitk.Image, allowed: set):
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    mask = ~np.isin(arr, list(allowed))
    if mask.any():
        arr[mask] = 0
        out = sitk.GetImageFromArray(arr.astype(np.int16))
        out.CopyInformation(img)
        return out
    return img

# -----------------------------
# Visualization
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _auto_slice_with_all_labels(arr3d, labels):
    for i in range(arr3d.shape[0]):
        uniq = np.unique(arr3d[i])
        if all(l in uniq for l in labels):
            return i
    return None

def _center_of_mass_slice(arr3d):
    idx = np.argwhere(arr3d > 0)
    if idx.size == 0:
        return arr3d.shape[0] // 2
    z_mean = int(np.round(idx[:,0].mean()))
    return int(np.clip(z_mean, 0, arr3d.shape[0]-1))

def _overlay_png(t2, pred, gt, out_png, subject_id, label_names, overlay_alpha=0.22):
    cmap = {
        1: np.array([1, 0, 0, overlay_alpha]),
        2: np.array([0, 1, 0, overlay_alpha]),
        3: np.array([0, 0, 1, overlay_alpha]),
        4: np.array([1, 1, 0, overlay_alpha]),
    }
    sl = _auto_slice_with_all_labels(gt, list(cmap.keys()))
    if sl is None:
        sl = _center_of_mass_slice(gt)

    h, w = pred.shape[1], pred.shape[2]
    pred_overlay = np.zeros((h, w, 4), np.float32)
    gt_overlay   = np.zeros((h, w, 4), np.float32)

    for lab, color in cmap.items():
        pred_overlay[pred[sl] == lab] = color
        gt_overlay[gt[sl] == lab] = color

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(t2[sl], cmap='gray')
    axes[0].imshow(pred_overlay, interpolation='none')
    axes[0].axis("off")

    axes[1].imshow(t2[sl], cmap='gray')
    axes[1].imshow(gt_overlay, interpolation='none')
    axes[1].axis("off")

    axes[0].set_title(f"{subject_id} — Predicted")
    axes[1].set_title(f"{subject_id} — Ground Truth")
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def _box_or_violin(
    ax,
    data_by_model,
    title,
    ylabel,
    violin=False,
    logy=False,
    add_jitter=True,
    jitter_width=0.08,
    jitter_alpha=0.55,
    jitter_size=18,
    seed=1337,
):
    models = list(data_by_model.keys())
    data = [np.asarray([v for v in data_by_model[m] if np.isfinite(v)], float) for m in models]
    pos = np.arange(1, len(models)+1)

    if violin:
        ax.violinplot(data, showmeans=True, showextrema=False, positions=pos)
    else:
        ax.boxplot(data, positions=pos, showfliers=False)

    if add_jitter:
        rng = np.random.default_rng(seed)
        for x0, arr in zip(pos, data):
            if arr.size == 0:
                continue
            xj = x0 + rng.uniform(-jitter_width, jitter_width, size=arr.size)
            ax.scatter(
                xj, arr,
                s=jitter_size,
                alpha=jitter_alpha,
                edgecolors="none",
                zorder=3,
            )

    ax.set_xticks(pos)
    ax.set_xticklabels(models, rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

def _bland_altman(ax, gt, pr, title, units="mL"):
    gt = np.asarray([v for v in gt if np.isfinite(v)], float)
    pr = np.asarray([v for v in pr if np.isfinite(v)], float)
    if gt.size == 0 or pr.size == 0:
        ax.set_title(title + " (no data)")
        return
    avg = 0.5*(gt+pr)
    diff = pr-gt
    md = diff.mean()
    sd = diff.std(ddof=1) if diff.size > 1 else 0.0
    lo, hi = md - 1.96*sd, md + 1.96*sd
    ax.scatter(avg, diff, s=10, alpha=0.6)
    ax.axhline(md, ls='--')
    ax.axhline(lo, ls=':')
    ax.axhline(hi, ls=':')
    ax.set_xlabel(f"Mean of GT and Pred ({units})")
    ax.set_ylabel(f"Pred - GT ({units})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# -----------------------------
# Dataset-aware helpers
# -----------------------------
def dataset_display_name(ds_cfg):
    return ds_cfg.get("display_name", ds_cfg["name"].upper())

def resolve_model_pred_dir(model_cfg, ds_cfg):
    return model_cfg["pred_dir"] + ds_cfg.get("pred_suffix", "")

def resolve_split_dirs(raw_dataset_dir, raw_split):
    images_dir = os.path.join(raw_dataset_dir, f"images{raw_split}")
    labels_dir = os.path.join(raw_dataset_dir, f"labels{raw_split}")
    return images_dir, labels_dir

def find_case_image(images_dir, case_id):
    candidates = [
        os.path.join(images_dir, f"{case_id}_0000.nii.gz"),
        os.path.join(images_dir, f"{case_id}_0000.nii"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    wildcard = glob.glob(os.path.join(images_dir, f"{case_id}_*.nii.gz")) + glob.glob(os.path.join(images_dir, f"{case_id}_*.nii"))
    wildcard = sorted(wildcard)
    return wildcard[0] if wildcard else None

def resolve_dataset_labels(ds_cfg, global_args):
    labels = list(ds_cfg.get("labels", global_args["labels"]))
    label_names = {int(k): v for k, v in ds_cfg.get("label_names", global_args["label_names"]).items()}
    return labels, label_names

def resolve_dataset_subjects(ds_cfg, global_args):
    subjects_cfg = ds_cfg.get("subjects", None)
    if subjects_cfg is None:
        subjects_cfg = global_args.get("subjects", None)

    mode = ds_cfg.get("source_mode", "nnunet_raw")

    if subjects_cfg is not None:
        if isinstance(subjects_cfg, list):
            return list(subjects_cfg)
        if isinstance(subjects_cfg, str) and os.path.exists(subjects_cfg):
            with open(subjects_cfg, "r") as f:
                return [ln.strip() for ln in f if ln.strip()]
        raise SystemExit(f"Invalid subjects spec for dataset '{ds_cfg['name']}'")

    if mode == "legacy_orig_dir":
        orig_dir = ds_cfg["orig_dir"]
        if not os.path.isdir(orig_dir):
            return []
        return sorted([os.path.basename(p) for p in glob.glob(os.path.join(orig_dir, "*")) if os.path.isdir(p)])

    if mode == "nnunet_raw":
        raw_dataset_dir = ds_cfg["raw_dataset_dir"]
        _, labels_dir = resolve_split_dirs(raw_dataset_dir, global_args.get("raw_split", "Ts"))
        if not os.path.isdir(labels_dir):
            return []
        return sorted([
            re.sub(r"\.nii(\.gz)?$", "", os.path.basename(p))
            for p in (glob.glob(os.path.join(labels_dir, "*.nii")) + glob.glob(os.path.join(labels_dir, "*.nii.gz")))
        ])

    raise ValueError(f"Unknown source_mode: {mode}")

def write_compiled_csv(path, rows, preferred_first_cols=None):
    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return

    preferred_first_cols = preferred_first_cols or []
    all_keys = []
    seen = set()

    for col in preferred_first_cols:
        if col not in seen:
            all_keys.append(col)
            seen.add(col)

    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    print(f"[OK] Wrote compiled CSV: {path}")

# -----------------------------
# Core per-dataset analysis
# -----------------------------
def run_one_dataset(global_args, ds_cfg):
    ds_name = ds_cfg["name"]
    ds_title = dataset_display_name(ds_cfg)
    source_mode = ds_cfg.get("source_mode", "nnunet_raw")
    labels, label_names = resolve_dataset_labels(ds_cfg, global_args)
    allowed = set([0] + labels)

    out_root = ensure_dir(os.path.join(global_args["output_dir"], ds_name))
    fig_dir = ensure_dir(os.path.join(out_root, "figures"))
    ovl_dir = ensure_dir(os.path.join(out_root, "overlays"))
    tex_dir = ensure_dir(os.path.join(out_root, "latex"))

    subjects = resolve_dataset_subjects(ds_cfg, global_args)
    if not subjects:
        print(f"[SKIP DATASET] {ds_name}: no subjects discovered.")
        return None

    remap = {int(k): int(v) for k, v in (global_args.get("label_map") or {}).items()}
    gt_cache, t2_cache = {}, {}

    print(f"\n{'='*80}")
    print(f"[DATASET]   {ds_name}")
    print(f"[TITLE]     {ds_title}")
    print(f"[MODE]      {source_mode}")
    if source_mode == "legacy_orig_dir":
        print(f"[ORIG DIR]  {ds_cfg['orig_dir']}")
    else:
        images_dir, labels_dir = resolve_split_dirs(ds_cfg["raw_dataset_dir"], global_args.get("raw_split", "Ts"))
        print(f"[RAW ROOT]  {ds_cfg['raw_dataset_dir']}")
        print(f"[IMAGES]    {images_dir}")
        print(f"[LABELS]    {labels_dir}")
    print(f"[SUBJECTS]   {len(subjects)}")
    print(f"[LABELS]     {labels}")
    print(f"[OUT]        {out_root}")
    print(f"{'='*80}")

    def load_gt(subj):
        if subj in gt_cache:
            return gt_cache[subj]

        if source_mode == "legacy_orig_dir":
            orig_dir = ds_cfg["orig_dir"]
            patterns = global_args.get("gt_name_patterns") or [
                "{id}_seg.nii.gz",
                "{id}_seq.nii.gz",
                "{id}_label.nii.gz",
                "{id}_labels.nii.gz",
            ]
            candidates = [os.path.join(orig_dir, subj, pat.format(id=subj)) for pat in patterns]
            found = [p for p in candidates if os.path.exists(p)]
            if not found:
                wild = glob.glob(os.path.join(orig_dir, subj, "*.nii.gz"))
                def rank(path):
                    base = os.path.basename(path).lower()
                    if "_seg" in base:
                        return 0
                    if "_seq" in base:
                        return 1
                    if "label" in base:
                        return 2
                    return 9
                found = sorted(
                    [p for p in wild if any(k in os.path.basename(p).lower() for k in ["_seg", "_seq", "label"])],
                    key=rank
                )
            if not found:
                print(f"[MISS GT][{ds_name}] {subj}")
                return None
            if len(found) > 1:
                print(f"[GT MULTI][{ds_name}] {subj} -> using {os.path.basename(found[0])}")
            img = sitk.ReadImage(found[0])

        elif source_mode == "nnunet_raw":
            _, labels_dir = resolve_split_dirs(ds_cfg["raw_dataset_dir"], global_args.get("raw_split", "Ts"))
            candidates = [
                os.path.join(labels_dir, f"{subj}.nii.gz"),
                os.path.join(labels_dir, f"{subj}.nii"),
            ]
            gt_path = next((p for p in candidates if os.path.exists(p)), None)
            if gt_path is None:
                print(f"[MISS GT][{ds_name}] {subj}")
                return None
            img = sitk.ReadImage(gt_path)

        else:
            raise ValueError(f"Unknown source_mode: {source_mode}")

        img = _clamp_labels(img, allowed)
        gt_cache[subj] = img
        return img

    def load_t2(subj):
        if subj in t2_cache:
            return t2_cache[subj]

        if source_mode == "legacy_orig_dir":
            p = os.path.join(ds_cfg["orig_dir"], subj, f"{subj}_t2.nii.gz")
            if os.path.exists(p):
                t2_cache[subj] = sitk.ReadImage(p)
                return t2_cache[subj]
            return None

        elif source_mode == "nnunet_raw":
            images_dir, _ = resolve_split_dirs(ds_cfg["raw_dataset_dir"], global_args.get("raw_split", "Ts"))
            img_path = find_case_image(images_dir, subj)
            if img_path is None:
                return None
            t2_cache[subj] = sitk.ReadImage(img_path)
            return t2_cache[subj]

        raise ValueError(f"Unknown source_mode: {source_mode}")

    def load_pred(file_path, pred_type_choice, proba_thresh):
        img = sitk.ReadImage(file_path)

        def remap_if_needed(lbl_img):
            if not remap:
                return lbl_img
            arr = sitk.GetArrayFromImage(lbl_img).astype(np.int32)
            for k, v in remap.items():
                arr[arr == k] = v
            out = sitk.GetImageFromArray(arr.astype(np.int16))
            out.CopyInformation(lbl_img)
            return out

        ptype = pred_type_choice
        if ptype == "auto":
            ptype = "prob-argmax" if _is_prob_map(img) else "labelmap"

        if ptype == "labelmap":
            pred = img
        elif ptype == "prob-argmax":
            pred = _argmax_labelmap_from_prob(img)
        elif ptype == "prob-thresh":
            pred = _threshold_labelmap_from_prob(img, thresh=proba_thresh)
        else:
            raise ValueError("Unknown pred_type.")

        pred = remap_if_needed(pred)
        pred = _clamp_labels(pred, allowed)
        return pred

    # prediction index
    models_all = [m["name"] for m in global_args["models"]]
    pred_index = {}
    active_models = []

    for m in global_args["models"]:
        mname = m["name"]
        mdir = resolve_model_pred_dir(m, ds_cfg)
        pred_index[mname] = {}

        if not os.path.isdir(mdir):
            print(f"[WARN][{ds_name}] Missing pred dir for {mname}: {mdir}")
            continue

        files = glob.glob(os.path.join(mdir, "*.nii")) + glob.glob(os.path.join(mdir, "*.nii.gz"))
        for fp in files:
            sid = re.sub(r"\.nii(\.gz)?$", "", os.path.basename(fp))
            pred_index[mname][sid] = fp

        if not pred_index[mname]:
            print(f"[WARN][{ds_name}] No predictions in: {mdir}")
            continue

        active_models.append(mname)
        print(f"[PRED][{ds_name}] {mname}: {len(pred_index[mname])}")

    if not active_models:
        print(f"[SKIP DATASET] {ds_name}: no model prediction outputs found.")
        return None

    detailed_csv = os.path.join(out_root, "per_subject_detailed.csv")
    per_subject_rows = []
    fields = ["dataset","model","subject","label",
              "present_case","gt_empty","pred_empty",
              "dice","jaccard","hd95_mm","assd_mm",
              "vol_gt_ml","vol_pred_ml","abs_vol_diff_ml","rel_vol_diff_pct"]

    with open(detailed_csv, "w", newline="") as fdet:
        wr = csv.DictWriter(fdet, fieldnames=fields)
        wr.writeheader()

        for m in global_args["models"]:
            mname = m["name"]
            if mname not in active_models:
                continue
            pred_type_choice = global_args["pred_type_global"] or m.get("pred_type", "auto")
            proba_thresh = global_args["proba_thresh"]
            mis_cnt = 0

            for sid in subjects:
                gt_img = load_gt(sid)
                if gt_img is None:
                    continue
                pred_path = pred_index[mname].get(sid, None)
                if not pred_path:
                    mis_cnt += 1
                    continue

                pr_img = load_pred(pred_path, pred_type_choice, proba_thresh)
                t2_img = load_t2(sid) if (global_args["make_overlays"] or global_args["save_case_qc"]) else None

                for lab in labels:
                    gt_bin = as_binary(gt_img, lab)
                    pr_bin = as_binary(pr_img, lab)
                    d, j, gt_empty, pr_empty = dice_jaccard(
                        sitk.GetArrayFromImage(gt_bin),
                        sitk.GetArrayFromImage(pr_bin)
                    )
                    hd = hd95_mm(gt_bin, pr_bin)
                    asd = assd_mm(gt_bin, pr_bin)
                    vg = volume_ml(gt_bin)
                    vp = volume_ml(pr_bin)
                    abs_vd = abs(vp - vg)
                    rel_vd = (abs_vd / vg * 100.0) if vg > 0 else (0.0 if vp == 0 else np.nan)

                    row = {
                        "dataset": ds_name,
                        "model": mname,
                        "subject": sid,
                        "label": lab,
                        "present_case": int(vg > 0),
                        "gt_empty": int(vg == 0),
                        "pred_empty": int(vp == 0),
                        "dice": float(d),
                        "jaccard": float(j),
                        "hd95_mm": float(hd) if np.isfinite(hd) else np.nan,
                        "assd_mm": float(asd) if np.isfinite(asd) else np.nan,
                        "vol_gt_ml": float(vg),
                        "vol_pred_ml": float(vp),
                        "abs_vol_diff_ml": float(abs_vd),
                        "rel_vol_diff_pct": float(rel_vd) if np.isfinite(rel_vd) else np.nan,
                    }
                    wr.writerow(row)
                    per_subject_rows.append(row)

                if (global_args["make_overlays"] or global_args["save_case_qc"]) and t2_img is not None:
                    t2 = sitk.GetArrayFromImage(t2_img)
                    pr = sitk.GetArrayFromImage(pr_img)
                    gt = sitk.GetArrayFromImage(gt_img)
                    out_png = os.path.join(ovl_dir, f"{sid}_{mname}.png")
                    _overlay_png(
                        t2, pr, gt, out_png, sid, label_names,
                        overlay_alpha=global_args.get("overlay_alpha", 0.22)
                    )

            if mis_cnt:
                print(f"[MISS PRED][{ds_name}] {mname}: {mis_cnt} missing subjects")

    print(f"[OK][{ds_name}] Wrote detailed per-case metrics: {detailed_csv}")

    def present_vals(model, label, key):
        return [r[key] for r in per_subject_rows
                if r["model"] == model and r["label"] == label and r["present_case"] == 1 and np.isfinite(r[key])]

    def macro_vals(model, key):
        by_subj = defaultdict(list)
        for r in per_subject_rows:
            if r["model"] == model and r["present_case"] == 1 and np.isfinite(r[key]):
                by_subj[r["subject"]].append(float(r[key]))
        return [float(np.mean(v)) for v in by_subj.values() if v]

    summary_label_csv = os.path.join(out_root, "summary_per_label.csv")
    summary_overall_csv = os.path.join(out_root, "summary_overall.csv")
    metrics = ["dice","jaccard","hd95_mm","assd_mm","abs_vol_diff_ml","rel_vol_diff_pct"]

    with open(summary_label_csv, "w", newline="") as fsum:
        hdr = ["dataset","model","label","label_name","n_present"] + \
              sum(([f"{m}_mean", f"{m}_std", f"{m}_median", f"{m}_iqr25", f"{m}_iqr75"] for m in metrics), [])
        wr = csv.DictWriter(fsum, fieldnames=hdr)
        wr.writeheader()
        for mname in active_models:
            for lab in labels:
                vals = {met: np.asarray(present_vals(mname, lab, met), float) for met in metrics}
                row = {
                    "dataset": ds_name,
                    "model": mname,
                    "label": lab,
                    "label_name": label_names.get(lab, f"Label {lab}"),
                    "n_present": int(len(vals["dice"]))
                }
                for met, arr in vals.items():
                    if arr.size == 0:
                        row[f"{met}_mean"] = row[f"{met}_std"] = row[f"{met}_median"] = row[f"{met}_iqr25"] = row[f"{met}_iqr75"] = ""
                    else:
                        q25, q75 = np.percentile(arr, [25, 75])
                        row[f"{met}_mean"]   = float(arr.mean())
                        row[f"{met}_std"]    = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                        row[f"{met}_median"] = float(np.median(arr))
                        row[f"{met}_iqr25"]  = float(q25)
                        row[f"{met}_iqr75"]  = float(q75)
                wr.writerow(row)
    print(f"[OK][{ds_name}] Wrote per-label summaries: {summary_label_csv}")

    with open(summary_overall_csv, "w", newline="") as fsum2:
        hdr = ["dataset","model","n_subjects"] + \
              sum(([f"{m}_macro_mean", f"{m}_macro_std", f"{m}_macro_median", f"{m}_macro_iqr25", f"{m}_macro_iqr75"] for m in metrics), [])
        wr = csv.DictWriter(fsum2, fieldnames=hdr)
        wr.writeheader()
        for mname in active_models:
            macro = {met: np.asarray(macro_vals(mname, met), float) for met in metrics}
            row = {"dataset": ds_name, "model": mname, "n_subjects": int(len(macro["dice"]))}
            for met, arr in macro.items():
                if arr.size == 0:
                    row[f"{met}_macro_mean"] = row[f"{met}_macro_std"] = row[f"{met}_macro_median"] = row[f"{met}_macro_iqr25"] = row[f"{met}_macro_iqr75"] = ""
                else:
                    q25, q75 = np.percentile(arr, [25, 75])
                    row[f"{met}_macro_mean"]   = float(arr.mean())
                    row[f"{met}_macro_std"]    = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                    row[f"{met}_macro_median"] = float(np.median(arr))
                    row[f"{met}_macro_iqr25"]  = float(q25)
                    row[f"{met}_macro_iqr75"]  = float(q75)
            wr.writerow(row)
    print(f"[OK][{ds_name}] Wrote macro summaries: {summary_overall_csv}")

    comp_metric = global_args["compare_metric"]
    pair_csv = os.path.join(out_root, "pairwise_by_label.csv")
    pair_overall_csv = os.path.join(out_root, "pairwise_overall.csv")

    def subject_metric(model, label=None):
        rows = [r for r in per_subject_rows if r["model"] == model]
        if label is not None:
            rows = [r for r in rows if r["label"] == label and r["present_case"] == 1]
        d = defaultdict(list)
        for r in rows:
            v = r.get(comp_metric, float("nan"))
            if np.isfinite(v):
                d[r["subject"]].append(float(v))
        return {k: float(np.mean(v)) for k, v in d.items() if v}

    with open(pair_csv, "w", newline="") as fcmp:
        hdr = ["dataset","label","model_a","model_b","n_pairs",
               f"{comp_metric}_mean_a", f"{comp_metric}_mean_b",
               "mean_diff_a_minus_b","boot_lo","boot_hi","bca_lo","bca_hi",
               "p_boot_two_sided","cliffs_delta","cliffs_lo","cliffs_hi"]
        wr = csv.DictWriter(fcmp, fieldnames=hdr)
        wr.writeheader()

        for lab in labels:
            per_model = {m: subject_metric(m, lab) for m in active_models}
            for i in range(len(active_models)):
                for j in range(i+1, len(active_models)):
                    ma, mb = active_models[i], active_models[j]
                    sids = sorted(set(per_model[ma]).intersection(per_model[mb]))
                    if len(sids) < 5:
                        continue
                    a = np.array([per_model[ma][s] for s in sids], float)
                    b = np.array([per_model[mb][s] for s in sids], float)
                    diff_obs, (plo, phi), boots = paired_bootstrap(a, b, n_boot=CONFIG["n_boot"], seed=CONFIG["seed"])
                    bca_lo, bca_hi = paired_bca_ci(a, b, n_boot=CONFIG["n_boot"], alpha=0.05, seed=CONFIG["seed"])
                    p_two = float(2.0 * min((boots <= 0).mean(), (boots >= 0).mean()))
                    delta, cd_lo, cd_hi = cliffs_delta_boot_ci(a, b, n_boot=CONFIG["n_boot"], seed=CONFIG["seed"])
                    wr.writerow({
                        "dataset": ds_name,
                        "label": lab,
                        "model_a": ma,
                        "model_b": mb,
                        "n_pairs": len(sids),
                        f"{comp_metric}_mean_a": float(a.mean()),
                        f"{comp_metric}_mean_b": float(b.mean()),
                        "mean_diff_a_minus_b": float(diff_obs),
                        "boot_lo": float(plo),
                        "boot_hi": float(phi),
                        "bca_lo": float(bca_lo),
                        "bca_hi": float(bca_hi),
                        "p_boot_two_sided": float(p_two),
                        "cliffs_delta": float(delta),
                        "cliffs_lo": float(cd_lo),
                        "cliffs_hi": float(cd_hi),
                    })
    print(f"[OK][{ds_name}] Wrote pairwise comparisons by label: {pair_csv}")

    with open(pair_overall_csv, "w", newline="") as fcmp2:
        hdr = ["dataset","model_a","model_b","n_pairs",
               f"{comp_metric}_macro_mean_a", f"{comp_metric}_macro_mean_b",
               "mean_diff_a_minus_b","boot_lo","boot_hi","bca_lo","bca_hi",
               "p_boot_two_sided","cliffs_delta","cliffs_lo","cliffs_hi"]
        wr = csv.DictWriter(fcmp2, fieldnames=hdr)
        wr.writeheader()

        per_model_macro = {}
        for m in active_models:
            per_subj = defaultdict(list)
            for r in per_subject_rows:
                if r["model"] == m and r["present_case"] == 1 and np.isfinite(r[comp_metric]):
                    per_subj[r["subject"]].append(float(r[comp_metric]))
            per_model_macro[m] = {sid: float(np.mean(vals)) for sid, vals in per_subj.items() if vals}

        for i in range(len(active_models)):
            for j in range(i+1, len(active_models)):
                ma, mb = active_models[i], active_models[j]
                sids = sorted(set(per_model_macro[ma]).intersection(per_model_macro[mb]))
                if len(sids) < 5:
                    continue
                a = np.array([per_model_macro[ma][s] for s in sids], float)
                b = np.array([per_model_macro[mb][s] for s in sids], float)
                diff_obs, (plo, phi), boots = paired_bootstrap(a, b, n_boot=CONFIG["n_boot"], seed=CONFIG["seed"])
                bca_lo, bca_hi = paired_bca_ci(a, b, n_boot=CONFIG["n_boot"], alpha=0.05, seed=CONFIG["seed"])
                p_two = float(2.0 * min((boots <= 0).mean(), (boots >= 0).mean()))
                delta, cd_lo, cd_hi = cliffs_delta_boot_ci(a, b, n_boot=CONFIG["n_boot"], seed=CONFIG["seed"])
                wr.writerow({
                    "dataset": ds_name,
                    "model_a": ma,
                    "model_b": mb,
                    "n_pairs": len(sids),
                    f"{comp_metric}_macro_mean_a": float(a.mean()),
                    f"{comp_metric}_macro_mean_b": float(b.mean()),
                    "mean_diff_a_minus_b": float(diff_obs),
                    "boot_lo": float(plo),
                    "boot_hi": float(phi),
                    "bca_lo": float(bca_lo),
                    "bca_hi": float(bca_hi),
                    "p_boot_two_sided": float(p_two),
                    "cliffs_delta": float(delta),
                    "cliffs_lo": float(cd_lo),
                    "cliffs_hi": float(cd_hi),
                })
    print(f"[OK][{ds_name}] Wrote overall (macro) pairwise comparisons: {pair_overall_csv}")

    for lab in labels:
        data = {m: [v for v in present_vals(m, lab, "dice")] for m in active_models}
        if not any(len(v) for v in data.values()):
            continue
        fig, ax = plt.subplots(figsize=(8,5))
        _box_or_violin(
            ax,
            data,
            f"{ds_title} — Dice by Model — Label {lab} ({label_names.get(lab,'')})",
            "Dice",
            violin=True,
            logy=False,
            add_jitter=global_args.get("jitter_points", True),
            jitter_width=global_args.get("jitter_width", 0.08),
            jitter_alpha=global_args.get("jitter_alpha", 0.55),
            jitter_size=global_args.get("jitter_size", 18),
            seed=global_args.get("seed", 1337) + int(lab),
        )
        of = os.path.join(fig_dir, f"dice_label{lab}_violin.png")
        plt.savefig(of, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"[FIG][{ds_name}] {of}")

    for lab in labels:
        data = {m: [v for v in present_vals(m, lab, "hd95_mm")] for m in active_models}
        if not any(len(v) for v in data.values()):
            continue
        fig, ax = plt.subplots(figsize=(8,5))
        _box_or_violin(
            ax,
            data,
            f"{ds_title} — HD95 (mm) by Model — Label {lab}",
            "HD95 (mm)",
            violin=False,
            logy=True,
            add_jitter=global_args.get("jitter_points", True),
            jitter_width=global_args.get("jitter_width", 0.08),
            jitter_alpha=global_args.get("jitter_alpha", 0.55),
            jitter_size=global_args.get("jitter_size", 18),
            seed=global_args.get("seed", 1337) + 100 + int(lab),
        )
        of = os.path.join(fig_dir, f"hd95_label{lab}_box.png")
        plt.savefig(of, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"[FIG][{ds_name}] {of}")

    lab_ba = 3 if 3 in labels else labels[0]
    for m in active_models:
        gt_vals, pr_vals = [], []
        for r in per_subject_rows:
            if r["model"] == m and r["label"] == lab_ba and r["present_case"] == 1:
                if np.isfinite(r["vol_gt_ml"]) and np.isfinite(r["vol_pred_ml"]):
                    gt_vals.append(r["vol_gt_ml"])
                    pr_vals.append(r["vol_pred_ml"])
        fig, ax = plt.subplots(figsize=(7,5))
        _bland_altman(
            ax, gt_vals, pr_vals,
            f"{ds_title} — Bland–Altman (Label {lab_ba} = {label_names.get(lab_ba,'')}) — {m}",
            "mL"
        )
        of = os.path.join(fig_dir, f"bland_altman_label{lab_ba}_{m}.png")
        plt.savefig(of, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"[FIG][{ds_name}] {of}")

    tex1 = os.path.join(tex_dir, "table_dice_per_label.tex")
    with open(tex1, "w") as ft:
        ft.write("\\begin{tabular}{l" + "c"*len(labels) + "}\n\\toprule\n")
        ft.write("Model " + " & " + " & ".join([f"Label {lab}" for lab in labels]) + " \\\\\n\\midrule\n")
        for m in active_models:
            cells = []
            for lab in labels:
                vals = np.asarray([v for v in present_vals(m, lab, "dice")], float)
                if vals.size < 3:
                    cells.append("--")
                else:
                    mean = vals.mean()
                    lo, hi = paired_bca_ci(vals, np.zeros_like(vals), n_boot=5000, alpha=0.05, seed=123)
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            ft.write(m + " & " + " & ".join(cells) + " \\\\\n")
        ft.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[TEX][{ds_name}] {tex1}")

    tex2 = os.path.join(tex_dir, "table_pairwise_overall.tex")
    with open(pair_overall_csv, "r") as f:
        rows_po = list(csv.DictReader(f))
    with open(tex2, "w") as ft:
        ft.write("\\begin{tabular}{l l c c c c}\n\\toprule\n")
        ft.write("Model A & Model B & $n$ & $\\Delta$ (A$-$B) & BCa 95\\% CI & $p_{boot}$ \\\\\n\\midrule\n")
        for r in rows_po:
            n = int(float(r["n_pairs"])) if r["n_pairs"] != "" else 0
            if n < 5:
                continue
            d = float(r["mean_diff_a_minus_b"])
            lo = float(r["bca_lo"])
            hi = float(r["bca_hi"])
            p = float(r["p_boot_two_sided"])
            ft.write(f"{r['model_a']} & {r['model_b']} & {n} & {d:.3f} & [{lo:.3f}, {hi:.3f}] & {p:.3f} \\\\\n")
        ft.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[TEX][{ds_name}] {tex2}")

    manifest = {
        "dataset": ds_name,
        "display_name": ds_title,
        "source_mode": source_mode,
        "dataset_config": ds_cfg,
        "labels": labels,
        "label_names": label_names,
        "subjects_total": len(subjects),
        "active_models": active_models,
        "per_model_pred_counts": {m: len(pred_index.get(m, {})) for m in models_all},
        "metrics": metrics,
        "compare_metric": comp_metric,
        "visualization": {
            "overlay_alpha": global_args.get("overlay_alpha", 0.22),
            "jitter_points": global_args.get("jitter_points", True),
            "jitter_width": global_args.get("jitter_width", 0.08),
            "jitter_alpha": global_args.get("jitter_alpha", 0.55),
            "jitter_size": global_args.get("jitter_size", 18),
        },
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DATASET DONE] {ds_name}\n"
          f"- Detailed per-case: {detailed_csv}\n"
          f"- Per-label summary: {summary_label_csv}\n"
          f"- Macro summary:     {summary_overall_csv}\n"
          f"- Pairwise by label: {pair_csv}\n"
          f"- Pairwise overall:  {pair_overall_csv}\n"
          f"- Figures:           {fig_dir}\n"
          f"- LaTeX:             {tex_dir}\n"
          f"- Manifest:          {os.path.join(out_root, 'manifest.json')}\n")

    with open(summary_label_csv, "r", newline="") as f:
        summary_label_rows = list(csv.DictReader(f))
    with open(summary_overall_csv, "r", newline="") as f:
        summary_overall_rows = list(csv.DictReader(f))
    with open(pair_csv, "r", newline="") as f:
        pair_label_rows = list(csv.DictReader(f))
    with open(pair_overall_csv, "r", newline="") as f:
        pair_overall_rows = list(csv.DictReader(f))

    return {
        "dataset": ds_name,
        "per_subject_rows": per_subject_rows,
        "summary_label_rows": summary_label_rows,
        "summary_overall_rows": summary_overall_rows,
        "pair_label_rows": pair_label_rows,
        "pair_overall_rows": pair_overall_rows,
    }

def main():
    args = CONFIG
    ensure_dir(args["output_dir"])

    datasets = args.get("datasets", None)
    if not datasets:
        raise SystemExit("CONFIG['datasets'] must be defined.")

    all_per_subject_rows = []
    all_summary_label_rows = []
    all_summary_overall_rows = []
    all_pair_label_rows = []
    all_pair_overall_rows = []

    for ds_cfg in datasets:
        result = run_one_dataset(args, ds_cfg)
        if result is None:
            continue
        all_per_subject_rows.extend(result.get("per_subject_rows", []))
        all_summary_label_rows.extend(result.get("summary_label_rows", []))
        all_summary_overall_rows.extend(result.get("summary_overall_rows", []))
        all_pair_label_rows.extend(result.get("pair_label_rows", []))
        all_pair_overall_rows.extend(result.get("pair_overall_rows", []))

    write_compiled_csv(
        os.path.join(args["output_dir"], "all_per_subject_detailed.csv"),
        all_per_subject_rows,
        preferred_first_cols=[
            "dataset", "model", "subject", "label",
            "present_case", "gt_empty", "pred_empty",
            "dice", "jaccard", "hd95_mm", "assd_mm",
            "vol_gt_ml", "vol_pred_ml", "abs_vol_diff_ml", "rel_vol_diff_pct"
        ],
    )

    write_compiled_csv(
        os.path.join(args["output_dir"], "all_summary_per_label.csv"),
        all_summary_label_rows,
        preferred_first_cols=["dataset", "model", "label", "label_name", "n_present"],
    )

    write_compiled_csv(
        os.path.join(args["output_dir"], "all_summary_overall.csv"),
        all_summary_overall_rows,
        preferred_first_cols=["dataset", "model", "n_subjects"],
    )

    write_compiled_csv(
        os.path.join(args["output_dir"], "all_pairwise_by_label.csv"),
        all_pair_label_rows,
        preferred_first_cols=["dataset", "label", "model_a", "model_b", "n_pairs"],
    )

    write_compiled_csv(
        os.path.join(args["output_dir"], "all_pairwise_overall.csv"),
        all_pair_overall_rows,
        preferred_first_cols=["dataset", "model_a", "model_b", "n_pairs"],
    )

    print("\nAll done.")

if __name__ == "__main__":
    main()