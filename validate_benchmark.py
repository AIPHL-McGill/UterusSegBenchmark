#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public-facing multi-dataset validator for uterine MRI segmentation benchmarks.

The script computes per-case metrics and paired statistics, then exports
publication-ready qualitative, Dice, and HD95 figures as PNG, TIFF, PDF, and SVG.
Edit the CONFIG block below to point to the datasets, prediction folders, and
output directory. Both legacy folder layouts and nnU-Net raw dataset layouts
are supported. Ensemble analysis is intentionally excluded.
"""

import os, re, csv, json, glob, math
from statistics import NormalDist
import hashlib, sys, argparse, zipfile
import copy
from itertools import product
from collections import defaultdict, Counter
import numpy as np
import SimpleITK as sitk
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# CONFIG — EDIT THESE
# -----------------------------
CONFIG = {
    "output_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UM_nnUnet/results_revision_20261011_v3",
    "previous_output_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UM_nnUnet/results_cmig_publication",
    # Preserve the original evaluator's same-index comparison for saved masks.
    # Arrays are left unchanged; physical distances/volumes use the reference grid.
    "geometry_policy": "same_index",  # same_index | resample | strict
    # Audit identified this reference's placeholder 1-mm header. Verify the
    # source MRI against every native prediction before correcting it in memory.
    "reference_geometry_from_mri": {"umd": ["UMD_221129_090"]},

    "datasets": [
        {
            "name": "umd",
            "display_name": "UMD Internal Testing",
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
            "labels": [1],
            "label_names": {1: "Endometrial cancer tumor"},
        },
        {
            "name": "lms",
            "display_name": "LMS",
            "source_mode": "nnunet_raw",
            "raw_dataset_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_raw_data/Dataset102_LMS",
            "pred_suffix": "_lms",
            "subjects": None,
            "labels": [1],
            "label_names": {1: "Uterine mass"},
        },
    ],

    "models": [
        {"name": "UNet3D",     "pred_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_unet3d",    "pred_type": "auto"},
        {"name": "Swin-UNETR", "pred_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_swinunetr", "pred_type": "auto"},
        {"name": "MedNeXt",    "pred_dir": "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_mednext",   "pred_type": "auto"},

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
    "save_case_qc": False,
    "compare_metric": "dice",
    "n_boot": 5000,
    "seed": 1337,

    # Publication visualization tuning
    "overlay_alpha": 0.24,
    "jitter_points": True,
    "jitter_width": 0.075,
    "jitter_alpha": 0.62,
    "jitter_size": 12,
    "figure_dpi": 600,
    "save_tiff": True,
    "save_svg": True,
    "save_pdf": True,
    "make_publication_figures": True,
    "make_case_qc_overlays": False,
    "qualitative_cases": {
        "umd": "UMD_221129_193",
        "umd_external": "MS2",
        "emca": "OS025",
        "lms": "TOAT22",
    },
    "qualitative_selection_fallback": "representative",  # representative | high | disagreement
    "qualitative_crop_padding": 18,
    "make_external_examples": True,
}
# -----------------------------


# -----------------------------
# Math helpers
# -----------------------------
def _z(u: float) -> float:
    """Standard-normal inverse CDF, including correctly signed extreme tails."""
    return NormalDist().inv_cdf(float(np.clip(u, 1e-12, 1 - 1e-12)))

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
    if np.all(d == d[0]):
        return theta_hat, theta_hat
    prop = float(np.clip((boots < theta_hat).mean(), 1e-12, 1-1e-12))
    z0 = _z(prop)
    s = d.sum()
    jk = np.empty(n, float)
    for i in range(n):
        jk[i] = float((s - d[i]) / (n - 1))
    jk_bar = float(jk.mean())
    num = float(np.sum((jk_bar - jk)**3))
    den = float(6.0 * np.sum((jk_bar - jk)**2) ** 1.5)
    a_hat = num / den if den > 0 else 0.0
    def bca_alpha(q):
        z = _z(q); adj = z0 + z; denom = 1 - a_hat * adj
        z_bca = z0 + adj / (denom + 1e-18)
        return float(np.clip(0.5*(1+math.erf(z_bca/math.sqrt(2))), 1e-12, 1-1e-12))
    lo = float(np.quantile(boots, bca_alpha(alpha/2)))
    hi = float(np.quantile(boots, bca_alpha(1 - alpha/2)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        lo, hi = np.percentile(boots, [100*(alpha/2), 100*(1-alpha/2)])
    return lo, hi



def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_evaluation_cases(dataset):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_cases.json")
    with open(path) as handle:
        manifest = json.load(handle)
    expected = {"umd": 30, "umd_external": 12, "emca": 43, "lms": 41}
    if manifest.get("expected_counts") != expected or set(manifest.get("datasets", {})) != set(expected):
        raise ValueError("Evaluation case manifest does not describe the four original test sets.")
    subjects = manifest["datasets"][dataset]
    if len(subjects) != expected[dataset] or len(set(subjects)) != len(subjects):
        raise ValueError(f"Wrong or duplicated test-case identifiers for {dataset}.")
    return list(subjects), file_sha256(path)


def assert_same_geometry(reference, prediction, context=""):
    if reference.GetDimension() != 3 or prediction.GetDimension() != 3:
        raise ValueError(f"Expected scalar 3D images for {context}.")
    if reference.GetSize() != prediction.GetSize():
        raise ValueError(f"Native shape mismatch for {context}.")
    for name, reference_values, prediction_values in [
        ("spacing", reference.GetSpacing(), prediction.GetSpacing()),
        ("origin", reference.GetOrigin(), prediction.GetOrigin()),
        ("direction", reference.GetDirection(), prediction.GetDirection()),
    ]:
        if not np.allclose(reference_values, prediction_values, rtol=0, atol=1e-5):
            raise ValueError(f"Native {name} mismatch for {context}.")


def image_geometry(image):
    return {"size": list(image.GetSize()), "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def correct_reference_geometry(reference, source_mri, predictions, context=""):
    """Validate an independently corroborated MRI grid, preserving mask voxels.

    Header agreement supports the physical grid, not anatomical correctness of
    the annotation. The original corresponding-voxel assumption is retained.
    """
    images = {"source MRI": source_mri, **predictions}
    if reference.GetDimension() != 3 or source_mri.GetDimension() != 3:
        raise ValueError(f"Expected 3D reference and source MRI for {context}.")
    if reference.GetSize() != source_mri.GetSize():
        raise ValueError(f"Source MRI/reference sizes differ for {context}.")
    if not predictions:
        raise ValueError(f"No native predictions to corroborate the MRI grid for {context}.")
    spacing = np.asarray(source_mri.GetSpacing())
    origin = np.asarray(source_mri.GetOrigin())
    direction = np.asarray(source_mri.GetDirection()).reshape(3, 3)
    corners = np.asarray(list(product(*[(0, n - 1) for n in source_mri.GetSize()])), float)
    mri_points = origin + (corners * spacing) @ direction.T
    discrepancies = {}
    for name, image in images.items():
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"Expected scalar 3D {name} for {context}.")
        if image.GetSize() != source_mri.GetSize():
            raise ValueError(f"Native {name}/MRI sizes differ for {context}.")
        grid_spacing = np.asarray(image.GetSpacing())
        grid_origin = np.asarray(image.GetOrigin())
        grid_direction = np.asarray(image.GetDirection()).reshape(3, 3)
        if (not np.all(np.isfinite(grid_spacing)) or np.any(grid_spacing <= 0)
                or not np.all(np.isfinite(grid_origin)) or not np.all(np.isfinite(grid_direction))
                or not np.allclose(grid_direction.T @ grid_direction, np.eye(3), rtol=0, atol=1e-4)):
            raise ValueError(f"Invalid physical grid for {context}/{name}.")
        points = grid_origin + (corners * grid_spacing) @ grid_direction.T
        displacement = float(np.linalg.norm(points - mri_points, axis=1).max())
        if (not np.allclose(grid_spacing, spacing, rtol=1e-6, atol=1e-5)
                or not np.allclose(grid_origin, origin, rtol=0, atol=1e-3)
                or not np.allclose(grid_direction, direction, rtol=0, atol=5e-5)
                or displacement > 0.1):
            raise ValueError(f"Source MRI and {name} grids disagree for {context}; "
                             f"maximum corner displacement={displacement:.6g} mm. "
                             "No reference-header correction was applied.")
        discrepancies[name] = {"geometry": image_geometry(image),
                               "maximum_corner_displacement_mm": displacement}
    def is_placeholder(image):
        # Identity NIfTI RAS affine is diag(-1,-1,1) after SimpleITK's LPS
        # conversion. The audited case090 has this exact direction; synthetic
        # SimpleITK masks can instead have native identity direction.
        default_direction = any(np.allclose(image.GetDirection(), candidate.ravel(), rtol=0, atol=1e-8)
                                for candidate in [np.eye(3), np.diag([-1, -1, 1])])
        return (np.allclose(image.GetSpacing(), (1, 1, 1), rtol=0, atol=1e-8)
                and np.allclose(image.GetOrigin(), (0, 0, 0), rtol=0, atol=1e-8)
                and default_direction)
    placeholder = is_placeholder(reference)
    if not placeholder:
        # The targeted fix must not silently replace a different real grid.
        assert_same_geometry(source_mri, reference, context + "/non-placeholder reference")
    if placeholder and is_placeholder(source_mri):
        raise ValueError(f"The MRI also has the placeholder grid for {context}; no corroborated correction.")
    before = sitk.GetArrayFromImage(reference)
    corrected = sitk.Image(reference)
    corrected.CopyInformation(source_mri)
    after = sitk.GetArrayFromImage(corrected)
    if not np.array_equal(before, after):
        raise RuntimeError("Reference voxel values changed during header correction.")
    event = {"context": context, "policy": "reference_header_from_corroborated_source_MRI",
             "original_reference_grid": image_geometry(reference),
             "corrected_reference_grid": image_geometry(corrected),
             "corroborating_images": discrepancies, "array_values_preserved": True,
             "reference_array_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
             "anatomical_alignment_note": "Original same-index correspondence retained; header agreement alone does not certify annotation alignment."}
    return corrected, event


def align_to_reference_grid(reference, image, args, context="", is_mask=True):
    """Restore historical index-based comparison and log header normalization.

    same_index assumes that the saved arrays represent corresponding voxels, as
    in the original evaluator. Equal shapes do not independently establish this
    correspondence. No mask voxels are moved, interpolated or changed under this
    policy; reference spacing supplies consistent physical units.
    """
    if reference.GetDimension() != 3 or image.GetDimension() != 3:
        raise ValueError(f"Expected scalar 3D images for {context}.")
    reference_geometry, image_grid = image_geometry(reference), image_geometry(image)
    if reference_geometry == image_grid:
        return image
    policy = args.get("geometry_policy", "same_index")
    if policy == "strict":
        assert_same_geometry(reference, image, context)
        aligned = sitk.Image(image)
        aligned.CopyInformation(reference)
    elif policy == "same_index":
        if reference.GetSize() != image.GetSize():
            raise ValueError(f"Array sizes differ for {context}: reference={reference.GetSize()}, image={image.GetSize()}. "
                             "The original voxel-wise comparison also requires equal sizes. Check the export grid; "
                             "use geometry_policy='resample' only for images sharing physical coordinates.")
        aligned = sitk.Image(image)
        aligned.CopyInformation(reference)
    elif policy == "resample":
        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        aligned = sitk.Resample(image, reference, sitk.Transform(3, sitk.sitkIdentity),
                                interpolator, 0.0, image.GetPixelID())
    else:
        raise ValueError(f"Unknown geometry policy: {policy}")
    events = args.setdefault("_geometry_events", [])
    if not any(event["context"] == context for event in events):
        events.append({"context": context, "policy": policy, "is_mask": is_mask,
                       "reference_grid": reference_geometry, "original_image_grid": image_grid,
                       "array_values_preserved": policy != "resample"})
        print(f"[GRID][{context}] {policy}: spacing {tuple(image_grid['spacing'])} -> {reference.GetSpacing()}")
        output = ensure_dir(args["output_dir"])
        with open(os.path.join(output, "geometry_adjustments.json"), "w") as handle:
            json.dump(events, handle, indent=2)
    return aligned

def annotate_pairwise_csv(path, n_boot, adjust_holm=False):
    """Keep raw empirical tail probabilities and label Monte Carlo resolution.

    The historical rule 2*min(P_boot(delta<=0), P_boot(delta>=0)) is unchanged.
    BCa intervals use a different transformation, so interval exclusion and
    this empirical percentile-tail rule need not agree. Holm is applied to
    full-precision raw probabilities. Zero empirical counts are reported as
    below resolution, never as evidence that a population probability is zero.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return
    additional = ["p_boot_report", "bootstrap_tail_resolution"]
    if adjust_holm:
        additional += ["p_holm", "p_holm_report", "holm_family_size"]
    fields += [field for field in additional if field not in fields]
    raw = np.asarray([float(row["p_boot_two_sided"]) for row in rows])
    if np.any(raw < 0) or np.any(raw > 1) or not np.all(np.isfinite(raw)):
        raise ValueError("Invalid empirical bootstrap probability.")
    resolution = 2.0 / n_boot
    adjusted = raw.copy()
    if adjust_holm:
        order = np.argsort(raw, kind="stable")
        adjusted[order] = np.minimum(1.0, np.maximum.accumulate(raw[order] * np.arange(len(raw), 0, -1)))
    for i, row in enumerate(rows):
        row["bootstrap_tail_resolution"] = resolution
        row["p_boot_report"] = f"<{resolution:.4g} (Monte Carlo resolution)" if raw[i] == 0 else f"{raw[i]:.6g}"
        if adjust_holm:
            row["p_holm"] = float(adjusted[i])
            row["holm_family_size"] = len(rows)
            report_resolution = min(1.0, len(rows) * resolution)
            row["p_holm_report"] = f"<{report_resolution:.4g} (Monte Carlo resolution)" if adjusted[i] == 0 else f"{adjusted[i]:.6g}"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

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
    """Symmetric 95th-percentile surface Hausdorff distance in millimetres.

    SimpleITK's HausdorffDistanceImageFilter exposes the maximum Hausdorff
    distance, not HD95. Compute both directed surface-distance distributions
    explicitly and return the larger directed 95th percentile.
    """
    gt = sitk.GetArrayFromImage(gt_img)
    pr = sitk.GetArrayFromImage(pr_img)
    if gt.sum() == 0 and pr.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pr.sum() == 0:
        return np.nan

    try:
        gt_surf = sitk.LabelContour(gt_img)
        pr_surf = sitk.LabelContour(pr_img)
        gt_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(
            gt_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
        ))
        pr_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(
            pr_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
        ))

        gt_to_pr = sitk.GetArrayFromImage(pr_dm)[sitk.GetArrayFromImage(gt_surf) > 0]
        pr_to_gt = sitk.GetArrayFromImage(gt_dm)[sitk.GetArrayFromImage(pr_surf) > 0]
        if gt_to_pr.size == 0 or pr_to_gt.size == 0:
            return np.nan
        return float(max(np.percentile(gt_to_pr, 95), np.percentile(pr_to_gt, 95)))
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
    gt_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img, insideIsPositive=False,
                                                  squaredDistance=False, useImageSpacing=True))
    pr_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(pr_img, insideIsPositive=False,
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
    is_4d_tail = (len(size) == 4 and size[3] >= 2)
    return is_vec or is_4d_tail

def _argmax_labelmap_from_prob(img: sitk.Image):
    if img.GetNumberOfComponentsPerPixel() > 1:
        comps = [sitk.VectorIndexSelectionCast(img, i) for i in range(img.GetNumberOfComponentsPerPixel())]
        arrs = [sitk.GetArrayFromImage(c) for c in comps]
        C = len(arrs)
        stack = np.stack(arrs, axis=0)
        lab_idx = np.argmax(stack, axis=0)
        out = sitk.GetImageFromArray(lab_idx.astype(np.int16))
        out.CopyInformation(img)
        return out
    else:
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim != 4:
            raise RuntimeError("Unexpected prob map shape.")
        C = arr.shape[0]
        lab_idx = np.argmax(arr, axis=0)
        out = sitk.GetImageFromArray(lab_idx.astype(np.int16))
        spatial = sitk.Extract(img, list(img.GetSize()[:3]) + [0], [0, 0, 0, 0])
        out.CopyInformation(spatial)
        return out

def _threshold_labelmap_from_prob(img: sitk.Image, thresh=0.5):
    if img.GetNumberOfComponentsPerPixel() > 1:
        comps = [sitk.VectorIndexSelectionCast(img, i) for i in range(img.GetNumberOfComponentsPerPixel())]
        arrs = [sitk.GetArrayFromImage(c) for c in comps]
        C = len(arrs)
        start = 1  # channel 0 is background
        out = np.zeros_like(arrs[0], dtype=np.int16)
        for lab in range(start, C):
            out[arrs[lab] >= thresh] = lab
        img_out = sitk.GetImageFromArray(out)
        img_out.CopyInformation(img)
        return img_out
    else:
        arr = sitk.GetArrayFromImage(img)
        C = arr.shape[0]
        start = 1  # channel 0 is background
        out = np.zeros_like(arr[0], dtype=np.int16)
        for lab in range(start, C):
            out[arr[lab] >= thresh] = lab
        img_out = sitk.GetImageFromArray(out)
        spatial = sitk.Extract(img, list(img.GetSize()[:3]) + [0], [0, 0, 0, 0])
        img_out.CopyInformation(spatial)
        return img_out

def _clamp_labels(img: sitk.Image, allowed: set):
    raw = sitk.GetArrayFromImage(img)
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
        raise ValueError("Label map has nonfinite or noninteger values; specify its prediction type explicitly.")
    arr = raw.astype(np.int32)
    mask = ~np.isin(arr, list(allowed))
    if mask.any():
        raise ValueError(f"Unexpected labels {np.unique(arr[mask]).tolist()}; expected {sorted(allowed)}. Check dataset label mapping.")
    return img


# -----------------------------
# Publication visualization
# -----------------------------
MODEL_ORDER = ["UNet3D", "Swin-UNETR", "nnU-Net", "MedNeXt"]
MODEL_LABELS = {
    "UNet3D": "U-Net",
    "Swin-UNETR": "Swin-\nUNETR",
    "nnU-Net": "nnU-Net",
    "MedNeXt": "MedNeXt",
}
MODEL_LABELS_COMPACT = {
    "UNet3D": "U-Net",
    "Swin-UNETR": "Swin",
    "nnU-Net": "nnU",
    "MedNeXt": "MedN",
}
MODEL_COLORS = {
    "UNet3D": "#4C78A8",
    "Swin-UNETR": "#F28E2B",
    "nnU-Net": "#59A14F",
    "MedNeXt": "#E15759",
}
LABEL_COLORS = {
    1: "#E15759",  # wall or binary target
    2: "#59A14F",  # cavity
    3: "#4C78A8",  # myoma
    4: "#F2CF5B",  # Nabothian cyst
}
LABEL_SHORT = {
    1: "Wall",
    2: "Cavity",
    3: "Myomas",
    4: "Nabothian cyst",
}
DATASET_SHORT = {
    "umd": "Internal UMD",
    "umd_external": "External UMD-style",
    "emca": "Endometrial cancer",
    "lms": "Uterine mass",
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _set_publication_style():
    """Use legible text at a two-column journal width and embedded vector fonts."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def _ordered_models(models):
    available = set(models)
    return [m for m in MODEL_ORDER if m in available] + sorted(available.difference(MODEL_ORDER))


def _save_figure(fig, stem, config, facecolor=None):
    """Save high-resolution raster files and editable PDF/SVG vector exports."""
    dpi = int(config.get("figure_dpi", 600))
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.06}
    if facecolor is not None:
        kwargs["facecolor"] = facecolor
    fig.savefig(stem + ".png", dpi=dpi, **kwargs)
    if config.get("save_tiff", True):
        fig.savefig(stem + ".tiff", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"}, **kwargs)
    if config.get("save_pdf", True):
        fig.savefig(stem + ".pdf", **kwargs)
    if config.get("save_svg", True):
        fig.savefig(stem + ".svg", **kwargs)


def _model_label(model):
    # Preserve the complete name in narrow subpanels; no unexplained abbreviations.
    return {"UNet3D": "U-Net", "Swin-UNETR": "Swin-\nUNETR",
            "nnU-Net": "nnU-Net", "MedNeXt": "MedNeXt"}.get(model, model)


def _clean_axis(ax, ylabel=None, title=None, logy=False, dice=False):
    if title:
        ax.set_title(title, pad=8, fontweight="semibold")
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)
    if logy:
        # A pure log axis silently hides valid zero distances. Preserve 0--1 mm.
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.75, base=10)
        upper = max(1.3, float(ax.get_ylim()[1]))
        ticks = [t for t in [0, 0.5, 1, 10, 100, 1000] if t <= upper]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_ylim(bottom=0)
    if dice:
        ax.set_ylim(-0.025, 1.025)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", pad=2)


def _distribution_plot(
    ax, data_by_model, metric, title="", ylabel=None, seed=1337,
    add_jitter=True, jitter_width=0.075, jitter_alpha=0.62, jitter_size=12,
    compact_labels=False, denominators=None,
):
    """Plot observed finite values, medians, and explicit scoring denominators.

    denominators may map a model to {'n_present': N, 'n_missed': M}, or to N.
    A missing surface result is never placed at zero or imputed into the plot.
    """
    models = _ordered_models(data_by_model)
    data = [np.asarray([v for v in data_by_model[m] if np.isfinite(v)], float) for m in models]
    positions = np.arange(1, len(models) + 1)
    max_value = max((float(a.max()) for a in data if a.size), default=1.0)

    for x0, arr, model in zip(positions, data, models):
        color = MODEL_COLORS.get(model, "#777777")
        if metric == "dice":
            # KDE is undefined for empty/singleton/constant data; show actual data.
            if arr.size >= 2 and np.ptp(arr) > 1e-12:
                violin = ax.violinplot([arr], positions=[x0], widths=0.72,
                                       showmeans=False, showmedians=False, showextrema=False)
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.22)
                    body.set_linewidth(0.8)
            if arr.size:
                q25, med, q75 = np.percentile(arr, [25, 50, 75])
                ax.vlines(x0, q25, q75, color=color, linewidth=2.2, zorder=4)
                ax.hlines(med, x0 - 0.20, x0 + 0.20, color=color, linewidth=1.8, zorder=5)
        elif arr.size:
            bp = ax.boxplot([arr], positions=[x0], widths=0.58, showfliers=False,
                            patch_artist=True, medianprops={"linewidth": 1.6, "color": color},
                            whiskerprops={"linewidth": 0.9, "color": color},
                            capprops={"linewidth": 0.9, "color": color})
            bp["boxes"][0].set_facecolor(mpl.colors.to_rgba(color, 0.12))
            bp["boxes"][0].set_edgecolor(color)
            bp["boxes"][0].set_linewidth(1.0)

    if add_jitter:
        rng = np.random.default_rng(seed)
        for x0, arr, model in zip(positions, data, models):
            if arr.size:
                xj = x0 + rng.uniform(-jitter_width, jitter_width, size=arr.size)
                ax.scatter(xj, arr, s=jitter_size, alpha=jitter_alpha,
                           color=MODEL_COLORS.get(model, "#777777"), edgecolors="none", zorder=3)
    elif any(a.size for a in data):
        # Singletons/constant Dice still need a visible observation without jitter.
        for x0, arr, model in zip(positions, data, models):
            if arr.size and (arr.size == 1 or np.ptp(arr) <= 1e-12):
                ax.scatter([x0], [arr[0]], s=jitter_size,
                           color=MODEL_COLORS.get(model, "#777777"), zorder=4)

    ax.set_xlim(0.45, len(models) + 0.55)
    if metric == "hd95_mm":
        ax.set_ylim(0, max(1.4, max_value * 1.3))
    ax.set_xticks(positions)
    ax.set_xticklabels([_model_label(m).replace("\n", "") if compact_labels else _model_label(m) for m in models],
                       rotation=55 if compact_labels else 0,
                       ha="right" if compact_labels else "center",
                       rotation_mode="anchor")
    _clean_axis(ax, ylabel=ylabel, title=title, logy=(metric == "hd95_mm"), dice=(metric == "dice"))

    for x0, arr, model in zip(positions, data, models):
        denom = (denominators or {}).get(model)
        if isinstance(denom, dict):
            n_present = int(denom.get("n_present", arr.size))
            n_missed = int(denom.get("n_missed", 0))
        else:
            n_present = int(denom) if denom is not None else int(arr.size)
            n_missed = None
        if metric == "hd95_mm" and denom is not None:
            annotation = f"{arr.size}\n/{n_present}" if compact_labels else f"{arr.size}/{n_present}"
            if n_missed is not None:
                # Compact subpanels have four annotations in approximately one
                # inch. Omit the wide equals sign there to keep miss counts
                # visibly separated without reducing the publication font.
                annotation += f"\nM{n_missed}" if compact_labels else f"\nM={n_missed}"
        else:
            annotation = f"n\n{arr.size}" if compact_labels else f"n={arr.size}"
        ax.text(x0, 1.025, annotation, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.8 if compact_labels else 8.2,
                linespacing=1.10, color="#333333", clip_on=False)
    if title:
        count_lines = 3 if (compact_labels and metric == "hd95_mm") else 2 if (compact_labels or metric == "hd95_mm") else 1
        ax.set_title(title, pad=10 + count_lines * 9, fontweight="semibold")
    if not any(a.size for a in data):
        ax.text(0.5, 0.48, "No defined values", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5, color="#777777")


def _panel_label(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left")


def _auto_slice_with_all_labels(arr3d, labels):
    labels = [int(l) for l in labels]
    best = None
    best_score = -1
    for i in range(arr3d.shape[0]):
        sl = arr3d[i]
        present = sum(bool(np.any(sl == lab)) for lab in labels)
        area = int(np.count_nonzero(sl))
        score = present * (sl.size + 1) + area
        if score > best_score:
            best_score = score
            best = i
    return int(best if best is not None else arr3d.shape[0] // 2)


def _center_of_mass_slice(arr3d):
    idx = np.argwhere(arr3d > 0)
    if idx.size == 0:
        return arr3d.shape[0] // 2
    return int(np.clip(np.round(idx[:, 0].mean()), 0, arr3d.shape[0] - 1))


def _robust_gray(img2d):
    arr = np.asarray(img2d, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _crop_bounds(mask2d, padding=18):
    yy, xx = np.where(mask2d)
    h, w = mask2d.shape
    if yy.size == 0:
        return 0, h, 0, w
    y0 = max(0, int(yy.min()) - padding)
    y1 = min(h, int(yy.max()) + padding + 1)
    x0 = max(0, int(xx.min()) - padding)
    x1 = min(w, int(xx.max()) + padding + 1)
    # Crop in pixels only; physical aspect is applied separately at rendering.
    ch, cw = y1 - y0, x1 - x0
    side = max(ch, cw)
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    y0 = max(0, int(round(cy - side / 2)))
    y1 = min(h, y0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, int(round(cx - side / 2)))
    x1 = min(w, x0 + side)
    x0 = max(0, x1 - side)
    return y0, y1, x0, x1


def _draw_segmentation(ax, image2d, seg2d, labels, alpha=0.24, spacing=None):
    """Draw a common crop with physical pixel aspect, reference label colors."""
    spacing = spacing if spacing is not None else (1.0, 1.0)
    sx, sy = float(spacing[0]), float(spacing[1])
    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
        raise ValueError("Figure spacing must contain finite positive x/y values.")
    h, w = image2d.shape
    extent = (-0.5 * sx, (w - 0.5) * sx, (h - 0.5) * sy, -0.5 * sy)
    ax.imshow(_robust_gray(image2d), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest", extent=extent, aspect="equal")
    for lab in labels:
        mask = np.asarray(seg2d == int(lab), dtype=float)
        if not np.any(mask):
            continue
        color = LABEL_COLORS.get(int(lab), "#E15759")
        rgba = np.zeros(mask.shape + (4,), dtype=float)
        rgba[..., :3] = mpl.colors.to_rgb(color)
        rgba[..., 3] = mask * alpha
        ax.imshow(rgba, interpolation="nearest", extent=extent, aspect="equal")
        if np.any(mask == 0):
            ax.contour(np.arange(w) * sx, np.arange(h) * sy, mask,
                       levels=[0.5], colors=[color], linewidths=0.8, alpha=0.95)
    ax.set_xlim(-0.5 * sx, (w - 0.5) * sx)
    ax.set_ylim((h - 0.5) * sy, -0.5 * sy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#6F6F6F")
        spine.set_linewidth(0.5)


def _overlay_png(t2, pred, gt, out_png, subject_id, label_names, overlay_alpha=0.24, spacing=None):
    """Optional QC view; these images are not quantitative evidence."""
    labels = sorted(int(k) for k in label_names)
    sl = _auto_slice_with_all_labels(gt, labels)
    union = (gt[sl] > 0) | (pred[sl] > 0)
    y0, y1, x0, x1 = _crop_bounds(union, padding=18)
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.9), facecolor="black")
    for ax, seg, title in zip(axes, [pred, gt], ["Prediction", "Reference"]):
        ax.set_facecolor("black")
        _draw_segmentation(ax, t2[sl, y0:y1, x0:x1], seg[sl, y0:y1, x0:x1],
                           labels, alpha=overlay_alpha, spacing=spacing)
        ax.set_title(title, color="white", fontsize=10, pad=4)
    fig.suptitle(f"{subject_id} | slice {sl} (0-based)", color="white", fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.03)
    fig.savefig(out_png, dpi=300, facecolor="black", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _bland_altman(ax, gt, pr, title, units="mL"):
    gt = np.asarray(gt, float)
    pr = np.asarray(pr, float)
    if gt.shape != pr.shape:
        raise ValueError("Bland-Altman reference/prediction arrays must be paired.")
    valid = np.isfinite(gt) & np.isfinite(pr)
    gt, pr = gt[valid], pr[valid]
    if gt.size == 0:
        ax.set_title(title + " (no data)")
        return
    avg, diff = 0.5 * (gt + pr), pr - gt
    md = diff.mean()
    sd = diff.std(ddof=1) if diff.size > 1 else 0.0
    lo, hi = md - 1.96 * sd, md + 1.96 * sd
    ax.scatter(avg, diff, s=12, alpha=0.6)
    ax.axhline(md, ls="--", color="#333333")
    ax.axhline(lo, ls=":", color="#777777")
    ax.axhline(hi, ls=":", color="#777777")
    ax.set_xlabel(f"Mean of reference and prediction ({units})")
    ax.set_ylabel(f"Prediction − reference ({units})")
    _clean_axis(ax, title=title)


def _qualitative_header(payload):
    dataset = DATASET_SHORT.get(payload['dataset'], payload['dataset'])
    target = {"emca": "endometrial tumor", "lms": "uterine mass"}.get(payload['dataset'], "uterine structures")
    selection = str(payload.get("selection_reason", "configured")).replace("_", " ")
    header = f"{dataset}: {target} | {payload['subject']} | slice {payload.get('slice_index', '?')} (0-based)"
    details = f"Selection: {selection}"
    score = payload.get("mean_dice_across_models")
    if score is not None and np.isfinite(float(score)):
        details += f"; mean Dice across models={float(score):.3f}"
    if "spacing" not in payload:
        details += "; pixel aspect (spacing unavailable)"
    return header, details


def _render_case_grid(payloads, stem, config, title=None):
    """Create one or multiple qualitative rows with traceable example selection."""
    if not payloads:
        return
    common_models = set.intersection(*(set(p['preds']) for p in payloads))
    models = _ordered_models(common_models)
    columns = models + ["Reference"]
    nrows = len(payloads)
    fig = plt.figure(figsize=(7.2, nrows * 2.20 + 0.65), facecolor="black")
    top = 0.94 if title else 0.985
    grid = fig.add_gridspec(nrows * 2, len(columns),
                           height_ratios=sum(([0.26, 1.0] for _ in payloads), []),
                           left=0.025, right=0.995, bottom=0.075 if nrows > 2 else 0.14 if nrows == 2 else 0.16,
                           top=top, hspace=0.25, wspace=0.035)
    for row, payload in enumerate(payloads):
        header_ax = fig.add_subplot(grid[row * 2, :])
        header_ax.axis("off")
        header, details = _qualitative_header(payload)
        header_ax.text(0, 0.88, f"{chr(ord('a') + row)}  {header}", color="white",
                       fontsize=9.0, va="top", weight="semibold", transform=header_ax.transAxes)
        header_ax.text(0, 0.23, details, color="#DEDEDE", fontsize=8.2,
                       va="top", transform=header_ax.transAxes)
        for col, column in enumerate(columns):
            ax = fig.add_subplot(grid[row * 2 + 1, col])
            ax.set_facecolor("black")
            seg = payload['gt'] if column == "Reference" else payload['preds'][column]
            _draw_segmentation(ax, payload['image'], seg, payload['labels'],
                               alpha=float(config.get('overlay_alpha', 0.24)),
                               spacing=payload.get('spacing'))
            ax.set_title("Reference" if column == "Reference" else _model_label(column).replace("\n", ""),
                         color="white", fontsize=9, pad=4)
    handles = [Patch(facecolor=LABEL_COLORS[lab], edgecolor="none", label=name) for lab, name in
               [(1, "Wall / binary lesion target"), (2, "Cavity"), (3, "Myomas"), (4, "Nabothian cyst")]
               if any(lab in p['labels'] for p in payloads)]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.012),
                        ncol=2 if len(handles) > 2 else len(handles), frameon=False,
                        fontsize=8.2, handlelength=1.0, columnspacing=1.2)
    for txt in legend.get_texts():
        txt.set_color("white")
    if title:
        fig.suptitle(title, color="white", fontsize=10.5, weight="semibold", y=0.995)
    _save_figure(fig, stem, config, facecolor="black")
    plt.close(fig)


def _render_qualitative_row(payload, stem, config):
    _render_case_grid([payload], stem, config)


def _render_qualitative_composite(payloads, out_dir, config):
    ordered = [payloads[d] for d in ["umd", "umd_external", "emca", "lms"] if d in payloads]
    if not ordered:
        return
    stem = os.path.join(out_dir, "Figure2_qualitative_composite")
    _render_case_grid(ordered, stem, config)
    print(f"[PUBLICATION FIGURE] {stem}.[png|tiff|pdf|svg]")


def _render_external_examples(payloads, stem, config):
    """Compare difficult/successful saved external cases without claiming representativeness."""
    _render_case_grid(payloads, stem, config,
                      title="External examples selected by lowest / highest mean Dice across models")
    print(f"[EXTERNAL EXAMPLES] {stem}.[png|tiff|pdf|svg]")


def _rows_to_data(rows, dataset, label, metric, models):
    out = {}
    for model in models:
        out[model] = [float(r[metric]) for r in rows
                      if r['dataset'] == dataset and r['model'] == model
                      and int(r['label']) == int(label)
                      and int(r.get('present_case', 1)) == 1
                      and np.isfinite(float(r[metric]))]
    return out


def _metric_denominators(rows, dataset, label, models):
    out = {}
    for model in models:
        selected = [r for r in rows if r['dataset'] == dataset and r['model'] == model
                    and int(r['label']) == int(label) and int(r.get('present_case', 1)) == 1]
        out[model] = {'n_present': len(selected),
                      'n_missed': sum(int(float(r.get('pred_empty', 0))) == 1 for r in selected)}
    return out


def _make_composite_metric_figure(all_rows, metric, out_dir, config):
    """Create clearer 2x2 journal-width figures with counts and accurate median labels."""
    models = _ordered_models({r['model'] for r in all_rows})
    fig = plt.figure(figsize=(7.2, 9.80))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.85, wspace=0.33,
                            left=0.095, right=0.99, bottom=0.135, top=0.845)
    available_ds = {r['dataset'] for r in all_rows}
    ds_order = [d for d in ['umd', 'umd_external', 'emca', 'lms'] if d in available_ds]
    group_titles = []
    for panel_idx, ds_name in enumerate(ds_order):
        spec = outer[panel_idx // 2, panel_idx % 2]
        if ds_name in {'umd', 'umd_external'}:
            sub = spec.subgridspec(2, 2, hspace=0.78, wspace=0.40)
            for li, lab in enumerate([1, 2, 3, 4]):
                ax = fig.add_subplot(sub[li // 2, li % 2])
                _distribution_plot(ax, _rows_to_data(all_rows, ds_name, lab, metric, models), metric,
                    title=LABEL_SHORT[lab], ylabel=('Dice' if metric == 'dice' else 'HD95 (mm)') if li % 2 == 0 else None,
                    seed=int(config.get('seed', 1337)) + panel_idx * 100 + lab,
                    add_jitter=config.get('jitter_points', True),
                    jitter_width=config.get('jitter_width', 0.075),
                    jitter_alpha=config.get('jitter_alpha', 0.62),
                    jitter_size=config.get('jitter_size', 12), compact_labels=True,
                    denominators=_metric_denominators(all_rows, ds_name, lab, models))
                if li < 2:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)
        else:
            ax = fig.add_subplot(spec)
            _distribution_plot(ax, _rows_to_data(all_rows, ds_name, 1, metric, models), metric,
                title='',
                ylabel='Dice' if metric == 'dice' else 'HD95 (mm)',
                seed=int(config.get('seed', 1337)) + panel_idx * 100,
                add_jitter=config.get('jitter_points', True),
                jitter_width=config.get('jitter_width', 0.075),
                jitter_alpha=config.get('jitter_alpha', 0.62),
                jitter_size=config.get('jitter_size', 12),
                denominators=_metric_denominators(all_rows, ds_name, 1, models))
        bbox = spec.get_position(fig)
        group_titles.append((bbox.x0, bbox.y1 + (0.093 if panel_idx < 2 else 0.055),
                             f"{chr(ord('a') + panel_idx)}  " + ({"emca": "Endometrial tumor (EMCA)", "lms": "Uterine mass (LMS)"}.get(ds_name, DATASET_SHORT[ds_name]))))
    for x, y, title in group_titles:
        fig.text(x, y, title, fontsize=10.3, fontweight='semibold', va='bottom', ha='left')
    if metric == 'dice':
        note = ('Points: individual cases. Horizontal bars: medians; vertical bars: interquartile ranges.\n'
                'n: reference-positive cases. Reference-negative labels are excluded. No significance is implied.')
    else:
        note = ('Points: individual cases. Boxes: interquartile ranges; center lines: medians.\n'
                'Counts: evaluable / reference-positive cases; M0, M1, ...: empty-prediction misses.\n'
                'Undefined surface results are omitted. Axis: linear from 0 to 1 mm, logarithmic above 1 mm.')
    fig.text(0.095, 0.025, note, fontsize=8.2, va='bottom', ha='left', linespacing=1.45)
    filename = 'Figure3_Dice_composite' if metric == 'dice' else 'Figure4_HD95_composite'
    stem = os.path.join(out_dir, filename)
    _save_figure(fig, stem, config)
    plt.close(fig)
    print(f"[PUBLICATION FIGURE] {stem}.[png|tiff|pdf|svg]")

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
def failure_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"], int(row["label"]))].append(row)
    result = []
    for (dataset, model, label), group in sorted(groups.items()):
        positives = [r for r in group if int(r["present_case"]) == 1]
        result.append({
            "dataset": dataset, "model": model, "label": label, "n_evaluated": len(group),
            "n_reference_positive": len(positives),
            "n_reference_positive_prediction_empty": sum(int(r["pred_empty"]) == 1 for r in positives),
            "n_reference_negative_prediction_positive": sum(int(r["gt_empty"]) == 1 and int(r["pred_empty"]) == 0 for r in group),
            "n_both_empty": sum(int(r["gt_empty"]) == 1 and int(r["pred_empty"]) == 1 for r in group),
            "n_dice_scored": sum(np.isfinite(float(r["dice"])) for r in positives),
            "n_hd95_scored": sum(np.isfinite(float(r["hd95_mm"])) for r in positives),
            "n_assd_scored": sum(np.isfinite(float(r["assd_mm"])) for r in positives),
        })
    return result


def compare_previous_outputs(current_rows, args):
    previous_dir = args.get("previous_output_dir")
    if not previous_dir:
        return []
    previous_csv = os.path.join(previous_dir, "all_per_subject_detailed.csv")
    if not os.path.isfile(previous_csv):
        print(f"[COMPARE] No previous CSV at {previous_csv}; comparison skipped.")
        return []
    with open(previous_csv, newline="") as handle:
        previous_rows = list(csv.DictReader(handle))
    def key(row):
        return row["dataset"], row["model"], row["subject"], int(row["label"])
    previous = {key(r): r for r in previous_rows}
    if len(previous) != len(previous_rows):
        raise ValueError("Previous results have duplicate case/model/label rows; comparison is ambiguous.")
    result = []
    groups = defaultdict(list)
    for row in current_rows:
        if int(row["present_case"]) == 1:
            groups[(row["dataset"], row["model"])].append(row)
    for (dataset, model), rows in sorted(groups.items()):
        for metric in ["dice", "hd95_mm", "assd_mm"]:
            old_values, new_values, subject_values = [], [], defaultdict(list)
            for row in rows:
                old = previous.get(key(row))
                if old is None:
                    continue
                old_value, new_value = float(old[metric]), float(row[metric])
                if np.isfinite(old_value) and np.isfinite(new_value):
                    old_values.append(old_value); new_values.append(new_value)
                    subject_values[row["subject"]].append((old_value, new_value))
            if not old_values:
                continue
            old_values, new_values = np.asarray(old_values), np.asarray(new_values)
            deltas = new_values - old_values
            old_macros = [np.mean([v[0] for v in values]) for values in subject_values.values()]
            new_macros = [np.mean([v[1] for v in values]) for values in subject_values.values()]
            result.append({"dataset": dataset, "model": model, "metric": metric,
                "n_matched_labels": len(deltas), "n_matched_subjects": len(subject_values),
                "old_mean_matched_labels": float(old_values.mean()), "new_mean_matched_labels": float(new_values.mean()),
                "old_macro_mean_matched_labels": float(np.mean(old_macros)), "new_macro_mean_matched_labels": float(np.mean(new_macros)),
                "mean_change": float(deltas.mean()), "mean_absolute_change": float(np.abs(deltas).mean()),
                "maximum_absolute_change": float(np.abs(deltas).max()), "n_changed_at_1e_minus_6": int((np.abs(deltas) > 1e-6).sum()),
                "n_reference_positive_rows_unmatched": len(rows) - len(old_values)})
    write_compiled_csv(os.path.join(args["output_dir"], "comparison_with_previous_metrics.csv"), result)
    old_pairs = os.path.join(previous_dir, "all_pairwise_overall.csv")
    new_pairs = os.path.join(args["output_dir"], "all_pairwise_overall.csv")
    if os.path.isfile(old_pairs) and os.path.isfile(new_pairs):
        with open(old_pairs, newline="") as handle:
            old = list(csv.DictReader(handle))
        with open(new_pairs, newline="") as handle:
            new = list(csv.DictReader(handle))
        lookup = {(r["dataset"], r["model_a"], r["model_b"]): r for r in old}
        differences = []
        for row in new:
            k = row["dataset"], row["model_a"], row["model_b"]
            matched = lookup.get(k)
            sign = 1
            if matched is None:
                matched = lookup.get((k[0], k[2], k[1])); sign = -1
            if matched is None:
                continue
            diff = {"dataset": k[0], "model_a": k[1], "model_b": k[2]}
            for field in ["mean_diff_a_minus_b", "bca_lo", "bca_hi", "p_boot_two_sided"]:
                old_field = ("bca_hi" if field == "bca_lo" else "bca_lo") if sign == -1 and field in {"bca_lo", "bca_hi"} else field
                old_value = float(matched[old_field]) * (sign if field != "p_boot_two_sided" else 1)
                diff["old_" + field] = old_value
                diff["new_" + field] = float(row[field])
                diff["change_" + field] = float(row[field]) - old_value
            differences.append(diff)
        write_compiled_csv(os.path.join(args["output_dir"], "comparison_with_previous_intervals.csv"), differences)
    return result


def write_review_bundle(args):
    root = os.path.abspath(args["output_dir"])
    target = os.path.join(root, "rerun_results_for_review.zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory, _, names in os.walk(root):
            for name in sorted(names):
                path = os.path.join(directory, name)
                rel = os.path.relpath(path, root)
                if name.endswith((".csv", ".json", ".txt")) or ("publication_figures" in rel and name.endswith(".pdf")):
                    archive.write(path, rel)
    print(f"[REVIEW PACKAGE] {target}")


# Only the audited V2 producer is accepted for this specific targeted repair.
AUDITED_V2_SCRIPT_SHA256 = "63b7ef9aa6dfd7534f2d9cecf98a8b4d88028cbd5739ea3156af77f7b595dfb0"
TARGETED_RECOMPUTE = {("umd", "UMD_221129_090")}
CASE_INTEGER_FIELDS = {"label", "present_case", "gt_empty", "pred_empty"}
CASE_FLOAT_FIELDS = {"dice", "jaccard", "hd95_mm", "assd_mm", "vol_gt_ml",
                     "vol_pred_ml", "abs_vol_diff_ml", "rel_vol_diff_pct"}


def prepare_targeted_correction(args, source_dir):
    """Reuse audited V2 metrics; recompute only the malformed-header case."""
    source_dir = os.path.abspath(source_dir)
    if os.path.realpath(source_dir) == os.path.realpath(args["output_dir"]):
        raise ValueError("Targeted correction requires a different output directory.")
    if os.path.isdir(args["output_dir"]) and os.listdir(args["output_dir"]):
        raise ValueError("Choose an empty new output directory for the targeted correction.")
    expected_datasets = {d["name"] for d in CONFIG["datasets"]}
    if {d["name"] for d in args["datasets"]} != expected_datasets:
        raise ValueError("--reuse-results requires all four original datasets.")
    cached, manifests, configs = {}, {}, []
    fields = {"dataset", "model", "subject"} | CASE_INTEGER_FIELDS | CASE_FLOAT_FIELDS
    for original_ds in args["datasets"]:
        dataset = original_ds["name"]
        manifest_path = os.path.join(source_dir, dataset, "manifest.json")
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if manifest["script_sha256"] != AUDITED_V2_SCRIPT_SHA256:
            raise ValueError(f"Cached {dataset} outputs were not produced by the audited V2 evaluator.")
        subjects, cases_hash = locked_evaluation_cases(dataset)
        if manifest["subjects"] != subjects or manifest["evaluation_cases_sha256"] != cases_hash:
            raise ValueError(f"Cached test-case provenance changed for {dataset}.")
        for setting, expected in [("n_boot", args["n_boot"]), ("bootstrap_seed", args["seed"])]:
            if manifest["revision_evaluation"][setting] != expected:
                raise ValueError(f"Cached bootstrap setting {setting} differs for {dataset}.")
        labels = list(manifest["labels"])
        if labels != list(original_ds["labels"]) or set(manifest["active_models"]) != set(m["name"] for m in args["models"]):
            raise ValueError(f"Cached label/model scope differs for {dataset}.")
        path = os.path.join(source_dir, dataset, "per_subject_detailed.csv")
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != fields:
                raise ValueError(f"Unexpected cached metric schema for {dataset}.")
            raw_rows = list(reader)
        indexed = {}
        for raw in raw_rows:
            row = dict(raw)
            for field in CASE_INTEGER_FIELDS:
                row[field] = int(row[field])
            for field in CASE_FLOAT_FIELDS:
                row[field] = float(row[field])
            key = row["model"], row["subject"], row["label"]
            if row["dataset"] != dataset or key in indexed:
                raise ValueError(f"Duplicated or misidentified cached row for {dataset}.")
            indexed[key] = row
        expected = {(m["name"], sid, lab) for m in args["models"] for sid in subjects for lab in labels}
        if set(indexed) != expected:
            raise ValueError(f"Incomplete cached evaluation for {dataset}.")
        cached[dataset] = indexed
        manifests[dataset] = manifest
        configs.append(copy.deepcopy(manifest["dataset_config"]))
    args["datasets"] = configs
    args["_cached_case_rows"] = cached
    args["_cached_manifests"] = manifests
    args["_cached_results_dir"] = source_dir
    args["_recompute_cases"] = TARGETED_RECOMPUTE.copy()
    geometry_path = os.path.join(source_dir, "geometry_adjustments.json")
    with open(geometry_path) as handle:
        prior_geometry = json.load(handle)
    args["_geometry_events"] = [e for e in prior_geometry if "UMD_221129_090" not in e["context"]]
    print("[TARGETED CORRECTION] Reusing all V2 case metrics except UMD_221129_090.")


def verify_targeted_correction(rows, args):
    if "_cached_case_rows" not in args:
        return
    actual = {}
    for row in rows:
        key = row["dataset"], row["model"], row["subject"], int(row["label"])
        if key in actual:
            raise RuntimeError("Targeted output contains duplicate rows.")
        actual[key] = row
    old = {(ds, *key): row for ds, indexed in args["_cached_case_rows"].items() for key, row in indexed.items()}
    if set(actual) != set(old):
        raise RuntimeError("Targeted correction changed the evaluated row set.")
    changes, preserved = [], 0
    for key, row in actual.items():
        previous = old[key]
        is_target = (key[0], key[2]) in args["_recompute_cases"]
        for field in CASE_INTEGER_FIELDS | CASE_FLOAT_FIELDS:
            a, b = float(previous[field]), float(row[field])
            equal = a == b or (np.isnan(a) and np.isnan(b))
            if is_target and field == "rel_vol_diff_pct":
                equal = equal or np.isclose(a, b, rtol=0, atol=1e-10)
            if not equal:
                if not is_target or field not in {"hd95_mm", "assd_mm", "vol_gt_ml", "vol_pred_ml", "abs_vol_diff_ml"}:
                    raise RuntimeError(f"Unexpected cached-result change: {key}/{field}")
                changes.append({"dataset": key[0], "model": key[1], "subject": key[2],
                                "label": key[3], "metric": field, "old": a, "new": b})
        if not is_target:
            preserved += 1
    target_rows = len(rows) - preserved
    report = {"cached_results_dir": args["_cached_results_dir"],
              "cached_detailed_csv_sha256": file_sha256(os.path.join(args["_cached_results_dir"], "all_per_subject_detailed.csv")),
              "total_case_model_label_rows": len(rows), "rows_recomputed": target_rows,
              "rows_preserved_exactly": preserved,
              "dice_jaccard_and_empty_flags_unchanged": True,
              "reference_geometry_corrections": args.get("_reference_geometry_corrections", []),
              "physical_metric_changes": changes}
    with open(os.path.join(args["output_dir"], "targeted_correction_audit.json"), "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    write_compiled_csv(os.path.join(args["output_dir"], "targeted_physical_metric_changes.csv"), changes)
    print(f"[VERIFIED] {target_rows} rows recomputed; {preserved} rows preserved exactly; Dice unchanged.")


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

    subjects, evaluation_cases_hash = locked_evaluation_cases(ds_name)
    if set(m["name"] for m in global_args["models"]) != {"UNet3D", "Swin-UNETR", "MedNeXt", "nnU-Net"}:
        raise ValueError("Exactly the four original evaluated models are required.")
    cached_rows = global_args.get("_cached_case_rows", {}).get(ds_name)
    cached_manifest = global_args.get("_cached_manifests", {}).get(ds_name)
    input_files = (copy.deepcopy(cached_manifest["input_files"]) if cached_manifest else
                   {"reference": {}, "prediction": {m["name"]: {} for m in global_args["models"]}})

    def record_input(group, sid, path, model=None):
        records = input_files[group] if model is None else input_files[group][model]
        digest = file_sha256(path)
        if cached_manifest and digest != records[sid]["sha256"]:
            raise ValueError(f"Input file changed since V2: {ds_name}/{model or group}/{sid}")
        records[sid] = {"path": path, "sha256": digest}

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
                raise FileNotFoundError(f"Missing reference mask: {ds_name}/{subj}")
            if len(found) > 1:
                print(f"[GT MULTI][{ds_name}] {subj} -> using {os.path.basename(found[0])}")
            record_input("reference", subj, found[0])
            img = sitk.ReadImage(found[0])

        elif source_mode == "nnunet_raw":
            _, labels_dir = resolve_split_dirs(ds_cfg["raw_dataset_dir"], global_args.get("raw_split", "Ts"))
            candidates = [
                os.path.join(labels_dir, f"{subj}.nii.gz"),
                os.path.join(labels_dir, f"{subj}.nii"),
            ]
            gt_path = next((p for p in candidates if os.path.exists(p)), None)
            if gt_path is None:
                raise FileNotFoundError(f"Missing reference mask: {ds_name}/{subj}")
            record_input("reference", subj, gt_path)
            img = sitk.ReadImage(gt_path)

        else:
            raise ValueError(f"Unknown source_mode: {source_mode}")

        img = _clamp_labels(img, allowed)
        if subj in global_args.get("reference_geometry_from_mri", {}).get(ds_name, []):
            source_mri = load_t2(subj)
            if source_mri is None:
                raise FileNotFoundError(f"Source MRI required for reference-header correction: {ds_name}/{subj}")
            native_preds = {}
            for model in global_args["models"]:
                fp = pred_index[model["name"]][subj]
                record_input("prediction", subj, fp, model["name"])
                ptype = global_args["pred_type_global"] or model.get("pred_type", "auto")
                native_preds[model["name"]] = load_pred(fp, ptype, global_args["proba_thresh"])
            img, event = correct_reference_geometry(img, source_mri, native_preds, f"{ds_name}/{subj}")
            if source_mode == "legacy_orig_dir":
                mri_path = os.path.join(ds_cfg["orig_dir"], subj, f"{subj}_t2.nii.gz")
                if not os.path.isfile(mri_path):
                    mri_path = os.path.join(ds_cfg["orig_dir"], subj, f"{subj}_t2.nii")
                event["source_mri_file"] = {"path": mri_path, "sha256": file_sha256(mri_path)}
            global_args.setdefault("_reference_geometry_corrections", []).append(event)
            with open(os.path.join(global_args["output_dir"], "reference_geometry_corrections.json"), "w") as handle:
                json.dump(global_args["_reference_geometry_corrections"], handle, indent=2)
            print(f"[REFERENCE GRID][{ds_name}/{subj}] Source MRI corroborated by all native predictions; voxels unchanged.")
        gt_cache[subj] = img
        return img

    def load_t2(subj):
        if subj in t2_cache:
            return t2_cache[subj]

        if source_mode == "legacy_orig_dir":
            for extension in [".nii.gz", ".nii"]:
                p = os.path.join(ds_cfg["orig_dir"], subj, f"{subj}_t2{extension}")
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
            original = arr.copy()
            for k, v in remap.items():
                arr[original == k] = v
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

    # prediction index (ordinary model outputs only; ensemble analysis removed)
    models_all = [m["name"] for m in global_args["models"]]
    model_cfg_by_name = {m["name"]: m for m in global_args["models"]}
    pred_index = {}
    active_models = []

    for m in global_args["models"]:
        mname = m["name"]
        pred_index[mname] = {}
        mdir = resolve_model_pred_dir(m, ds_cfg)

        if cached_manifest:
            pred_index[mname] = {sid: cached_manifest["input_files"]["prediction"][mname][sid]["path"] for sid in subjects}
            active_models.append(mname)
            continue

        if not os.path.isdir(mdir):
            raise FileNotFoundError(f"Missing prediction directory for {ds_name}/{mname}: {mdir}")

        files = glob.glob(os.path.join(mdir, "*.nii")) + glob.glob(os.path.join(mdir, "*.nii.gz"))
        for fp in files:
            sid = re.sub(r"\.nii(\.gz)?$", "", os.path.basename(fp))
            if sid in pred_index[mname]:
                raise ValueError(f"Duplicate prediction file for {ds_name}/{mname}/{sid}.")
            pred_index[mname][sid] = fp

        missing_subjects = sorted(set(subjects) - set(pred_index[mname]))
        if missing_subjects:
            raise FileNotFoundError(f"Missing predictions for {ds_name}/{mname}: {missing_subjects}")

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
                if cached_rows is not None and (ds_name, sid) not in global_args["_recompute_cases"]:
                    for lab in labels:
                        row = dict(cached_rows[(mname, sid, lab)])
                        wr.writerow(row)
                        per_subject_rows.append(row)
                    continue
                gt_img = load_gt(sid)
                if gt_img is None:
                    raise FileNotFoundError(f"Missing reference mask for {ds_name}/{sid}")
                pred_path = pred_index[mname].get(sid, None)
                if not pred_path:
                    raise FileNotFoundError(f"Missing prediction for {ds_name}/{mname}/{sid}")

                record_input("prediction", sid, pred_path, mname)
                pr_img = load_pred(pred_path, pred_type_choice, proba_thresh)
                pr_img = align_to_reference_grid(gt_img, pr_img, global_args, context=f"{ds_name}/{mname}/{sid}")

                t2_img = load_t2(sid) if (
                    global_args.get("make_overlays", False)
                    or global_args.get("save_case_qc", False)
                    or global_args.get("make_case_qc_overlays", False)
                ) else None

                for lab in labels:
                    gt_bin = as_binary(gt_img, lab)
                    pr_bin = as_binary(pr_img, lab)
                    d, j, gt_empty, pr_empty = dice_jaccard(
                        sitk.GetArrayFromImage(gt_bin),
                        sitk.GetArrayFromImage(pr_bin)
                    )
                    hd = hd95_mm(gt_bin, pr_bin)
                    asd = assd_mm(gt_bin, pr_bin)
                    if not gt_empty and not pr_empty and (not np.isfinite(hd) or not np.isfinite(asd)):
                        raise RuntimeError(f"Undefined boundary metric for nonempty masks: {ds_name}/{mname}/{sid}/{lab}")
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

                if (global_args.get("make_overlays", False) or global_args.get("make_case_qc_overlays", False)) and t2_img is not None:
                    t2_img = align_to_reference_grid(gt_img, t2_img, global_args, context=f"{ds_name}/{sid}/MRI overlay", is_mask=False)
                    t2 = sitk.GetArrayFromImage(t2_img)
                    pr = sitk.GetArrayFromImage(pr_img)
                    gt = sitk.GetArrayFromImage(gt_img)
                    out_png = os.path.join(ovl_dir, f"{sid}_{mname}.png")
                    _overlay_png(
                        t2, pr, gt, out_png, sid, label_names,
                        overlay_alpha=global_args.get("overlay_alpha", 0.22),
                        spacing=gt_img.GetSpacing(),
                    )

            if mis_cnt:
                print(f"[MISS PRED][{ds_name}] {mname}: {mis_cnt} missing subjects")

    print(f"[OK][{ds_name}] Wrote detailed per-case metrics: {detailed_csv}")
    write_compiled_csv(os.path.join(out_root, "failure_counts.csv"), failure_summary(per_subject_rows))

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
        hdr = ["dataset","model","label","label_name","n_present"] + [f"{m}_n" for m in metrics] + \
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
                    row[f"{met}_n"] = int(arr.size)
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
        hdr = ["dataset","model","n_subjects"] + [f"{m}_n_subjects" for m in metrics] + \
              sum(([f"{m}_macro_mean", f"{m}_macro_std", f"{m}_macro_median", f"{m}_macro_iqr25", f"{m}_macro_iqr75"] for m in metrics), [])
        wr = csv.DictWriter(fsum2, fieldnames=hdr)
        wr.writeheader()
        for mname in active_models:
            macro = {met: np.asarray(macro_vals(mname, met), float) for met in metrics}
            row = {"dataset": ds_name, "model": mname, "n_subjects": int(len(macro["dice"]))}
            for met, arr in macro.items():
                row[f"{met}_n_subjects"] = int(arr.size)
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
        rows = [r for r in per_subject_rows if r["model"] == model and r["present_case"] == 1]
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
               "p_boot_two_sided"]
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
                    diff_obs, (plo, phi), boots = paired_bootstrap(a, b, n_boot=global_args["n_boot"], seed=global_args["seed"])
                    bca_lo, bca_hi = paired_bca_ci(a, b, n_boot=global_args["n_boot"], alpha=0.05, seed=global_args["seed"])
                    p_two = float(min(1.0, 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())))
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
                    })
    print(f"[OK][{ds_name}] Wrote pairwise comparisons by label: {pair_csv}")

    with open(pair_overall_csv, "w", newline="") as fcmp2:
        hdr = ["dataset","model_a","model_b","n_pairs",
               f"{comp_metric}_macro_mean_a", f"{comp_metric}_macro_mean_b",
               "mean_diff_a_minus_b","boot_lo","boot_hi","bca_lo","bca_hi",
               "p_boot_two_sided"]
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
                diff_obs, (plo, phi), boots = paired_bootstrap(a, b, n_boot=global_args["n_boot"], seed=global_args["seed"])
                bca_lo, bca_hi = paired_bca_ci(a, b, n_boot=global_args["n_boot"], alpha=0.05, seed=global_args["seed"])
                p_two = float(min(1.0, 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())))
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
                })
    print(f"[OK][{ds_name}] Wrote overall (macro) pairwise comparisons: {pair_overall_csv}")
    annotate_pairwise_csv(pair_csv, global_args["n_boot"], adjust_holm=False)
    annotate_pairwise_csv(pair_overall_csv, global_args["n_boot"], adjust_holm=True)

    # ------------------------------------------------------------------
    # Publication-ready per-dataset figures and qualitative case payload
    # ------------------------------------------------------------------
    pub_dir = ensure_dir(os.path.join(fig_dir, "publication"))
    _set_publication_style()

    plot_labels = [
        lab for lab in labels
        if any(len(present_vals(m, lab, "dice")) > 0 for m in active_models)
    ]

    if global_args.get("make_publication_figures", True) and plot_labels:
        for metric, suffix in [("dice", "Dice"), ("hd95_mm", "HD95")]:
            if len(plot_labels) > 1:
                ncols = 2
                nrows = int(math.ceil(len(plot_labels) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(6.7, 3.2 * nrows), squeeze=False)
                for idx, lab in enumerate(plot_labels):
                    ax = axes[idx // ncols, idx % ncols]
                    data = {m: present_vals(m, lab, metric) for m in active_models}
                    _distribution_plot(
                        ax,
                        data,
                        metric,
                        denominators=_metric_denominators(per_subject_rows, ds_name, lab, active_models),
                        title=LABEL_SHORT.get(lab, label_names.get(lab, f"Label {lab}")),
                        ylabel=("Dice" if metric == "dice" else "HD95 (mm)") if idx % ncols == 0 else None,
                        seed=global_args.get("seed", 1337) + idx + (100 if metric != "dice" else 0),
                        add_jitter=global_args.get("jitter_points", True),
                        jitter_width=global_args.get("jitter_width", 0.075),
                        jitter_alpha=global_args.get("jitter_alpha", 0.62),
                        jitter_size=global_args.get("jitter_size", 12),
                    )
                    _panel_label(ax, chr(ord("A") + idx), x=-0.15, y=1.10)
                for idx in range(len(plot_labels), nrows * ncols):
                    axes[idx // ncols, idx % ncols].axis("off")
                fig.subplots_adjust(left=0.09, right=0.995, bottom=0.11, top=0.85, hspace=0.75, wspace=0.27)
            else:
                lab = plot_labels[0]
                fig, ax = plt.subplots(figsize=(3.6, 3.0))
                data = {m: present_vals(m, lab, metric) for m in active_models}
                _distribution_plot(
                    ax,
                    data,
                    metric,
                    denominators=_metric_denominators(per_subject_rows, ds_name, lab, active_models),
                    title=label_names.get(lab, f"Label {lab}"),
                    ylabel="Dice" if metric == "dice" else "HD95 (mm)",
                    seed=global_args.get("seed", 1337) + (100 if metric != "dice" else 0),
                    add_jitter=global_args.get("jitter_points", True),
                    jitter_width=global_args.get("jitter_width", 0.075),
                    jitter_alpha=global_args.get("jitter_alpha", 0.62),
                    jitter_size=global_args.get("jitter_size", 12),
                )
                fig.subplots_adjust(left=0.16, right=0.99, bottom=0.17, top=0.98)

            stem = os.path.join(pub_dir, f"{ds_name}_{suffix}_publication")
            _save_figure(fig, stem, global_args)
            plt.close(fig)
            print(f"[PUBLICATION PANEL][{ds_name}] {stem}.[png|pdf|svg]")

    # Score cases by across-model Dice so the selected qualitative examples are transparent.
    subject_model_dice = defaultdict(lambda: defaultdict(list))
    for r in per_subject_rows:
        if int(r.get("present_case", 0)) != 1 or not np.isfinite(r.get("dice", np.nan)):
            continue
        subject_model_dice[r["subject"]][r["model"]].append(float(r["dice"]))

    ranking_rows = []
    for sid, model_vals in subject_model_dice.items():
        if not all(m in model_vals and model_vals[m] for m in active_models):
            continue
        model_means = {m: float(np.mean(model_vals[m])) for m in active_models}
        vals = np.asarray(list(model_means.values()), float)
        ranking_rows.append({
            "dataset": ds_name,
            "subject": sid,
            "mean_dice_across_models": float(vals.mean()),
            "minimum_model_dice": float(vals.min()),
            "maximum_model_dice": float(vals.max()),
            "model_dice_range": float(vals.max() - vals.min()),
            **{f"{m}_mean_dice": model_means[m] for m in active_models},
        })
    ranking_rows.sort(key=lambda r: r["mean_dice_across_models"], reverse=True)
    ranking_csv = os.path.join(pub_dir, "qualitative_case_rankings.csv")
    if ranking_rows:
        write_compiled_csv(
            ranking_csv,
            ranking_rows,
            preferred_first_cols=[
                "dataset", "subject", "mean_dice_across_models",
                "minimum_model_dice", "maximum_model_dice", "model_dice_range",
            ],
        )

    def make_payload(selected_sid, selection_reason):
        gt_img = load_gt(selected_sid)
        t2_img = load_t2(selected_sid)
        if t2_img is None:
            print(f"[WARN] Qualitative MRI missing for {ds_name}/{selected_sid}; numerical metrics are retained.")
            return None
        t2_img = align_to_reference_grid(gt_img, t2_img, global_args, context=f"{ds_name}/{selected_sid}/qualitative MRI", is_mask=False)
        gt_arr = sitk.GetArrayFromImage(gt_img)
        t2_arr = sitk.GetArrayFromImage(t2_img)
        # Include all configured labels so false-positive labels remain visible.
        sl = _auto_slice_with_all_labels(gt_arr, labels)
        pred_arrays = {}
        for mname in _ordered_models(active_models):
            mcfg = model_cfg_by_name[mname]
            ptype = global_args["pred_type_global"] or mcfg.get("pred_type", "auto")
            record_input("prediction", selected_sid, pred_index[mname][selected_sid], mname)
            pred_img = load_pred(pred_index[mname][selected_sid], ptype, global_args["proba_thresh"])
            pred_img = align_to_reference_grid(gt_img, pred_img, global_args, context=f"{ds_name}/{mname}/{selected_sid}")
            pred_arrays[mname] = sitk.GetArrayFromImage(pred_img)
        union = gt_arr[sl] > 0
        for arr in pred_arrays.values():
            union |= arr[sl] > 0
        y0, y1, x0, x1 = _crop_bounds(union, padding=int(global_args.get("qualitative_crop_padding", 18)))
        ranked = next(r for r in ranking_rows if r["subject"] == selected_sid)
        return {
            "dataset": ds_name, "subject": selected_sid,
            "selection_reason": selection_reason, "slice_index": int(sl),
            "spacing": gt_img.GetSpacing(), "mean_dice_across_models": ranked["mean_dice_across_models"],
            "image": t2_arr[sl, y0:y1, x0:x1], "gt": gt_arr[sl, y0:y1, x0:x1],
            "preds": {m: arr[sl, y0:y1, x0:x1] for m, arr in pred_arrays.items()}, "labels": labels,
        }

    qualitative_payload = None
    external_payloads = []
    selection_records = []
    if ranking_rows:
        configured_sid = (global_args.get("qualitative_cases", {}) or {}).get(ds_name)
        valid_sids = {r["subject"] for r in ranking_rows}
        if configured_sid in valid_sids:
            selected_sid = configured_sid
            selection_reason = "configured"
            if selected_sid == ranking_rows[0]["subject"]:
                selection_reason = "highest across-model mean Dice"
        else:
            mode = str(global_args.get("qualitative_selection_fallback", "representative")).lower()
            means = np.asarray([r["mean_dice_across_models"] for r in ranking_rows], float)
            if mode == "high":
                selected = max(ranking_rows, key=lambda r: r["mean_dice_across_models"])
            elif mode == "disagreement":
                q25 = float(np.quantile(means, 0.25))
                eligible = [r for r in ranking_rows if r["mean_dice_across_models"] >= q25]
                selected = max(eligible, key=lambda r: r["model_dice_range"])
            else:
                target = float(np.median(means))
                selected = min(ranking_rows, key=lambda r: abs(r["mean_dice_across_models"] - target))
            selected_sid = selected["subject"]
            selection_reason = {
                "high": "highest across-model mean Dice",
                "disagreement": "largest model Dice range above the lower quartile",
                "representative": "closest to median across-model mean Dice",
            }.get(mode, "closest to median across-model mean Dice")
            if configured_sid is not None:
                print(f"[WARN][{ds_name}] Configured qualitative case '{configured_sid}' was unavailable; used {mode} case '{selected_sid}'.")

        qualitative_payload = make_payload(selected_sid, selection_reason)
        if qualitative_payload:
            _render_qualitative_row(qualitative_payload, os.path.join(pub_dir, f"qualitative_{ds_name}_{selected_sid}"), global_args)
            selection_records.append({k: v for k, v in qualitative_payload.items() if k not in {"image", "gt", "preds"}})
        if ds_name == "umd_external" and global_args.get("make_external_examples", True):
            for ranked, reason in [(ranking_rows[0], "highest across-model mean Dice"), (ranking_rows[-1], "lowest across-model mean Dice")]:
                payload = make_payload(ranked["subject"], reason)
                if payload is not None:
                    external_payloads.append(payload)
                    selection_records.append({k: v for k, v in payload.items() if k not in {"image", "gt", "preds"}})
            if external_payloads:
                _render_external_examples(external_payloads, os.path.join(pub_dir, "FigureS1_external_success_and_failure"), global_args)
    with open(os.path.join(pub_dir, "qualitative_selection.json"), "w") as handle:
        json.dump(selection_records, handle, indent=2)

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
                    lo, hi = paired_bca_ci(vals, np.zeros_like(vals), n_boot=global_args["n_boot"], alpha=0.05, seed=global_args["seed"])
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            ft.write(m + " & " + " & ".join(cells) + " \\\\\n")
        ft.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[TEX][{ds_name}] {tex1}")

    tex2 = os.path.join(tex_dir, "table_pairwise_overall.tex")
    with open(pair_overall_csv, "r") as f:
        rows_po = list(csv.DictReader(f))
    with open(tex2, "w") as ft:
        ft.write("\\begin{tabular}{l l c c c c}\n\\toprule\n")
        ft.write("Model A & Model B & $n$ & $\\Delta$ (A$-$B) & BCa 95\\% CI & $p_{Holm}$ \\\\\n\\midrule\n")
        for r in rows_po:
            n = int(float(r["n_pairs"])) if r["n_pairs"] != "" else 0
            if n < 5:
                continue
            d = float(r["mean_diff_a_minus_b"])
            lo = float(r["bca_lo"])
            hi = float(r["bca_hi"])
            p = r.get("p_holm_report", r.get("p_boot_report", str(r["p_boot_two_sided"])))
            ft.write(f"{r['model_a']} & {r['model_b']} & {n} & {d:.3f} & [{lo:.3f}, {hi:.3f}] & {p} \\\\\n")
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
        "subjects": subjects,
        "evaluation_cases_sha256": evaluation_cases_hash,
        "script_sha256": file_sha256(__file__),
        "input_files": input_files,
        "cached_evaluation": ({"source_dir": global_args["_cached_results_dir"],
                               "source_script_sha256": cached_manifest["script_sha256"],
                               "recomputed_subjects": sorted(sid for ds, sid in global_args["_recompute_cases"] if ds == ds_name)}
                              if cached_manifest else None),
        "versions": {"python": sys.version, "numpy": np.__version__, "SimpleITK": sitk.Version_VersionString(), "matplotlib": mpl.__version__},
        "revision_evaluation": {
            "n_boot": global_args["n_boot"], "bootstrap_seed": global_args["seed"],
            "resampling_unit": "paired patient-level reference-present macro Dice",
            "bca_intervals": "unadjusted 95 percent intervals",
            "tail_probability": "historical empirical two-sided percentile-tail rule; not inversion of BCa",
            "holm_family": "six overall model comparisons within each dataset, using full precision",
            "hd95": "maximum of directed foreground-surface 95th percentiles, physical mm",
            "assd": "equal average of two directed foreground-surface mean distances, physical mm",
            "absent_labels": "excluded from reference-present primary endpoint; detailed rows retained",
            "empty_predictions": "reference-present Dice zero; boundary metrics undefined and explicitly counted",
            "geometry_policy": global_args.get("geometry_policy", "same_index"),
            "geometry_note": "same_index preserves the original evaluator's corresponding-voxel assumption; mask arrays are unchanged and physical units use the reference grid. Header normalization does not independently verify anatomical alignment.",
            "reference_geometry_corrections": [e for e in global_args.get("_reference_geometry_corrections", []) if e["context"].startswith(ds_name + "/")],
        },
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
        "qualitative_payload": qualitative_payload,
        "external_payloads": external_payloads,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Override the new results directory.")
    parser.add_argument("--previous-output-dir", help="Previous results directory for matched comparisons.")
    parser.add_argument("--datasets", nargs="+", choices=[d["name"] for d in CONFIG["datasets"]])
    parser.add_argument("--reuse-results", help="Reuse audited V2 CSVs and recompute only UMD_221129_090; rebuild all reports and figures.")
    cli = parser.parse_args()
    args = dict(CONFIG)
    if cli.output_dir:
        args["output_dir"] = cli.output_dir
    if cli.previous_output_dir:
        args["previous_output_dir"] = cli.previous_output_dir
    if cli.datasets:
        args["datasets"] = [d for d in args["datasets"] if d["name"] in cli.datasets]
    if os.path.realpath(args["output_dir"]) == os.path.realpath(args.get("previous_output_dir") or ""):
        raise ValueError("Choose a new output directory to retain the previous results for comparison.")
    if cli.reuse_results:
        prepare_targeted_correction(args, cli.reuse_results)
    run_evaluation(args)


def run_evaluation(args):
    _set_publication_style()
    ensure_dir(args["output_dir"])
    publication_dir = ensure_dir(os.path.join(args["output_dir"], "publication_figures"))

    datasets = args.get("datasets", None)
    if not datasets:
        raise SystemExit("CONFIG['datasets'] must be defined.")

    all_per_subject_rows = []
    all_summary_label_rows = []
    all_summary_overall_rows = []
    all_pair_label_rows = []
    all_pair_overall_rows = []
    qualitative_payloads = {}
    external_payloads = []

    for ds_cfg in datasets:
        result = run_one_dataset(args, ds_cfg)
        if result is None:
            continue
        all_per_subject_rows.extend(result.get("per_subject_rows", []))
        all_summary_label_rows.extend(result.get("summary_label_rows", []))
        all_summary_overall_rows.extend(result.get("summary_overall_rows", []))
        all_pair_label_rows.extend(result.get("pair_label_rows", []))
        all_pair_overall_rows.extend(result.get("pair_overall_rows", []))
        payload = result.get("qualitative_payload")
        if payload is not None:
            qualitative_payloads[result["dataset"]] = payload
        external_payloads.extend(result.get("external_payloads", []))

    verify_targeted_correction(all_per_subject_rows, args)
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

    if args.get("make_publication_figures", True) and all_per_subject_rows:
        _make_composite_metric_figure(all_per_subject_rows, "dice", publication_dir, args)
        _make_composite_metric_figure(all_per_subject_rows, "hd95_mm", publication_dir, args)
        _render_qualitative_composite(qualitative_payloads, publication_dir, args)
        if external_payloads:
            _render_external_examples(external_payloads, os.path.join(publication_dir, "FigureS1_external_success_and_failure"), args)

    write_compiled_csv(os.path.join(args["output_dir"], "all_failure_counts.csv"), failure_summary(all_per_subject_rows))
    if "_cached_case_rows" in args:
        for filename in ["all_failure_counts.csv", "all_pairwise_overall.csv", "all_pairwise_by_label.csv"]:
            with open(os.path.join(args["_cached_results_dir"], filename), newline="") as handle:
                old = list(csv.DictReader(handle))
            with open(os.path.join(args["output_dir"], filename), newline="") as handle:
                new = list(csv.DictReader(handle))
            if old != new:
                raise RuntimeError(f"Targeted physical-grid repair unexpectedly changed {filename}.")
        with open(os.path.join(args["output_dir"], "geometry_adjustments.json"), "w") as handle:
            json.dump(args.get("_geometry_events", []), handle, indent=2)
        print("[VERIFIED] Failure denominators and both label-wise and overall paired Dice statistics are unchanged.")
    compare_previous_outputs(all_per_subject_rows, args)
    with open(os.path.join(args["output_dir"], "figure_caption_notes.txt"), "w") as handle:
        handle.write("Figure 2: The supplied cases are the highest across-model mean reference-positive macro Dice examples in each dataset; illustrative successful cases, not typical samples. Case identifiers and selection are displayed. Labels show reference-mask targets, not independent biological confirmation. Slice selected by greatest number of reference labels, then greatest labeled area. Crops enclose the union of reference and predicted labels on that slice; display aspect uses voxel spacing.\n"
                     "Figure 3: Reference-positive per-label Dice; points are patients, horizontal bars are medians, violins describe the distribution. Complete misses contribute Dice zero; reference-negative labels are excluded. n is the number of patients scored.\n"
                     "Figure 4: HD95 in mm, conditional on nonempty reference and prediction. Horizontal lines are medians; boxes span quartiles; whiskers follow the 1.5-IQR rule; individual points are shown. n/N gives evaluable/reference-positive cases; M0, M1, etc. indicate the number of empty-prediction misses in compact panels. The symmetric-log axis retains zero values with a linear region near zero.\n"
                     "Figure S1: External cases with highest and lowest across-model mean reference-positive macro Dice; illustrative extremes, not representative samples or statistical comparisons.\n")
    write_review_bundle(args)

    print("\nAll done.")
    print(f"Publication-ready assets: {publication_dir}")

if __name__ == "__main__":
    main()
