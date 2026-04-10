#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public-facing launcher for k-fold MONAI segmentation training on multiple GPUs.

This script assumes nnU-Net planning and preprocessing have already been run and
that you have the resulting nnUNetPlans.json, dataset.json, splits file, and
preprocessed configuration directory (for example nnUNetPlans_3d_fullres).
"""

import os
import subprocess
import sys
import argparse
import time
import threading


def _file_md5(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# defaults
GPUS = ["0", "1"]
KFOLDS = 5
DATASET_JSON = None
PLANS = None
PREPROC_DIR = None
SPLITS = None
SCRIPT = "train_umdfibroid_monai.py"  # override with --script
TASK = "multiclass"


def build_argparser():
    p = argparse.ArgumentParser(
        description="Launch k-fold training on multiple GPUs with live logs (compatible with updated trainer)."
    )

    p.add_argument(
        "--model", type=str, default=None,
        choices=["unet3d", "unetr", "dynunet", "segresnet", "swinunetr"],
        help="If omitted, run baseline models: unet3d/unetr/segresnet/swinunetr",
    )

    p.add_argument("--folds", type=str, default=None, help="Comma-separated folds to run. If omitted, run all.")
    p.add_argument("--exclude", type=str, default=None, help="Comma-separated folds to skip.")
    p.add_argument("--kfolds", type=int, default=KFOLDS)
    p.add_argument("--gpus", type=str, default=",".join(GPUS))

    p.add_argument("--plans-json", type=str, default=PLANS)
    p.add_argument(
        "--dataset-json", type=str, default=DATASET_JSON,
        help="Path to nnU-Net dataset.json (used by train script to infer num-classes if omitted).",
    )
    p.add_argument("--preproc-dir", type=str, default=PREPROC_DIR)
    p.add_argument("--splits-json", type=str, default=SPLITS)
    p.add_argument("--nnunet-config", type=str, default="3d_fullres")

    p.add_argument(
        "--outdir-base", type=str, default=None,
        help="Base output dir. If omitted, uses runs/lms_<task>/<model>/fold_XX",
    )
    p.add_argument("--script", type=str, default=SCRIPT)

    # passthrough training args
    p.add_argument("--task", type=str, default=TASK, choices=["binary", "multiclass"])
    p.add_argument(
        "--num-classes", type=int, default=None,
        help="If set, forwarded to train script. If omitted, train script can infer from dataset.json.",
    )
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--lesion-class-id", type=int, default=1)

    # sampler knobs
    p.add_argument("--oversample-foreground-percent", type=float, default=0.33)
    p.add_argument(
        "--fg-sampling", type=str, default="perclass", choices=["perclass", "any"],
        help="Foreground sampling strategy. 'perclass' mimics nnU-Net class_locations.",
    )
    p.add_argument(
        "--fg-max-points-per-class", type=int, default=200000,
        help="Cap cached per-class coords per case per label in the trainer.",
    )
    p.add_argument(
        "--sampler", type=str, default="nnunet", choices=["native", "nnunet"],
        help="Patch sampler for b2nd backend. 'nnunet' uses vendored nnU-Net-like sampler.",
    )
    p.add_argument(
        "--fg-perclass-selection", type=str, default="uniform", choices=["uniform", "inverse_frequency"],
        help="When fg-sampling perclass: how to choose among present classes.",
    )
    p.add_argument(
        "--fg-min-voxels-in-patch", type=int, default=0,
        help="If >0, forced-FG patches are re-sampled until the chosen FG class has at least this many voxels (best-effort).",
    )
    p.add_argument("--fg-resample-max-tries", type=int, default=8)

    # training schedule
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--num-iterations-per-epoch", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=3e-5)

    # validation
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--val-max-batches", type=int, default=32)
    p.add_argument("--val-mode", type=str, choices=["patch", "fullcase"], default="fullcase")
    p.add_argument("--val-num-cases", type=int, default=8)

    # system / reproducibility
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deep-supervision", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--force-amp-segresnet", action="store_true")

    # optional debug overrides
    p.add_argument("--patch-size", type=int, nargs=3, default=None)
    p.add_argument("--batch-size", type=int, default=None)

    # ignore label handling
    p.add_argument("--ignore-index", type=int, default=-1)
    p.add_argument("--ignore-labels", type=int, nargs="*", default=[255, -1])

    # loss knobs
    p.add_argument("--loss-dice-weight", type=float, default=1.0)
    p.add_argument("--loss-ce-weight", type=float, default=1.0)
    p.add_argument("--dice-smooth-nr", type=float, default=1e-5)
    p.add_argument("--dice-smooth-dr", type=float, default=1e-5)
    p.add_argument("--dice-include-background", action="store_true")

    # lr / stability
    p.add_argument("--auto-lr-scale", action="store_true")
    p.add_argument("--segresnet-lr-scale", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=12.0)

    # caches / performance
    p.add_argument("--case-cache-size", type=int, default=16)
    p.add_argument("--fg-cache-size", type=int, default=512)
    p.add_argument("--mta-cache", type=int, default=6)
    p.add_argument("--no-reuse-within-batch", action="store_true")
    p.add_argument("--torch-num-threads", type=int, default=1)

    # robustness / debugging
    p.add_argument("--allow-missing-cases", action="store_true")
    p.add_argument(
        "--debug-sampler-every", type=int, default=0,
        help="Forwarded to trainer (and then to sampler): print sampler debug stats every N iterations.",
    )
    p.add_argument(
        "--debug-label4", action="store_true",
        help="Forwarded to trainer: print per-case GT/pred counts for label 4 during validation.",
    )

    # Swin knobs
    # Smaller default to better fit common GPUs; previous default 24 can be heavy.
    p.add_argument("--swin-feature-size", type=int, default=12)
    p.add_argument("--swin-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--swin-depths", type=int, nargs=4, default=[2, 2, 2, 1])
    p.add_argument("--swin-window", type=int, default=7)
    p.add_argument("--swin-use-checkpoint", action="store_true")

    return p


def main():
    args = build_argparser().parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("[ERR] No GPUs specified", file=sys.stderr)
        sys.exit(1)

    required_missing = []
    for label, value in [("plans-json", args.plans_json), ("preproc-dir", args.preproc_dir), ("splits-json", args.splits_json)]:
        if value in (None, ""):
            required_missing.append(label)
    if args.dataset_json is None and args.num_classes is None:
        required_missing.append("dataset-json or num-classes")
    if required_missing:
        msg = ", ".join(required_missing)
        print(f"[ERR] Missing required arguments: {msg}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.script):
        print(f"[ERR] Training script not found: {args.script}", file=sys.stderr)
        sys.exit(1)

    script_abs = os.path.abspath(args.script)
    print(f"[audit] launcher_target_script={script_abs}")
    try:
        print(f"[audit] launcher_target_script_md5={_file_md5(script_abs)}")
    except Exception as e:
        print(f"[audit] launcher_target_script_md5_error={e}")

    for need, kind in [(args.plans_json, "plans"), (args.splits_json, "splits")]:
        if not os.path.isfile(need):
            print(f"[ERR] {kind} file not found: {need}", file=sys.stderr)
            sys.exit(1)
    if args.dataset_json is not None and not os.path.isfile(args.dataset_json):
        print(f"[ERR] dataset-json not found: {args.dataset_json}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.preproc_dir):
        print(f"[ERR] preproc-dir not found: {args.preproc_dir}", file=sys.stderr)
        sys.exit(1)

    models_to_run = [args.model.lower()] if args.model else ["unet3d", "unetr", "segresnet", "swinunetr"]

    kfolds = int(args.kfolds)
    folds_all = list(range(kfolds))
    if args.folds:
        folds_to_run = [int(x) for x in args.folds.split(",") if x.strip()]
    else:
        folds_to_run = folds_all
    if args.exclude:
        excl = {int(x) for x in args.exclude.split(",") if x.strip()}
        folds_to_run = [f for f in folds_to_run if f not in excl]
    folds_to_run = sorted(set(folds_to_run))

    outdir_base = args.outdir_base or os.path.join("runs", f"project_{args.task}")
    os.makedirs(outdir_base, exist_ok=True)

    jobs_pending = []
    for model_name in models_to_run:
        model_outdir = os.path.join(outdir_base, model_name)
        os.makedirs(model_outdir, exist_ok=True)

        for fold in folds_to_run:
            fold_outdir = os.path.join(model_outdir, f"fold_{fold:02d}")
            os.makedirs(fold_outdir, exist_ok=True)
            log_path = os.path.join(fold_outdir, "train.log")

            # skip if a best checkpoint exists
            best_exists = any(fn.startswith("best_") and fn.endswith(".pt") for fn in os.listdir(fold_outdir))
            if best_exists:
                print(f"[skip] {model_name} fold {fold} already has best checkpoint")
                continue

            cmd = [
                sys.executable, args.script,
                "--plans-json", args.plans_json,
                "--preproc-dir", args.preproc_dir,
                "--splits-json", args.splits_json,
                "--nnunet-config", args.nnunet_config,
                "--fold", str(fold),
                "--outdir", fold_outdir,
                "--model", model_name,
                "--task", args.task,
                "--in-channels", str(args.in_channels),
                "--lesion-class-id", str(args.lesion_class_id),

                "--oversample-foreground-percent", str(args.oversample_foreground_percent),
                "--fg-sampling", str(args.fg_sampling),
                "--fg-max-points-per-class", str(args.fg_max_points_per_class),
                "--sampler", str(args.sampler),
                "--fg-perclass-selection", str(args.fg_perclass_selection),
                "--fg-min-voxels-in-patch", str(args.fg_min_voxels_in_patch),
                "--fg-resample-max-tries", str(args.fg_resample_max_tries),

                "--max-epochs", str(args.max_epochs),
                "--num-iterations-per-epoch", str(args.num_iterations_per_epoch),
                "--lr", str(args.lr),
                "--poly-power", str(args.poly_power),
                "--weight-decay", str(args.weight_decay),

                "--val-every", str(args.val_every),
                "--val-max-batches", str(args.val_max_batches),
                "--val-mode", str(args.val_mode),
                "--val-num-cases", str(args.val_num_cases),

                "--num-workers", str(args.num_workers),
                "--seed", str(args.seed),

                "--ignore-index", str(args.ignore_index),
                "--ignore-labels", *[str(x) for x in args.ignore_labels],

                "--loss-dice-weight", str(args.loss_dice_weight),
                "--loss-ce-weight", str(args.loss_ce_weight),
                "--dice-smooth-nr", str(args.dice_smooth_nr),
                "--dice-smooth-dr", str(args.dice_smooth_dr),

                "--grad-clip-norm", str(args.grad_clip_norm),

                "--case-cache-size", str(args.case_cache_size),
                "--fg-cache-size", str(args.fg_cache_size),
                "--mta-cache", str(args.mta_cache),
                "--torch-num-threads", str(args.torch_num_threads),
            ]

            if args.dataset_json is not None:
                cmd.extend(["--dataset-json", args.dataset_json])

            if args.num_classes is not None:
                cmd.extend(["--num-classes", str(args.num_classes)])
            else:
                # Keep prior launcher behavior: if you didn't provide dataset-json OR num-classes,
                # fallback to 5 to avoid trainer aborting (you can change this default if needed).
                if args.dataset_json is None:
                    cmd.extend(["--num-classes", "5"])

            if args.deep_supervision:
                cmd.append("--deep-supervision")
            if args.no_amp:
                cmd.append("--no-amp")
            if args.force_amp_segresnet:
                cmd.append("--force-amp-segresnet")

            if args.dice_include_background:
                cmd.append("--dice-include-background")

            if args.auto_lr_scale:
                cmd.append("--auto-lr-scale")
            if args.segresnet_lr_scale is not None:
                cmd.extend(["--segresnet-lr-scale", str(args.segresnet_lr_scale)])

            if args.allow_missing_cases:
                cmd.append("--allow-missing-cases")

            if args.no_reuse_within_batch:
                cmd.append("--no-reuse-within-batch")

            if int(getattr(args, "debug_sampler_every", 0)) > 0:
                cmd.extend(["--debug-sampler-every", str(int(args.debug_sampler_every))])
            if bool(getattr(args, "debug_label4", False)):
                cmd.append("--debug-label4")

            if args.patch_size is not None:
                cmd.extend(["--patch-size", *[str(x) for x in args.patch_size]])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])

            if model_name == "swinunetr":
                # If the user kept launcher defaults geared for CNNs (lr=1e-2 / wd=3e-5),
                # switch to Swin-friendly defaults unless they explicitly set something else.
                # (You can still override via CLI: --lr / --weight-decay / --respect-lr on trainer.)
                if abs(float(args.lr) - 1e-2) < 1e-12:
                    # warmup+cosine in trainer expects a smaller lr
                    cmd[cmd.index("--lr") + 1] = str(2e-4)
                if abs(float(args.weight_decay) - 3e-5) < 1e-12:
                    cmd[cmd.index("--weight-decay") + 1] = str(5e-2)
                cmd.extend([
                    "--swin-feature-size", str(args.swin_feature_size),
                    "--swin-heads", *[str(x) for x in args.swin_heads],
                    "--swin-depths", *[str(x) for x in args.swin_depths],
                    "--swin-window", str(args.swin_window),
                ])
                if args.swin_use_checkpoint:
                    cmd.append("--swin-use-checkpoint")

            jobs_pending.append({
                "model": model_name,
                "fold": fold,
                "fold_outdir": fold_outdir,
                "log_path": log_path,
                "cmd": cmd,
                "gpu": None,
            })

    if not jobs_pending:
        print("[info] Nothing to run; all requested folds already completed.")
        return

    running_jobs = []

    def reader_thread(proc, log_f):
        # Stream output live
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

    def gpu_in_use(gpu_id: str) -> bool:
        return any(j["gpu"] == gpu_id for j in running_jobs)

    while jobs_pending or running_jobs:
        launched_this_round = False

        for job in list(jobs_pending):
            free_gpu = None
            for g in gpus:
                if not gpu_in_use(g):
                    free_gpu = g
                    break
            if free_gpu is None:
                break

            job["gpu"] = free_gpu
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(free_gpu)

            os.makedirs(os.path.dirname(job["log_path"]), exist_ok=True)
            log_f = open(job["log_path"], "w")
            log_f.write(" ".join(job["cmd"]) + "\n\n")
            log_f.flush()

            print("\n" + "=" * 80)
            print(f"[launch] model={job['model']} fold={job['fold']} gpu={free_gpu}")
            print(" ".join(job["cmd"]))
            print("=" * 80 + "\n")

            proc = subprocess.Popen(
                job["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )

            t = threading.Thread(target=reader_thread, args=(proc, log_f), daemon=True)
            t.start()

            running_jobs.append({
                "proc": proc,
                "thread": t,
                "log_f": log_f,
                "gpu": free_gpu,
                "model": job["model"],
                "fold": job["fold"],
            })
            jobs_pending.remove(job)
            launched_this_round = True

        for j in list(running_jobs):
            ret = j["proc"].poll()
            if ret is not None:
                j["thread"].join()
                try:
                    j["log_f"].flush()
                finally:
                    j["log_f"].close()
                running_jobs.remove(j)
                if ret == 0:
                    print(f"[ok] Finished model={j['model']} fold={j['fold']} on gpu={j['gpu']}")
                else:
                    print(f"[ERR] model={j['model']} fold={j['fold']} on gpu={j['gpu']} exited with code {ret}")

        if not launched_this_round and running_jobs:
            time.sleep(1.0)

    print("[info] All jobs completed.")


if __name__ == "__main__":
    main()