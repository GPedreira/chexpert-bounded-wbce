"""WBCE positive-weight rescaling on CheXpert: training, evaluation and figures."""

import gc
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

CLASS_NAMES_14 = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
FOCUS_6 = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pneumothorax",
    "Pneumonia",
]
TAIL_2 = ["Pneumothorax", "Pneumonia"]
CHEXPERT_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
LAMBDAS = [1, 3, 5]
SEEDS = [2026, 123, 456]


def apply_uncertainty_policy(df, strategy="u_soft", class_names=CLASS_NAMES_14):
    if strategy not in {"u_soft", "u_zeros", "u_ones", "u_ignore", "hybrid"}:
        raise ValueError(f"Unknown uncertainty policy: {strategy}")
    targets = df[class_names].fillna(0.0).astype(np.float64).copy()
    masks = pd.DataFrame(1.0, index=df.index, columns=class_names)
    for name in class_names:
        uncertain = targets[name].eq(-1)
        value = (
            0.55
            if strategy == "u_soft"
            else float(
                strategy == "u_ones"
                or (strategy == "hybrid" and name in {"Atelectasis", "Edema"})
            )
        )
        if strategy == "u_ignore":
            masks.loc[uncertain, name] = 0.0
        targets.loc[uncertain, name] = value
    return (targets.to_numpy(dtype=np.float32), masks.to_numpy(dtype=np.float32))


def evaluation_targets(df, class_names=CLASS_NAMES_14):
    raw = df[class_names]
    binary = raw.isin([0, 1])
    return (
        raw.where(binary, 0).fillna(0).to_numpy(dtype=np.float32),
        binary.to_numpy(dtype=np.float32),
    )


def compute_w_tilde(df, class_names=CLASS_NAMES_14):
    if len(df) == 0:
        raise ValueError("Empty training partition.")
    frequency = df[class_names].eq(1).sum() / len(df)
    if (frequency == 0).any():
        raise ValueError("A training category has no positive observations.")
    return (1 - frequency) / frequency


def make_pos_weight(lambda_max, weights):
    if lambda_max == "raw":
        return np.asarray(weights, dtype=np.float32)
    value = float(lambda_max)
    if value < 1:
        raise ValueError("lambda_max must be at least one.")
    weights = np.asarray(weights, dtype=np.float64)
    if value == 1:
        return np.ones_like(weights, dtype=np.float32)
    spread = weights.max() - weights.min()
    if spread == 0:
        raise ValueError("Min-max rescaling is undefined for identical raw costs.")
    return (1 + (value - 1) * (weights - weights.min()) / spread).astype(np.float32)


def optimize_thresholds_vectorized(
    y_true, y_score, masks=None, class_names=CLASS_NAMES_14
):
    thresholds = {}
    for i, c in enumerate(class_names):
        m = masks[:, i].astype(bool) if masks is not None else slice(None)
        yt, ys = (y_true[m, i], y_score[m, i])
        if len(np.unique(yt)) < 2:
            thresholds[c] = 0.5
            continue
        precision, recall, thresh = precision_recall_curve(yt, ys)
        denom = precision + recall
        f1_scores = np.zeros_like(precision)
        valid = denom > 0
        f1_scores[valid] = 2 * (precision[valid] * recall[valid]) / denom[valid]
        if len(thresh) > 0:
            thresholds[c] = float(thresh[np.argmax(f1_scores[:-1])])
        else:
            thresholds[c] = 0.5
    return thresholds


def fbeta_score_safe(y_true, y_pred, beta=2.0):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp == 0:
        return 0.0
    prec, rec = (tp / (tp + fp), tp / (tp + fn))
    return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)


def expected_calibration_error_masked(y_true, y_score, mask=None, n_bins=10):
    if mask is not None:
        v = mask.astype(bool)
        y_true, y_score = (y_true[v], y_score[v])
    else:
        y_true, y_score = (y_true.ravel(), y_score.ravel())
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.percentile(y_score, quantiles * 100))
    if len(bin_edges) == 1:
        return float(abs(y_true.mean() - y_score.mean()))
    bin_edges[0], bin_edges[-1] = (0.0, 1.0)
    ece, n = (0.0, len(y_score))
    for i in range(len(bin_edges) - 1):
        lo, hi = (bin_edges[i], bin_edges[i + 1])
        idx = (
            (y_score >= lo) & (y_score <= hi)
            if i == len(bin_edges) - 2
            else (y_score >= lo) & (y_score < hi)
        )
        if idx.sum() > 0:
            ece += idx.sum() / n * abs(y_true[idx].mean() - y_score[idx].mean())
    return float(ece)


def compute_metric_suite(
    y_true, y_score, thresholds, masks=None, class_names=CLASS_NAMES_14
):
    per_class = {}
    for i, c in enumerate(class_names):
        m = masks[:, i].astype(bool) if masks is not None else slice(None)
        yt, ys = (y_true[m, i], y_score[m, i])
        if len(np.unique(yt)) < 2:
            per_class[c] = dict(
                f1=np.nan,
                f2=np.nan,
                auc=np.nan,
                ap=np.nan,
                sens_at_95spec=np.nan,
                ece=np.nan,
                brier=np.nan,
            )
            continue
        preds = (ys >= thresholds[c]).astype(int)
        fpr, tpr, _ = roc_curve(yt, ys)
        valid_idx = 1 - fpr >= 0.95
        per_class[c] = dict(
            f1=f1_score(yt, preds, zero_division=0),
            f2=fbeta_score_safe(yt, preds, beta=2.0),
            auc=roc_auc_score(yt, ys),
            ap=average_precision_score(yt, ys),
            sens_at_95spec=tpr[valid_idx].max() if valid_idx.any() else 0.0,
            ece=expected_calibration_error_masked(yt, ys),
            brier=brier_score_loss(yt, ys),
        )
    return pd.DataFrame(per_class).T


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_paths(root):
    paths = {
        (lam, seed): Path(root)
        / "checkpoints"
        / f"phase2b_lambda{lam}_seed{seed}_best.pt"
        for lam in LAMBDAS
        for seed in SEEDS
    }
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required checkpoints: " + ", ".join(missing))
    return paths


def checkpoint_thresholds(checkpoint):
    thresholds = checkpoint.get("dev_metrics", {}).get("dev_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(CLASS_NAMES_14):
        raise ValueError(
            "Checkpoint must contain all 14 development thresholds; no fallback is allowed."
        )
    if not all((np.isfinite(v) and 0 <= v <= 1 for v in thresholds.values())):
        raise ValueError("Invalid development threshold.")
    return thresholds


def find_csv(root, name="valid.csv"):
    matches = sorted(Path(root).rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name} under {root}; found {len(matches)}."
        )
    return matches[0]


def new_output(root):
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(root) / tag
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def make_transforms(train=False, seed=None):
    transforms = [A.Resize(320, 320)]
    if train:
        transforms.append(A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT))
    transforms.extend(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms, seed=seed)


def resolve_image_path(base_dir, raw_path):
    candidate = Path(base_dir) / raw_path.replace("CheXpert-v1.0-small/", "")
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing image: {candidate}")
    return candidate


class CheXpertDataset(Dataset):

    def __init__(self, df, base_dir, transform, is_train=False, strategy="u_soft"):
        self.paths = [resolve_image_path(base_dir, p) for p in df["Path"]]
        self.transform = transform
        self.targets, self.masks = (
            apply_uncertainty_policy(df, strategy)
            if is_train
            else evaluation_targets(df)
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise ValueError(f"Unreadable image: {self.paths[index]}")
        image = self.transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"]
        return (
            image,
            torch.from_numpy(self.targets[index]),
            torch.from_numpy(self.masks[index]),
        )


def build_model(pretrained=False):
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES_14))
    return model


def image_bootstrap(
    y_true, baseline, weighted, thresholds_b, thresholds_w, attempts=2000, seed=2026
):
    rng = np.random.RandomState(seed)
    records = []
    for name in TAIL_2:
        j = CLASS_NAMES_14.index(name)
        yt = y_true[:, j]
        pb = (baseline[:, j] >= thresholds_b[name]).astype(int)
        pw = (weighted[:, j] >= thresholds_w[name]).astype(int)
        vb, vw = ([], [])
        for _ in range(attempts):
            ids = rng.randint(0, len(yt), size=len(yt))
            if len(np.unique(yt[ids])) < 2:
                continue
            vb.append(f1_score(yt[ids], pb[ids], zero_division=0))
            vw.append(f1_score(yt[ids], pw[ids], zero_division=0))
        if not vb:
            raise ValueError(f"No valid bootstrap draws for {name}.")
        vb, vw = (np.array(vb), np.array(vw))
        delta = vw - vb
        records.append(
            {
                "class": name,
                "attempts": attempts,
                "valid_draws": len(vb),
                "baseline_f1_observed": f1_score(yt, pb, zero_division=0),
                "lambda5_f1_observed": f1_score(yt, pw, zero_division=0),
                "baseline_boot_mean": vb.mean(),
                "lambda5_boot_mean": vw.mean(),
                "baseline_ci_lo": np.percentile(vb, 2.5),
                "baseline_ci_hi": np.percentile(vb, 97.5),
                "lambda5_ci_lo": np.percentile(vw, 2.5),
                "lambda5_ci_hi": np.percentile(vw, 97.5),
                "gain_percent": (
                    100 * (vw.mean() - vb.mean()) / vb.mean() if vb.mean() else np.nan
                ),
                "paired_delta_mean": delta.mean(),
                "paired_delta_ci_lo": np.percentile(delta, 2.5),
                "paired_delta_ci_hi": np.percentile(delta, 97.5),
            }
        )
    return pd.DataFrame(records)


METRICS = [
    "macro_f1_tail",
    "macro_f2_tail",
    "macro_auc_tail",
    "sens_at_95spec_tail",
    "ece_tail",
    "brier_tail",
    "chexpert5_macro_auc",
]


def infer(model, loader, device, precision):
    model.eval()
    targets, scores, masks = ([], [], [])
    with torch.inference_mode():
        for images, y, mask in loader:
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if precision == "amp"
                else nullcontext()
            )
            with context:
                logits = model(images.to(device))
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(y.numpy())
            masks.append(mask.numpy())
    return (np.vstack(targets), np.vstack(scores), np.vstack(masks))


def environment_manifest(device, precision, batch_size):
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "albumentations",
        "opencv-python-headless",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    repo = Path(__file__).resolve().parent
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "python": sys.version,
        "packages": versions,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "precision": precision,
        "batch_size": batch_size,
        "tf32": False,
        "code_commit": git_commit,
    }


def evaluate(args):
    root = Path(args.experiment_root)
    expected = checkpoint_paths(root)
    valid_csv = find_csv(args.data_dir)
    df = pd.read_csv(valid_csv)
    if len(df) != 234 or not df["Path"].is_unique:
        raise ValueError("Expected 234 unique official-validation images.")
    patients = df["Path"].str.extract("(patient\\d+)", expand=False)
    if patients.isna().any() or patients.nunique() != 200:
        raise ValueError(
            "Expected valid image paths from 200 official-validation patients."
        )
    raw = df[CLASS_NAMES_14]
    if not raw[TAIL_2 + CHEXPERT_5].isin([0, 1]).all().all():
        raise ValueError(
            "Tail and CheXpert-5 evaluation requires binary reference labels."
        )
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    precisions = ["fp32", "amp"] if args.precision == "both" else [args.precision]
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise ValueError("CUDA requested but unavailable.")
    if "amp" in precisions and device.type != "cuda":
        raise ValueError("AMP evaluation requires CUDA.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = new_output(args.output_root)
    counts = pd.DataFrame(
        {
            "n_binary": raw.isin([0, 1]).sum(),
            "n_positive": raw.eq(1).sum(),
            "n_uncertain": raw.eq(-1).sum(),
            "n_missing": raw.isna().sum(),
        }
    )
    counts.to_csv(output / "label_counts.csv")
    loader = DataLoader(
        CheXpertDataset(df, valid_csv.parent, make_transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    manifest = environment_manifest(device, precisions, args.batch_size)
    manifest.update(
        valid_csv_sha256=sha256_file(valid_csv),
        checkpoint_sha256={},
        preprocessing="resize_normalize",
    )
    write_json(output / "manifest.json", manifest)
    rows, class_rows, cache = ([], [], {})
    idx_tail = [CLASS_NAMES_14.index(c) for c in TAIL_2]
    started = time.perf_counter()
    for precision in precisions:
        for (lam, seed), path in expected.items():
            print(f"Evaluating {precision}: lambda={lam}, seed={seed}", flush=True)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            thresholds = checkpoint_thresholds(checkpoint)
            manifest["checkpoint_sha256"][path.name] = sha256_file(path)
            model = build_model()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model = model.to(device)
            y_true, scores, masks = infer(model, loader, device, precision)
            if not np.isfinite(scores).all() or scores.shape != (234, 14):
                raise ValueError("Invalid prediction array.")
            per_class = compute_metric_suite(y_true, scores, thresholds, masks)
            rows.append(
                {
                    "precision": precision,
                    "lambda_max": lam,
                    "seed": seed,
                    "macro_f1_tail": per_class.loc[TAIL_2, "f1"].mean(),
                    "macro_f2_tail": per_class.loc[TAIL_2, "f2"].mean(),
                    "macro_auc_tail": per_class.loc[TAIL_2, "auc"].mean(),
                    "sens_at_95spec_tail": per_class.loc[
                        TAIL_2, "sens_at_95spec"
                    ].mean(),
                    "ece_tail": expected_calibration_error_masked(
                        y_true[:, idx_tail], scores[:, idx_tail], masks[:, idx_tail]
                    ),
                    "brier_tail": per_class.loc[TAIL_2, "brier"].mean(),
                    "chexpert5_macro_auc": per_class.loc[CHEXPERT_5, "auc"].mean(),
                }
            )
            class_rows.append(
                per_class.rename_axis("class")
                .reset_index()
                .assign(precision=precision, lambda_max=lam, seed=seed)
            )
            cache[precision, lam, seed] = (scores, thresholds)
            np.savez_compressed(
                output / f"predictions_{precision}_lambda{lam}_seed{seed}.npz",
                y_true=y_true,
                y_score=scores,
                masks=masks,
                paths=df["Path"].to_numpy(dtype=str),
                patients=patients.to_numpy(dtype=str),
                classes=np.array(CLASS_NAMES_14),
            )
            write_json(
                output / f"thresholds_{precision}_lambda{lam}_seed{seed}.json",
                thresholds,
            )
            del model, checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    results = pd.DataFrame(rows)
    if len(results) != 9 * len(precisions):
        raise ValueError("Incomplete evaluation: expected nine models per precision.")
    results.to_csv(output / "results_per_model.csv", index=False)
    pd.concat(class_rows).to_csv(output / "metrics_per_class.csv", index=False)
    summary = results.groupby(["precision", "lambda_max"])[METRICS].agg(["mean", "std"])
    summary.to_csv(output / "summary_mean_sd.csv")
    boots = []
    for precision in precisions:
        baseline, tb = cache[precision, 1, 2026]
        weighted, tw = cache[precision, 5, 2026]
        bootstrap = image_bootstrap(y_true, baseline, weighted, tb, tw, args.bootstrap)
        boots.append(bootstrap.assign(precision=precision))
    pd.concat(boots).to_csv(output / "bootstrap_images_seed2026.csv", index=False)
    manifest.update(elapsed_seconds=time.perf_counter() - started, status="complete")
    write_json(output / "manifest.json", manifest)
    print(summary.round(6).to_string())
    print(f"Evaluation saved to: {output}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def masked_weighted_bce(logits, targets, masks, positive_weights):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weights.to(logits.device), reduction="none"
    )
    return (loss * masks).sum() / masks.sum().clamp(min=1.0)


def unmasked_development_ece(y_true, scores, n_bins=10):
    """Phase 1/2a ECE uses zero-filled development targets without the binary mask."""
    yt, ys = (y_true.ravel(), scores.ravel())
    edges = np.quantile(ys, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = (0.0, 1.0)
    result = 0.0
    for j in range(n_bins):
        ids = (
            (ys >= edges[j]) & (ys <= edges[j + 1])
            if j == n_bins - 1
            else (ys >= edges[j]) & (ys < edges[j + 1])
        )
        if ids.any():
            result += ids.sum() / len(ys) * abs(yt[ids].mean() - ys[ids].mean())
    return float(result)


def split_training(df, split_path, external_split=None):
    df = df.copy()
    df["patient_id"] = df["Path"].str.extract("(patient\\d+)", expand=False)
    if df["patient_id"].isna().any():
        raise ValueError("Invalid patient IDs in training paths.")
    if split_path.is_file():
        patients = json.loads(split_path.read_text())
        if external_split and set(patients) != set(
            json.loads(Path(external_split).read_text())
        ):
            raise ValueError(
                "External split conflicts with the saved experiment partition."
            )
    elif external_split:
        patients = json.loads(Path(external_split).read_text())
        write_json(split_path, patients)
    else:
        flags = df.groupby("patient_id")[TAIL_2].apply(
            lambda group: bool(group.eq(1).any().any())
        )
        rng = np.random.RandomState(2026)
        patients = []
        for flag in [True, False]:
            subset = flags.index[flags.eq(flag)].to_numpy()
            if len(subset):
                n = max(1, int(round(len(subset) * 0.1)))
                patients.extend(
                    rng.choice(subset, size=min(n, len(subset)), replace=False).tolist()
                )
        patients = sorted(set(patients))
        write_json(split_path, patients)
    if (
        not isinstance(patients, list)
        or len(patients) != len(set(patients))
        or (not set(patients) <= set(df["patient_id"]))
    ):
        raise ValueError("Invalid development patient list.")
    mask = df["patient_id"].isin(patients)
    train, dev = (df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True))
    if (
        len(train) == 0
        or len(dev) == 0
        or set(train["patient_id"]) & set(dev["patient_id"])
    ):
        raise ValueError("Invalid or overlapping training/development partitions.")
    return (train, dev)


def train_one_run(
    root, train_df, dev_df, base_dir, strategy, lam, seed, args, weights, ece_mode
):
    seed_everything(seed)
    name = (
        f"phase1_{strategy}"
        if args.phase == "1"
        else (
            f"phase2a_lambda{lam}"
            if args.phase == "2a"
            else f"phase2b_lambda{lam}_seed{seed}"
        )
    )
    best, last = (
        root / "checkpoints" / f"{name}_best.pt",
        root / "checkpoints" / f"{name}_last.pt",
    )
    if (best.exists() or last.exists()) and (not args.resume):
        raise FileExistsError(
            f"Run already exists: {name}. Review it and use --resume explicitly."
        )
    state = (
        torch.load(last, map_location="cpu", weights_only=False)
        if last.exists()
        else None
    )
    expected_config = {
        "strategy": strategy,
        "lambda_max": lam,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "ece_mode": ece_mode,
        "preprocessing": "resize_normalize",
    }
    if state and state.get("run_config") != expected_config:
        raise ValueError(f"Resume configuration mismatch: {name}")
    if state and state.get("complete"):
        if not best.exists():
            raise FileNotFoundError(f"Completed run is missing best checkpoint: {name}")
        stored_metrics = torch.load(best, map_location="cpu", weights_only=False)[
            "dev_metrics"
        ]
        return {
            "run_name": name,
            "strategy": strategy,
            "lambda_max": lam,
            **{k: v for k, v in stored_metrics.items() if k != "dev_thresholds"},
        }
    if best.exists() and state is None:
        raise ValueError(f"Incomplete run; last training state is missing: {name}")
    if len(train_df) < args.batch_size:
        raise ValueError("Training partition is smaller than one complete batch.")
    transform = make_transforms(train=True, seed=seed)
    train_loader = DataLoader(
        CheXpertDataset(train_df, base_dir, transform, True, strategy),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        CheXpertDataset(dev_df, base_dir, make_transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = build_model(pretrained=state is None).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2
    )
    scaler = torch.amp.GradScaler("cuda")
    start, best_metric, no_improve, history = (1, -np.inf, 0, [])
    if state:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optim_state"])
        scheduler.load_state_dict(state["sched_state"])
        scaler.load_state_dict(state["scaler_state"])
        start, best_metric, no_improve, history = (
            state["epoch"] + 1,
            state["best_metric"],
            state["epochs_no_improve"],
            state["history"],
        )
        if state.get("rng_state"):
            random.setstate(state["rng_state"]["python"])
            np.random.set_state(state["rng_state"]["numpy"])
            torch.set_rng_state(state["rng_state"]["torch"])
            torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
    positive_weights = torch.tensor(
        make_pos_weight(lam, weights), dtype=torch.float32, device="cuda"
    )
    indices = [CLASS_NAMES_14.index(c) for c in FOCUS_6]
    for epoch in range(start, args.epochs + 1):
        model.train()
        norms, losses = ([], [])
        for images, targets, masks in tqdm(train_loader, desc=f"{name} epoch {epoch}"):
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                loss = masked_weighted_bce(
                    model(images.cuda()), targets.cuda(), masks.cuda(), positive_weights
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norms.append(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        y, scores, masks = infer(model, dev_loader, torch.device("cuda"), "amp")
        thresholds = optimize_thresholds_vectorized(y, scores, masks)
        suite = compute_metric_suite(y, scores, thresholds, masks)
        f1, auc = (suite.loc[FOCUS_6, "f1"].mean(), suite.loc[FOCUS_6, "auc"].mean())
        masked_ece = expected_calibration_error_masked(
            y[:, indices], scores[:, indices], masks[:, indices]
        )
        old_ece = unmasked_development_ece(y[:, indices], scores[:, indices])
        ece = old_ece if ece_mode == "unmasked" else masked_ece
        scheduler.step(f1)
        metrics = {
            "dev_macro_f1_focus6": f1,
            "dev_macro_auc_focus6": auc,
            "dev_ece_focus6": ece,
            "dev_ece_masked_focus6": masked_ece,
            "dev_ece_unmasked_focus6": old_ece,
            "ece_mode": ece_mode,
            "dev_thresholds": thresholds,
        }
        history.append(
            {
                "epoch": epoch,
                "loss": np.mean(losses),
                **{k: v for k, v in metrics.items() if k != "dev_thresholds"},
                "grad_norm_mean": np.mean(norms),
                "grad_norm_std": np.std(norms),
            }
        )
        pd.DataFrame(history).to_csv(root / "logs" / f"{name}_log.csv", index=False)
        if f1 > best_metric:
            best_metric, no_improve = (f1, 0)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dev_metrics": metrics,
                    "run_config": expected_config,
                },
                best,
            )
        else:
            no_improve += 1
        complete = no_improve >= 4 or epoch == args.epochs
        torch.save(
            {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "epochs_no_improve": no_improve,
                "history": history,
                "complete": complete,
                "run_config": expected_config,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            },
            last,
        )
        if complete:
            break
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    metrics = torch.load(best, map_location="cpu", weights_only=False)["dev_metrics"]
    return {
        "run_name": name,
        "strategy": strategy,
        "lambda_max": lam,
        **{k: v for k, v in metrics.items() if k != "dev_thresholds"},
    }


def train_phase(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA runtime. Evaluation can use CPU.")
    root = Path(args.output_root).resolve()
    marker = root / "experiment.json"
    if root.exists() and any(root.iterdir()) and (not marker.is_file()):
        raise FileExistsError("Choose an empty output folder for a new experiment.")
    if marker.exists() and (not args.resume):
        raise FileExistsError(
            "Experiment exists; review it and pass --resume explicitly."
        )
    for name in ["checkpoints", "logs", "results", "splits"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    csv_path = find_csv(args.data_dir, "train.csv")
    protocol = {
        "train_csv_sha256": sha256_file(csv_path),
        "image_size": 320,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "pipeline": "wvc-v1",
        "preprocessing": "resize_normalize",
    }
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError("Experiment-level configuration mismatch.")
    write_json(marker, protocol)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_df, dev_df = split_training(
        pd.read_csv(csv_path), root / "splits/dev_patients.json", args.split_file
    )
    write_json(
        root / "split_manifest.json",
        {
            "n_train": len(train_df),
            "n_dev": len(dev_df),
            "n_dev_patients": dev_df["patient_id"].nunique(),
            "split_sha256": sha256_file(root / "splits/dev_patients.json"),
        },
    )
    env = environment_manifest(torch.device("cuda"), ["amp"], args.batch_size)
    env["tf32"] = True
    env["resume_limit"] = (
        "Worker/augmentation RNG state is not restored; exact interrupted-run equivalence is not claimed."
    )
    write_json(root / f"environment_phase{args.phase}.json", env)
    weights = compute_w_tilde(train_df)
    weights.to_csv(root / "results/raw_positive_weights.csv")
    if args.phase == "1":
        scenarios = [
            (policy, 1, 2026)
            for policy in ["u_zeros", "u_ones", "u_ignore", "hybrid", "u_soft"]
        ]
        ece_mode, filename = ("unmasked", "phase1_uncertainty_ranking.csv")
    else:
        strategy = args.strategy
        lambdas = [1, 3, 5, 10, 20, "raw"] if args.phase == "2a" else LAMBDAS
        seeds = [2026] if args.phase == "2a" else SEEDS
        scenarios = [(strategy, lam, seed) for lam in lambdas for seed in seeds]
        ece_mode = "unmasked" if args.phase == "2a" else "masked"
        filename = (
            "phase2a_weight_screening.csv"
            if args.phase == "2a"
            else "phase2b_development_metrics.csv"
        )
    records = []
    for strategy, lam, seed in scenarios:
        if args.phase == "2a" and lam == 1:
            source_best = root / "checkpoints" / f"phase1_{strategy}_best.pt"
            source_last = root / "checkpoints" / f"phase1_{strategy}_last.pt"
            destination_best = root / "checkpoints" / "phase2a_lambda1_best.pt"
            destination_last = root / "checkpoints" / "phase2a_lambda1_last.pt"
            if (
                source_best.is_file()
                and source_last.is_file()
                and (not destination_best.exists())
                and (not destination_last.exists())
            ):
                state = torch.load(source_last, map_location="cpu", weights_only=False)
                expected = {
                    "strategy": strategy,
                    "lambda_max": 1,
                    "seed": seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "ece_mode": ece_mode,
                    "preprocessing": "resize_normalize",
                }
                if state.get("complete") and state.get("run_config") == expected:
                    shutil.copyfile(source_best, destination_best)
                    shutil.copyfile(source_last, destination_last)
        if args.phase == "2b" and seed == 2026:
            original = root / "checkpoints" / f"phase2a_lambda{lam}_best.pt"
            last_a = root / "checkpoints" / f"phase2a_lambda{lam}_last.pt"
            destination = root / "checkpoints" / f"phase2b_lambda{lam}_seed2026_best.pt"
            if original.is_file() and last_a.is_file():
                completed = torch.load(last_a, map_location="cpu", weights_only=False)
                if (
                    not completed.get("complete")
                    or completed["run_config"]["strategy"] != strategy
                ):
                    raise ValueError(
                        "Only a completed phase2a run using the same policy may be reused."
                    )
                if destination.exists():
                    if sha256_file(original) != sha256_file(destination):
                        raise ValueError(
                            "Existing phase2b checkpoint conflicts with phase2a reuse; no overwrite allowed."
                        )
                else:
                    shutil.copyfile(original, destination)
                metrics = torch.load(original, map_location="cpu", weights_only=False)[
                    "dev_metrics"
                ]
                records.append(
                    {
                        "run_name": destination.stem,
                        "strategy": strategy,
                        "lambda_max": lam,
                        "reused_from": original.name,
                        **{k: v for k, v in metrics.items() if k != "dev_thresholds"},
                    }
                )
                continue
        records.append(
            train_one_run(
                root,
                train_df,
                dev_df,
                csv_path.parent,
                strategy,
                lam,
                seed,
                args,
                weights,
                ece_mode,
            )
        )
    table = pd.DataFrame(records)
    if args.phase == "1":
        table = table.sort_values(
            ["dev_macro_f1_focus6", "dev_macro_auc_focus6", "dev_ece_focus6"],
            ascending=[False, False, True],
        )
    table.to_csv(root / "results" / filename, index=False)
    print(
        table.drop(columns=["dev_thresholds"], errors="ignore").to_string(index=False)
    )
    print(
        "Training phase complete. Official validation is evaluated only through the evaluate command."
    )


def make_figure(args):
    import matplotlib.pyplot as plt
    import seaborn as sns

    folder = Path(args.evaluation_dir)
    df = pd.read_csv(folder / "results_per_model.csv")
    df = df[df["precision"].eq(args.precision)]
    pairs = {(int(l), int(s)) for l, s in zip(df["lambda_max"], df["seed"])}
    if len(df) != 9 or pairs != {(l, s) for l in LAMBDAS for s in [2026, 123, 456]}:
        raise ValueError(
            "Figure requires all nine distinct lambda/seed pairs for the selected precision."
        )
    grouped = df.groupby("lambda_max")[["macro_f1_tail", "macro_f2_tail"]].agg(
        ["mean", "std"]
    )
    sns.set_theme(style="whitegrid", font_scale=0.7)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.75), dpi=450)
    for metric, label, marker in [
        ("macro_f1_tail", "Macro-F1", "o"),
        ("macro_f2_tail", "Macro-F2", "s"),
    ]:
        axes[0].errorbar(
            LAMBDAS,
            grouped.loc[LAMBDAS, (metric, "mean")],
            yerr=grouped.loc[LAMBDAS, (metric, "std")],
            label=label,
            marker=marker,
            capsize=3,
        )
    axes[0].set(
        xlabel="Positive-weight limit ($\\lambda_{max}$)",
        ylabel="Tail score",
        title="Tail detection across weight limits",
        xticks=LAMBDAS,
    )
    axes[0].legend()
    indices = [CLASS_NAMES_14.index(name) for name in TAIL_2]
    for lam, color in [(1, "#7f7f7f"), (5, "#d62728")]:
        with np.load(
            folder / f"predictions_{args.precision}_lambda{lam}_seed2026.npz",
            allow_pickle=False,
        ) as artifact:
            score = artifact["y_score"][:, indices]
            mask = artifact["masks"][:, indices].astype(bool)
            values = score[mask]
        sns.kdeplot(
            values,
            ax=axes[1],
            label=f"$\\lambda={lam}$",
            color=color,
            fill=True,
            alpha=0.3,
            cut=0,
        )
    axes[1].set(
        xlabel="Predicted probability",
        ylabel="Density",
        title="Tail scores in seed 2026",
        xlim=(0, 1),
    )
    axes[1].legend()
    fig.tight_layout()
    target = folder / f"figure_tail_{args.precision}.png"
    fig.savefig(target)
    plt.close(fig)
    print(target)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train on CheXpert using a new output folder."
    )
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-root", required=True)
    train.add_argument("--phase", choices=["1", "2a", "2b", "all"], default="2b")
    train.add_argument(
        "--strategy",
        choices=["u_soft", "u_zeros", "u_ones", "u_ignore", "hybrid"],
        default="u_soft",
    )
    train.add_argument("--split-file")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate the models produced by training."
    )
    evaluation.add_argument("--data-dir", required=True)
    evaluation.add_argument("--experiment-root", required=True)
    evaluation.add_argument("--output-root", default="runs/evaluation")
    evaluation.add_argument(
        "--precision", choices=["fp32", "amp", "both"], default="fp32"
    )
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--workers", type=int, default=0)
    evaluation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluation.add_argument("--bootstrap", type=int, default=2000)
    figure = commands.add_parser("figures", help="Plot saved evaluation results.")
    figure.add_argument("--evaluation-dir", required=True)
    figure.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    args = parser.parse_args(argv)
    if args.command in {"train", "evaluate"} and (
        args.batch_size < 1 or args.workers < 0
    ):
        parser.error("Batch size must be positive and workers must be nonnegative.")
    if args.command == "train":
        if args.epochs < 1:
            parser.error("Epoch count must be positive.")
        phases = ["1", "2a", "2b"] if args.phase == "all" else [args.phase]
        for index, phase in enumerate(phases):
            args.phase = phase
            args.resume = args.resume or index > 0
            train_phase(args)
    elif args.command == "evaluate":
        if args.bootstrap < 1:
            parser.error("Bootstrap attempts must be positive.")
        evaluate(args)
    else:
        make_figure(args)


if __name__ == "__main__":
    main()
