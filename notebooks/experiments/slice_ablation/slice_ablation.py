"""E00/E01: controlled three-versus-five slice experiment; not a submission script."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch import nn


PLANES = ("Sagittal", "Coronal", "Axial")
SEEDS = (42, 2026, 3407)
EPOCHS = 5


def slice_indices(length, count):
    if length < 1 or count < 1:
        raise ValueError("A series and the requested slice count must be nonempty.")
    return np.unique(np.rint(np.linspace(.25, .75, count) * (length - 1)).astype(int))


def align_features(cache, study_ids):
    cached_ids = cache["study_ids"]
    x = cache["features"].detach().cpu().float()
    if (len(set(cached_ids)) != len(cached_ids) or len(set(study_ids)) != len(study_ids)
            or set(cached_ids) != set(study_ids)):
        raise ValueError("Feature cache must contain exactly the requested unique study IDs.")
    if x.shape != (len(cached_ids), 1536) or not torch.isfinite(x).all():
        raise ValueError("Expected finite, unnormalized ResNet18 features of shape (N, 1536).")
    row = {uid: i for i, uid in enumerate(cached_ids)}
    return x[[row[uid] for uid in study_ids]]


def make_folds(study_ids, y):
    if (len(set(study_ids)) != len(study_ids) or y.ndim != 2
            or len(y) != len(study_ids) or not np.isin(y, [0, 1]).all()):
        raise ValueError("Expected unique study IDs and complete binary targets.")
    order = np.argsort(study_ids)
    folds = np.full(len(study_ids), -1, dtype=int)
    for fold, (train, valid) in enumerate(KFold(3, shuffle=True, random_state=42).split(order)):
        for indices in (order[train], order[valid]):
            positives = y[indices].sum(axis=0)
            if not ((positives > 0) & (positives < len(indices))).all():
                raise ValueError(f"Fold {fold} lacks a positive or negative example for a label.")
        folds[order[valid]] = fold
    return folds


def fit_head(x_train, y_train, x_valid, seed):
    torch.manual_seed(seed)
    # Validation labels are deliberately absent: neither fitting nor epoch selection uses them.
    mean = x_train.mean(0)
    std = x_train.std(0).clamp_min(.01)
    train = (x_train - mean) / std
    valid = (x_valid - mean) / std
    positives = y_train.sum(0)
    if not ((positives > 0) & (positives < len(y_train))).all():
        raise ValueError("Every training label needs both classes.")
    head = nn.Linear(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(y_train) - positives) / positives)
    history = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train), y_train)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "train_loss": loss.item()})
    head.eval()
    with torch.inference_mode():
        logits = head(valid)
    state = {"head": head.state_dict(), "mean": mean, "std": std}
    return logits, history, state


def score(y, logits, labels):
    probabilities = logits.sigmoid().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions.")
    per_label = roc_auc_score(y, probabilities, average=None)
    return {"macro_auc": float(np.mean(per_label)),
            "bce": nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(y)).item(),
            "per_label_auc": dict(zip(labels, per_label.tolist()))}


def evaluate(features, y, ids, labels, folds, out, seeds=SEEDS):
    out.mkdir(parents=True, exist_ok=True)
    targets = torch.from_numpy(y)
    runs, predictions, histories = [], [], []
    for seed in seeds:
        logits = torch.full(targets.shape, float("nan"))
        visits = np.zeros(len(y), dtype=int)
        fold_metrics = []
        for fold in range(3):
            train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            predicted, history, state = fit_head(features[train], targets[train], features[valid], seed)
            logits[valid] = predicted
            visits[valid] += 1
            fold_metrics.append({"fold": fold, "train_studies": len(train),
                                 "valid_studies": len(valid), **score(y[valid], predicted, labels)})
            histories.extend({"seed": seed, "fold": fold, **row} for row in history)
            torch.save({**state, "seed": seed, "fold": fold, "labels": labels,
                        "train_ids": [ids[i] for i in train], "valid_ids": [ids[i] for i in valid]},
                       out / f"head_seed{seed}_fold{fold}.pt")
        if not (visits == 1).all() or not torch.isfinite(logits).all():
            raise ValueError("Every study must receive exactly one finite OOF prediction per seed.")
        runs.append({"seed": seed, **score(y, logits, labels), "folds": fold_metrics,
                     "fold_macro_auc_mean": float(np.mean([row["macro_auc"] for row in fold_metrics]))})
        frame = pd.DataFrame(logits.sigmoid().numpy(), columns=labels)
        frame.insert(0, "fold", folds)
        frame.insert(0, "seed", seed)
        frame.insert(0, "StudyInstanceUID", ids)
        predictions.append(frame)
    aucs = [run["macro_auc"] for run in runs]
    metrics = {"runs": runs, "macro_auc_mean": float(np.mean(aucs)),
               "macro_auc_seed_std": float(np.std(aucs)),
               "fold_macro_auc_mean": float(np.mean([r["fold_macro_auc_mean"] for r in runs])),
               "bce_mean": float(np.mean([r["bce"] for r in runs])),
               "note": "Development CV on 58 gold studies. Seed SD is not a confidence interval."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
    return metrics


# Baseline DICOM preprocessing is frozen here; the 3-slice cache parity check guards this copy.
def ordered_slices(folder):
    import pydicom
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM slices in {folder}")
    headers = [pydicom.dcmread(path, stop_before_pixels=True,
                              specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"])
               for path in paths]
    if all(hasattr(h, "ImagePositionPatient") and hasattr(h, "ImageOrientationPatient") for h in headers):
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        assert np.linalg.norm(normal) > .9, "Invalid DICOM orientation."
        positions = [float(np.dot(np.asarray(h.ImagePositionPatient, dtype=float), normal)) for h in headers]
    elif all(hasattr(h, "InstanceNumber") for h in headers):
        positions = [float(h.InstanceNumber) for h in headers]
    else:
        raise ValueError(f"Missing slice position and InstanceNumber: {folder}")
    return [paths[i] for i in np.argsort(positions, kind="stable")]


def image_tensor(path, transform):
    import pydicom
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    assert pixels.ndim == 2 and np.isfinite(pixels).all(), f"Invalid MRI slice: {path}"
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = -pixels
    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())
    if high <= low:
        raise ValueError(f"Constant-valued MRI slice: {path}")
    pixels = np.clip((pixels - low) / (high - low), 0, 1)
    return transform(torch.from_numpy(pixels).float().unsqueeze(0).repeat(3, 1, 1))


@torch.inference_mode()
def study_features(uid, series, image_root, encoder, transform, count):
    rows = series.loc[series.StudyInstanceUID == uid]
    assert len(rows), f"No series metadata for {uid}"
    features, audit = [], []
    used_planes = 0
    for plane in PLANES:
        choices = rows.loc[rows.Anatomical_Plane == plane].sort_values(
            ["Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"], ascending=[False, False, True])
        if choices.empty:
            features.append(torch.zeros(512))
            audit.append({"StudyInstanceUID": uid, "plane": plane, "used_slices": 0})
            continue
        row = choices.iloc[0]
        paths = ordered_slices(image_root / uid / row.SeriesInstanceUID)
        indices = slice_indices(len(paths), count)
        batch = torch.stack([image_tensor(paths[i], transform) for i in indices]).cuda()
        features.append(encoder(batch).mean(0).cpu())
        used_planes += 1
        audit.append({"StudyInstanceUID": uid, "plane": plane, "SeriesInstanceUID": row.SeriesInstanceUID,
                      "available_slices": len(paths), "used_slices": len(indices),
                      "indices": indices.tolist(), "files": [paths[i].name for i in indices]})
    assert used_planes, f"No recognized anatomical planes for {uid}"
    result = torch.cat(features)
    assert result.shape == (1536,) and torch.isfinite(result).all()
    return result, audit


def extract_features(ids, series, data, source, out, counts):
    from torchvision.models import ResNet18_Weights, resnet18
    from tqdm.auto import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError("Run feature extraction on Kaggle GPU; use --cached-only locally.")
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    checkpoint = torch.load(source / "baseline.pt", map_location="cpu", weights_only=True)
    assert checkpoint["planes"] == list(PLANES)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder = encoder.cuda().eval().requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    variants, timings = {}, {}
    for count in counts:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        features, audit = [], []
        for uid in tqdm(ids, desc=f"{count} slices per plane"):
            feature, rows = study_features(uid, series, data / "train_series", encoder, transform, count)
            features.append(feature)
            audit.extend(rows)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        x = torch.stack(features)
        variants[count] = x
        torch.save({"features": x, "study_ids": ids, "slices_per_plane": count},
                   out / f"features_{count}.pt")
        pd.DataFrame(audit).to_csv(out / f"slice_audit_{count}.csv", index=False)
        timings[str(count)] = {"feature_seconds": elapsed, "seconds_per_study": elapsed / len(ids),
                               "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2}
        if count == 3:
            reference = align_features(torch.load(source / "gold_features.pt", weights_only=True), ids)
            torch.testing.assert_close(x, reference, rtol=1e-4, atol=1e-4)
            timings["baseline_parity_max_abs_error"] = (x - reference).abs().max().item()
    torch.save(checkpoint["encoder"], out / "encoder.pt")
    return variants, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/slice_ablation"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = args.data
    if data is None:
        data = next((p for p in [Path("/kaggle/input/rsna-knee-abnormality-detection"),
                     Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")]
                     if (p / "train.csv").is_file()), None)
    source = args.source
    if source is None:
        candidates = list(Path("/kaggle/input").glob("*/baseline.pt"))
        candidates += list(Path("/kaggle/input").glob("notebooks/*/*/baseline.pt"))
        if len(candidates) == 1:
            source = candidates[0].parent
    if data is None or source is None:
        raise FileNotFoundError("Attach competition data and baseline training version 4, or set --data/--source.")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Keep previous results; choose an empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / "train.csv")
    labels = pd.read_csv(data / "sample_submission.csv", nrows=0).columns.drop("StudyInstanceUID").tolist()
    assert len(labels) == 12 and train.StudyInstanceUID.is_unique
    assert train[labels].stack().dropna().isin([0, 1]).all()
    gold = train.loc[train[labels].notna().all(axis=1)].sort_values("StudyInstanceUID").reset_index(drop=True)
    ids = gold.StudyInstanceUID.tolist()
    assert len(ids) == 58, "Gold population changed: review the locked experiment protocol."
    y = gold[labels].to_numpy(dtype=np.float32)
    folds = make_folds(ids, y)
    pd.DataFrame({"StudyInstanceUID": ids, "fold": folds}).to_csv(args.out / "folds.csv", index=False)
    if not args.cached_only:
        missing = [pkg for module, pkg in [("pydicom", "pydicom"), ("gdcm", "python-gdcm")]
                   if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    config = {"protocol": "e00-e01-v1", "gold_studies": len(ids), "labels": labels,
              "fold_seed": 42, "initialization_seeds": list(SEEDS), "epochs": EPOCHS,
              "slice_counts": [3] if args.cached_only else [3, 5], "slice_range": [.25, .75],
              "encoder": "baseline-v4 frozen ResNet18 ImageNet1K_V1", "pooling": "plane mean + concat",
              "loss": "BCEWithLogitsLoss(train_negative/train_positive)",
              "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .1},
              "head_device": "cpu", "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ["numpy", "pandas", "torch", "torchvision", "scikit-learn", "pydicom", "python-gdcm"]},
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [data / "train.csv", data / "train_series.csv", source / "baseline.pt",
                          source / "gold_features.pt", Path(__file__)]}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.cached_only:
        variants = {3: align_features(torch.load(source / "gold_features.pt", weights_only=True), ids)}
        timings = {"note": "Cached features only; no image extraction timing."}
    else:
        series = pd.read_csv(data / "train_series.csv")
        assert series.SeriesInstanceUID.is_unique
        assert set(series.StudyInstanceUID) <= set(train.StudyInstanceUID)
        variants, timings = extract_features(ids, series, data, source, args.out, (3, 5))
    (args.out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    results = {count: evaluate(x, y, ids, labels, folds, args.out / f"slices_{count}")
               for count, x in variants.items()}
    comparison = [{"slices_per_plane": count, **{key: result[key] for key in
                   ["macro_auc_mean", "macro_auc_seed_std", "fold_macro_auc_mean", "bce_mean"]}}
                  for count, result in results.items()]
    pd.DataFrame(comparison).to_csv(args.out / "comparison.csv", index=False)
    if 5 in results:
        pairs = [{"seed": a["seed"], "auc_3": a["macro_auc"], "auc_5": b["macro_auc"],
                  "delta_auc": b["macro_auc"] - a["macro_auc"],
                  "delta_fold_mean_auc": b["fold_macro_auc_mean"] - a["fold_macro_auc_mean"],
                  "delta_bce": b["bce"] - a["bce"]}
                 for a, b in zip(results[3]["runs"], results[5]["runs"])]
        pd.DataFrame(pairs).to_csv(args.out / "paired_deltas.csv", index=False)
    print(json.dumps({"comparison": comparison, "timings": timings}, indent=2), flush=True)
    print("Experiment complete; existing production submission is unchanged.", flush=True)


if __name__ == "__main__":
    main()
