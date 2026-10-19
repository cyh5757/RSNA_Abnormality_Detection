"""Shared source embedded into each standalone experiment notebook."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

PLANES = ("Sagittal", "Coronal", "Axial")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEEDS = (42, 2026, 3407)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_bundle(path, expected_config):
    path = Path(path).resolve()
    manifest = json.loads((path / "assets.json").read_text(encoding="utf-8"))
    if manifest["experiment"] != expected_config:
        raise ValueError("Wrong experiment bundle: attach this experiment's completed training version.")
    files = manifest["files"]
    wheels = {p.relative_to(path).as_posix() for p in (path / "wheels").glob("*.whl")}
    declared_wheels = {name for name in files if name.startswith("wheels/") and name.endswith(".whl")}
    if (not {"encoder.ts", "model.pt"}.issubset(files) or wheels != declared_wheels
            or len(wheels) != 2
            or not any(name.startswith("wheels/pydicom-3.0.2-") for name in wheels)
            or not any(name.startswith("wheels/python_gdcm-3.2.6-") for name in wheels)):
        raise ValueError("Bundle must include hashes for both models and exactly the two required wheels.")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or not target.is_file() or file_hash(target) != digest:
            raise ValueError(f"Invalid or corrupted bundle file: {name}")
    return manifest


def find_data(input_root=Path("/kaggle/input")):
    for path in (input_root / "rsna-knee-abnormality-detection",
                 input_root / "competitions/rsna-knee-abnormality-detection"):
        if (path / "test.csv").is_file():
            return path
    raise FileNotFoundError("Attach the RSNA Knee competition data.")


def find_bundle(input_root=Path("/kaggle/input")):
    paths = list(input_root.glob("*/run/bundle/assets.json"))
    paths += list(input_root.glob("notebooks/*/*/run/bundle/assets.json"))
    paths += list(input_root.glob("kernels/*/*/run/bundle/assets.json"))
    if len(paths) != 1:
        raise ValueError("Attach exactly one completed experiment training output.")
    return paths[0].parent


def make_transform(config):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms._presets import ImageClassification
    return ImageClassification(crop_size=config["crop"], resize_size=config["resize"],
                               interpolation=InterpolationMode(config["interpolation"]), antialias=True)


def ordered_slices(folder):
    import pydicom
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM slices in {folder}")
    headers = [pydicom.dcmread(p, stop_before_pixels=True,
               specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"]) for p in paths]
    if all(hasattr(h, "ImagePositionPatient") and hasattr(h, "ImageOrientationPatient") for h in headers):
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if np.linalg.norm(normal) <= .9:
            raise ValueError("Invalid DICOM orientation.")
        coordinates = np.asarray([np.dot(np.asarray(h.ImagePositionPatient, dtype=float), normal) for h in headers])
    elif all(hasattr(h, "InstanceNumber") for h in headers):
        coordinates = np.asarray([float(h.InstanceNumber) for h in headers])
    else:
        raise ValueError(f"No slice position or InstanceNumber: {folder}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Non-finite slice positions.")
    order = np.argsort(coordinates, kind="stable")
    coordinates = coordinates[order]
    span = np.ptp(coordinates)
    positions = (coordinates - coordinates.min()) / span if span > 0 else np.full(len(paths), .5)
    return [paths[i] for i in order], positions.astype(np.float32)


def image_tensor(path, transform):
    import pydicom
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError(f"Invalid MRI slice: {path}")
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = -pixels
    low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        low, high = float(pixels.min()), float(pixels.max())
    if high <= low:
        raise ValueError(f"Constant MRI slice: {path}")
    pixels = np.clip((pixels - low) / (high - low), 0, 1)
    return transform(torch.from_numpy(pixels).float().unsqueeze(0).repeat(3, 1, 1))


@torch.inference_mode()
def study_tokens(uid, series, image_root, encoder, transform, config, device):
    count, dim = config["slices"], config["feature_dim"]
    features = torch.zeros(3, count, dim)
    mask = torch.zeros(3, count, dtype=torch.bool)
    positions = torch.zeros(3, count)
    rows = series.loc[series.StudyInstanceUID == uid]
    for plane_index, plane in enumerate(PLANES):
        choices = rows.loc[rows.Anatomical_Plane == plane].sort_values(
            ["Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"], ascending=[False, False, True])
        if choices.empty:
            continue
        paths, coords = ordered_slices(image_root / uid / choices.iloc[0].SeriesInstanceUID)
        indices = np.unique(np.rint(np.linspace(.25, .75, count) * (len(paths) - 1)).astype(int))
        batch = torch.stack([image_tensor(paths[i], transform) for i in indices]).to(device)
        output = encoder(batch).cpu().float()
        if output.shape != (len(indices), dim) or not torch.isfinite(output).all():
            raise ValueError("Unexpected encoder features.")
        features[plane_index, :len(indices)] = output
        mask[plane_index, :len(indices)] = True
        positions[plane_index, :len(indices)] = torch.from_numpy(coords[indices])
    if not mask.any():
        raise ValueError(f"No supported MRI planes for study {uid}")
    return features, mask, positions


def pooled_features(tokens, mask):
    clean = torch.where(mask[..., None], tokens, 0.)
    return (clean.sum(2) / mask.sum(2).clamp_min(1)[..., None]).flatten(1)


class StudyHead(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__()
        self.mode = config["aggregation"]
        self.position = config["position"]
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        if self.mode == "mean":
            self.classifier = nn.Linear(3 * config["feature_dim"], 12)
        elif self.mode == "attention":
            self.project = nn.Linear(config["feature_dim"], 64)
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0., batch_first=True, norm_first=True)
            self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.classifier = nn.Linear(3 * 64, 12)
            # Initialize common layers identically before adding optional plane parameters.
            self.plane_embedding = nn.Parameter(torch.zeros(3, 64)) if config["plane_embedding"] else None
            self.register_buffer("frequencies", torch.exp(torch.arange(0, 64, 2) * (-math.log(10000.) / 64)))
        else:
            raise ValueError(f"Unknown aggregation: {self.mode}")

    def forward(self, tokens, mask, positions):
        if self.mode == "mean":
            return self.classifier((pooled_features(tokens, mask) - self.mean) / self.std)
        clean = torch.where(mask[..., None], tokens, 0.)
        x = self.project((clean - self.mean) / self.std)
        if self.position:
            angles = positions[..., None] * (2 * math.pi) * self.frequencies
            x = x + torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        if self.plane_embedding is not None:
            x = x + self.plane_embedding[None, :, None, :]
        b, planes, slices, dim = x.shape
        flat_mask = mask.reshape(b * planes, slices)
        safe_mask = flat_mask.clone()
        safe_mask[~safe_mask.any(1), 0] = True
        x = self.attention(x.reshape(b * planes, slices, dim), src_key_padding_mask=~safe_mask)
        x = torch.where(flat_mask[..., None], x, 0.)
        vectors = x.sum(1) / flat_mask.sum(1).clamp_min(1)[:, None]
        return self.classifier(vectors.reshape(b, planes * dim))


def fit_model(tokens, mask, positions, y, train_idx, seed, config):
    torch.manual_seed(seed)
    x, m, p, target = tokens[train_idx], mask[train_idx], positions[train_idx], y[train_idx]
    statistics = pooled_features(x, m) if config["aggregation"] == "mean" else x[m]
    mean, std = statistics.mean(0), statistics.std(0).clamp_min(.01)
    positives = target.sum(0)
    if not torch.isfinite(statistics).all() or not ((positives > 0) & (positives < len(target))).all():
        raise ValueError("Finite training features and both classes per label are required.")
    head = StudyHead(config, mean, std)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(target) - positives) / positives)
    history = []
    head.train()
    for epoch in range(1, config["epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(x, m, p), target)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "train_loss": loss.item()})
    return head.eval(), history


def make_folds(ids, y):
    from sklearn.model_selection import KFold
    if len(set(ids)) != len(ids) or len(y) != len(ids) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected unique IDs and complete binary labels.")
    order = np.argsort(ids)
    folds = np.full(len(ids), -1, dtype=int)
    for fold, (train, valid) in enumerate(KFold(3, shuffle=True, random_state=42).split(order)):
        for index in (order[train], order[valid]):
            count = y[index].sum(0)
            if not ((count > 0) & (count < len(index))).all():
                raise ValueError("Each fold/label needs positives and negatives.")
        folds[order[valid]] = fold
    return folds


def metric_scores(y, logits):
    from sklearn.metrics import roc_auc_score
    probabilities = logits.sigmoid().numpy()
    aucs = roc_auc_score(y, probabilities, average=None)
    return {"macro_auc": float(np.mean(aucs)), "per_label_auc": dict(zip(LABELS, aucs.tolist())),
            "bce": nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y)).item()}


def save_submission(ids, probabilities, sample_columns, target):
    probabilities = np.asarray(probabilities)
    if (not ids or len(set(ids)) != len(ids) or list(sample_columns) != ["StudyInstanceUID", *LABELS]
            or probabilities.shape != (len(ids), 12) or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Invalid submission IDs, label order, shape or probabilities.")
    frame = pd.DataFrame(probabilities, columns=LABELS)
    frame.insert(0, "StudyInstanceUID", ids)
    target = Path(target)
    temporary = target.with_suffix(".csv.part")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    return frame
