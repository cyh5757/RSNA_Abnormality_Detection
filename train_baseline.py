# %% Environment: run on Kaggle with GPU and Internet enabled.
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import subprocess
import sys

missing = [package for module, package in [("pydicom", "pydicom"), ("gdcm", "python-gdcm")]
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

SEED = 42
EPOCHS = 100
# 방향별 특징을 항상 같은 자리에 연결해 분류기가 촬영 방향을 구분하도록 한다.
PLANES = ("Sagittal", "Coronal", "Axial")
torch.manual_seed(SEED)
np.random.seed(SEED)
assert torch.cuda.is_available(), "Enable a GPU in Kaggle Notebook Settings."
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

candidates = [Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")]
DATA = next((path for path in candidates if (path / "train.csv").exists()), None)
assert DATA is not None, "Attach the RSNA Knee Abnormality Detection competition data."
OUT = Path("/kaggle/working")
OUT.mkdir(exist_ok=True)

# %% Read metadata; missing labels remain unknown, never negative.
train = pd.read_csv(DATA / "train.csv")
series = pd.read_csv(DATA / "train_series.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")
labels = sample.columns.drop("StudyInstanceUID").tolist()
assert len(labels) == 12 and train.StudyInstanceUID.is_unique
assert series.SeriesInstanceUID.is_unique
assert set(series.StudyInstanceUID) <= set(train.StudyInstanceUID)
assert train[labels].stack().isin([0, 1]).all()
gold = train.loc[train[labels].notna().all(axis=1)].copy()
assert len(gold) >= 8, "Too few completely labeled studies for this pilot."
print({"studies": len(train), "series": len(series), "fully_labeled": len(gold),
       "reports": int(train.Report.notna().sum())})
display(pd.DataFrame({"labeled": train[labels].count(), "positive": train[labels].sum()}))

# Split studies before fitting any normalization or classifier.
gold["split_key"] = gold.StudyInstanceUID.map(
    lambda uid: hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest())
gold = gold.sort_values("split_key").reset_index(drop=True)
n_valid = math.ceil(len(gold) * 0.25)
valid_idx = np.arange(n_valid)
train_idx = np.arange(n_valid, len(gold))
# 정답은 슬라이스별 이미지가 아니라 gold의 각 검사에 붙은 12개 소견의 0/1 벡터다.
y = gold[labels].to_numpy(dtype=np.float32)
for idx in (train_idx, valid_idx):
    assert ((y[idx].sum(axis=0) > 0) & (y[idx].sum(axis=0) < len(idx))).all(), \
        "Each label needs positive and negative studies in both splits."
assert set(gold.iloc[train_idx].StudyInstanceUID).isdisjoint(gold.iloc[valid_idx].StudyInstanceUID)
split = gold[["StudyInstanceUID"]].copy()
split["split"] = np.where(np.arange(len(gold)) < n_valid, "validation", "train")
split.to_csv(OUT / "split.csv", index=False)
print(f"Train: {len(train_idx)} studies; validation: {len(valid_idx)} studies")

# %% DICOM decoding and frozen image features.
weights = ResNet18_Weights.IMAGENET1K_V1
transform = weights.transforms()
encoder = resnet18(weights=weights)
encoder.fc = nn.Identity()
encoder = encoder.to(device).eval()
encoder.requires_grad_(False)


def ordered_slices(folder):
    # 이미 2D 단면으로 저장된 DICOM들을 촬영 시간이나 파일명 대신 공간적 위치로 정렬한다.
    paths = sorted(folder.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No DICOM slices in {folder}")
    headers = [pydicom.dcmread(path, stop_before_pixels=True,
                              specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"])
               for path in paths]
    if all(hasattr(h, "ImagePositionPatient") and hasattr(h, "ImageOrientationPatient") for h in headers):
        # 영상의 행·열 방향으로 단면의 법선을 구하고, 위치를 그 축에 투영해 앞뒤 순서를 얻는다.
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        assert np.linalg.norm(normal) > 0.9, "Invalid DICOM orientation."
        positions = [float(np.dot(np.asarray(h.ImagePositionPatient, dtype=float), normal)) for h in headers]
    elif all(hasattr(h, "InstanceNumber") for h in headers):
        # 위치 정보가 없을 때의 대안이며, 이 번호만으로 실제 슬라이스 간격은 알 수 없다.
        positions = [float(h.InstanceNumber) for h in headers]
    else:
        raise ValueError(f"Missing slice position and InstanceNumber: {folder}")
    return [paths[i] for i in np.argsort(positions, kind="stable")]


def image_tensor(path):
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
    # ImageNet 입력에 맞춘 흑백 영상 복제이며, 서로 다른 위치의 슬라이스 3장을 쌓는 것이 아니다.
    tensor = torch.from_numpy(pixels).float().unsqueeze(0).repeat(3, 1, 1)
    return transform(tensor)


@torch.inference_mode()
def study_features(study_uid, series_table, image_root):
    rows = series_table.loc[series_table.StudyInstanceUID == study_uid]
    assert len(rows), f"No series metadata for {study_uid}"
    features = []
    used_planes = 0
    for plane in PLANES:
        # 한 검사에서 방향별 시리즈 하나를 고르며, fluid-sensitive·지방억제 시리즈를 우선한다.
        choices = rows.loc[rows.Anatomical_Plane == plane].sort_values(
            ["Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"], ascending=[False, False, True])
        if choices.empty:
            features.append(torch.zeros(512))
            continue
        row = choices.iloc[0]
        paths = ordered_slices(image_root / study_uid / row.SeriesInstanceUID)
        # shortcut: 방향별 한 시리즈의 최대 3장만 사용한다. 성능 개선 시 영상 범위와 위치 정보 활용을 확장한다.
        # 정렬된 인덱스의 25·50·75% 지점을 고른다. 검사 간 같은 해부학적 위치를 보장하지는 않는다.
        indices = np.unique(np.rint(np.linspace(0.25, 0.75, 3) * (len(paths) - 1)).astype(int))
        batch = torch.stack([image_tensor(paths[i]) for i in indices]).to(device)
        # 각 슬라이스를 독립적으로 인코딩한 뒤 평균내므로, 같은 장들의 입력 순서를 바꿔도 특징은 같다.
        # 정렬은 슬라이스 선택에만 쓰이며, 위치·간격이나 인접 슬라이스의 연결 관계는 모델에 전달하지 않는다.
        features.append(encoder(batch).mean(dim=0).cpu())
        used_planes += 1
    assert used_planes, f"No recognized anatomical planes for {study_uid}"
    # 슬라이스 순서는 평균에서 사라지지만, 시상·관상·축상 특징의 자리(각 512개)는 유지한다.
    result = torch.cat(features)
    assert result.shape == (1536,) and torch.isfinite(result).all()
    return result


# y를 만든 gold와 동일한 검사 순서로 특징을 쌓아 X[i]와 y[i]의 StudyInstanceUID를 맞춘다.
X = torch.stack([study_features(uid, series, DATA / "train_series")
                 for uid in tqdm(gold.StudyInstanceUID, desc="MRI features: labeled studies")])
assert X.shape == (len(gold), 1536) and X.std() > 0
torch.save({"features": X, "study_ids": gold.StudyInstanceUID.tolist()}, OUT / "gold_features.pt")

# %% Train a small classification head; the image encoder stays frozen.
mean = X[train_idx].mean(dim=0)
std = X[train_idx].std(dim=0).clamp_min(0.01)
X = ((X - mean) / std).to(device)
targets = torch.from_numpy(y).to(device)
head = nn.Linear(1536, len(labels)).to(device)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.1)
positives = targets[train_idx].sum(dim=0)
criterion = nn.BCEWithLogitsLoss(pos_weight=(len(train_idx) - positives) / positives)
best_loss = float("inf")
best_state = None
history = []
for epoch in range(1, EPOCHS + 1):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    # 각 검사의 12개 예측과 그 검사의 정답을 비교한다. 학습 영상과 테스트 영상을 서로 비교하지 않는다.
    loss = criterion(head(X[train_idx]), targets[train_idx])
    assert torch.isfinite(loss), "Non-finite training loss."
    loss.backward()
    optimizer.step()
    head.eval()
    with torch.inference_mode():
        logits = head(X[valid_idx])
        # 검증도 해당 검사의 정답과 비교하며, 공개되지 않은 실제 테스트 정답은 여기서 사용하지 않는다.
        val_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets[valid_idx]).item()
        probabilities = logits.sigmoid().cpu().numpy()
    auc = roc_auc_score(y[valid_idx], probabilities, average="macro")
    history.append({"epoch": epoch, "train_loss": loss.item(), "valid_loss": val_loss, "macro_auc": auc})
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        best_epoch = epoch
    if epoch == 1 or epoch % 20 == 0:
        print(history[-1])

assert best_state is not None
head.load_state_dict(best_state)
head.eval()
with torch.inference_mode():
    validation_predictions = head(X[valid_idx]).sigmoid().cpu().numpy()
assert np.isfinite(validation_predictions).all()
metrics = {label: float(roc_auc_score(y[valid_idx, i], validation_predictions[:, i]))
           for i, label in enumerate(labels)}
metrics["macro_auc"] = float(np.mean(list(metrics.values())))
metrics["best_epoch"] = best_epoch
metrics["train_studies"] = len(train_idx)
metrics["validation_studies"] = len(valid_idx)
metrics["note"] = "Pilot only: 15 validation studies; not a leaderboard estimate."
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
predictions = pd.DataFrame(validation_predictions, columns=labels)
predictions.insert(0, "StudyInstanceUID", gold.iloc[valid_idx].StudyInstanceUID.to_numpy())
predictions.to_csv(OUT / "validation_predictions.csv", index=False)
torch.save({"encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
            "head": best_state, "mean": mean, "std": std, "labels": labels,
            "planes": list(PLANES), "seed": SEED, "best_epoch": best_epoch}, OUT / "baseline.pt")
print(json.dumps(metrics, indent=2))
print("Saved baseline.pt, metrics.json, history.csv, split.csv and validation_predictions.csv")
print("Training pilot complete. An offline inference notebook is needed for competition submission.")
