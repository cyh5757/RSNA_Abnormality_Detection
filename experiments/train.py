"""Training entry point; embedded after the core and one backbone constructor."""
import importlib.metadata
import platform
import subprocess
import sys
import time

from experiments.core import *


def export_encoder(encoder, config, target):
    torch.backends.mha.set_fastpath_enabled(False)
    encoder = encoder.cpu().eval().requires_grad_(False)
    size = config["crop"]
    example = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        # Fixed H/W is part of the bundle contract; batch sizes are independently checked below.
        traced = torch.jit.trace(encoder, example, check_trace=False)
        for batch_size in range(1, config["slices"] + 1):
            probe = torch.rand(batch_size, 3, size, size)
            actual, expected = traced(probe), encoder(probe)
            print(f"Encoder export batch={batch_size}: max_abs_error={(actual - expected).abs().max().item():.8g}", flush=True)
            # Float32 eager/JIT execution can round differently near zero, as observed in DINOv2.
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    traced.save(str(target))
    loaded = torch.jit.load(str(target), map_location="cpu").eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(probe), encoder(probe), rtol=1e-4, atol=1e-4)
    return loaded


def evaluate_cv(tokens, mask, positions, y, ids, folds, config, out):
    runs, histories, predictions = [], [], []
    for seed in SEEDS:
        oof = torch.full((len(ids), 12), float("nan"))
        fold_scores = []
        for fold in range(3):
            train_index, valid_index = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            head, history = fit_model(tokens, mask, positions, torch.from_numpy(y), train_index, seed, config)
            with torch.inference_mode():
                logits = head(tokens[valid_index], mask[valid_index], positions[valid_index])
            oof[valid_index] = logits
            fold_scores.append({"fold": fold, **metric_scores(y[valid_index], logits)})
            histories.extend({"seed": seed, "fold": fold, **row} for row in history)
        if not torch.isfinite(oof).all():
            raise ValueError("Incomplete OOF predictions.")
        runs.append({"seed": seed, **metric_scores(y, oof), "folds": fold_scores})
        prediction = pd.DataFrame(oof.sigmoid().numpy(), columns=LABELS)
        prediction.insert(0, "fold", folds)
        prediction.insert(0, "seed", seed)
        prediction.insert(0, "StudyInstanceUID", ids)
        predictions.append(prediction)
    metrics = {"experiment": config, "runs": runs,
               "macro_auc_mean": float(np.mean([r["macro_auc"] for r in runs])),
               "macro_auc_seed_std": float(np.std([r["macro_auc"] for r in runs])),
               "fold_auc_mean": float(np.mean([f["macro_auc"] for r in runs for f in r["folds"]])),
               "bce_mean": float(np.mean([r["bce"] for r in runs])),
               "note": "Development CV, 58 studies; seed SD is not a confidence interval. Final refit is not validation."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
    return metrics


def train_experiment(config, encoder_factory, data, out):
    from tqdm.auto import tqdm
    out, data = Path(out), Path(data)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty training output directory; previous results are preserved.")
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"
    bundle.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("Run training notebook with a Kaggle GPU.")
    device = torch.device("cuda")
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sample = pd.read_csv(data / "sample_submission.csv", nrows=0)
    assert sample.columns.tolist() == ["StudyInstanceUID", *LABELS]
    assert train.StudyInstanceUID.is_unique and series.SeriesInstanceUID.is_unique
    assert set(series.StudyInstanceUID) <= set(train.StudyInstanceUID)
    assert train[LABELS].stack().dropna().isin([0, 1]).all()
    gold = train.loc[train[LABELS].notna().all(axis=1)].sort_values("StudyInstanceUID").reset_index(drop=True)
    assert len(gold) == 58, "Gold population changed; review the fixed experiment protocol."
    ids, y = gold.StudyInstanceUID.tolist(), gold[LABELS].to_numpy(dtype=np.float32)
    folds = make_folds(ids, y)
    pd.DataFrame({"StudyInstanceUID": ids, "fold": folds}).to_csv(out / "folds.csv", index=False)
    encoder = export_encoder(encoder_factory(), config, bundle / "encoder.ts").to(device)
    transform = make_transform(config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = [study_tokens(uid, series, data / "train_series", encoder, transform, config, device)
            for uid in tqdm(ids, desc=config["id"] + " MRI tokens")]
    tokens, mask, positions = [torch.stack([row[i] for row in rows]).clone() for i in range(3)]
    torch.cuda.synchronize()
    timing = {"feature_seconds": time.perf_counter() - started,
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
              "gpu": torch.cuda.get_device_name(0), "images": int(mask.sum()),
              "note": "Single-run extraction timing; not a warmed repeated latency benchmark."}
    torch.save({"tokens": tokens, "mask": mask, "positions": positions, "study_ids": ids}, out / "tokens.pt")
    metrics = evaluate_cv(tokens, mask, positions, y, ids, folds, config, out)
    final_head, history = fit_model(tokens, mask, positions, torch.from_numpy(y), np.arange(len(ids)), 42, config)
    torch.save({"state": final_head.state_dict(), "experiment": config, "labels": LABELS,
                "train_ids": ids, "seed": 42}, bundle / "model.pt")
    pd.DataFrame(history).to_csv(out / "final_fit_history.csv", index=False)
    wheels = bundle / "wheels"
    wheels.mkdir()
    subprocess.check_call([sys.executable, "-m", "pip", "download", "-q", "--only-binary=:all:",
                           "--no-deps", "--dest", str(wheels), "pydicom==3.0.2", "python-gdcm==3.2.6"])
    manifest = {"experiment": config, "labels": LABELS, "planes": list(PLANES),
                "python": list(sys.version_info[:2]), "torch": torch.__version__,
                "packages": {p: importlib.metadata.version(p) for p in
                             ["torchvision", "numpy", "pandas", "scikit-learn", "pydicom", "python-gdcm"]},
                "final_training_studies": len(ids), "final_seed": 42,
                "folds_sha256": file_hash(out / "folds.csv"), "train_csv_sha256": file_hash(data / "train.csv"),
                "head_parameters": sum(p.numel() for p in final_head.parameters()),
                "files": {p.relative_to(bundle).as_posix(): file_hash(p) for p in sorted(bundle.rglob("*")) if p.is_file()}}
    (bundle / "assets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "timings.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps({"id": config["id"], "macro_auc_mean": metrics["macro_auc_mean"],
                      "fold_auc_mean": metrics["fold_auc_mean"], "bce_mean": metrics["bce_mean"],
                      "timing": timing, "final_train_studies": len(ids)}, indent=2), flush=True)
    return metrics
