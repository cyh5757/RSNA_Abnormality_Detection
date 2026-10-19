"""Offline inference entry point embedded into each submission notebook."""
import subprocess
import sys
import time

from experiments.core import *


def submit_experiment(config, data, bundle, out):
    from tqdm.auto import tqdm
    started = time.monotonic()
    data, bundle, out = Path(data), Path(bundle), Path(out)
    manifest = validate_bundle(bundle, config)
    if manifest["python"] != list(sys.version_info[:2]) or manifest["torch"] != torch.__version__:
        raise ValueError("Use the same pinned Kaggle Docker image as the training notebook.")
    assert manifest["labels"] == LABELS and manifest["planes"] == list(PLANES)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                           "--find-links", str(bundle / "wheels"), "pydicom==3.0.2", "python-gdcm==3.2.6"])
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.jit.load(str(bundle / "encoder.ts"), map_location=device).eval()
    saved = torch.load(bundle / "model.pt", map_location="cpu", weights_only=True)
    assert saved["experiment"] == config and saved["labels"] == LABELS
    state = saved["state"]
    head = StudyHead(config, state["mean"], state["std"])
    head.load_state_dict(state, strict=True)
    head = head.to(device).eval()
    transform = make_transform(config)
    test, series = pd.read_csv(data / "test.csv"), pd.read_csv(data / "test_series.csv")
    sample = pd.read_csv(data / "sample_submission.csv", nrows=0)
    assert test.StudyInstanceUID.notna().all() and test.StudyInstanceUID.is_unique and len(test)
    assert series.SeriesInstanceUID.is_unique and set(test.StudyInstanceUID) <= set(series.StudyInstanceUID)
    probabilities = []
    with torch.inference_mode():
        for uid in tqdm(test.StudyInstanceUID, desc=config["id"] + " test studies"):
            tokens, mask, positions = study_tokens(uid, series, data / "test_series", encoder, transform, config, device)
            prediction = head(tokens[None].to(device), mask[None].to(device), positions[None].to(device))
            probabilities.append(prediction.sigmoid().cpu().numpy()[0])
    out.mkdir(parents=True, exist_ok=True)
    frame = save_submission(test.StudyInstanceUID.tolist(), probabilities, sample.columns, out / "submission.csv")
    summary = {"experiment": config, "studies": len(test), "labels": 12,
               "seconds": time.monotonic() - started, "model_sha256": manifest["files"]["model.pt"],
               "encoder_sha256": manifest["files"]["encoder.ts"], "internet_required": False}
    (out / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(frame.head().to_string(index=False), flush=True)
    return frame
