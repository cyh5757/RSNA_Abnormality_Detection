"""Offline submission for individual weak-label models and their fixed ensemble."""
import subprocess
import sys
import time
from experiments.core import *
from experiments.weak_core import *


def load_weak_predictor(bundle, config, device):
    manifest = validate_weak_bundle(bundle, config)
    if manifest['python'] != list(sys.version_info[:2]) or manifest['torch'] != torch.__version__:
        raise ValueError('Use the same pinned Kaggle runtime for training and submission.')
    saved = torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True)
    if saved['experiment'] != config or saved['labels'] != LABELS:
        raise ValueError('Wrong trained head.')
    head = WeakHead(config, saved['state']['mean'], saved['state']['std'])
    head.load_state_dict(saved['state'], strict=True)
    encoder = torch.jit.load(str(bundle / 'encoder.ts'), map_location=device).eval()
    return encoder, head.to(device).eval(), make_transform(config), config


def submit_weak_experiment(config, data, bundle, out):
    from tqdm.auto import tqdm
    data, bundle, out = Path(data), Path(bundle), Path(out)
    started = time.monotonic()
    manifest = validate_weak_bundle(bundle, config)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                           '--find-links', str(bundle / 'wheels'), 'pydicom==3.0.2', 'python-gdcm==3.2.6'])
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    members = []
    if config.get('ensemble'):
        for eid in config['train_dependencies']:
            path = bundle / 'members' / eid
            member = json.loads((path / 'assets.json').read_text())
            members.append(load_weak_predictor(path, member['experiment'], device))
    else:
        members.append(load_weak_predictor(bundle, config, device))
    test, series = pd.read_csv(data / 'test.csv'), pd.read_csv(data / 'test_series.csv')
    sample = pd.read_csv(data / 'sample_submission.csv', nrows=0)
    if (not len(test) or test.StudyInstanceUID.isna().any() or not test.StudyInstanceUID.is_unique
            or not series.SeriesInstanceUID.is_unique or not set(test.StudyInstanceUID) <= set(series.StudyInstanceUID)):
        raise ValueError('Invalid test study/series metadata.')
    predictions = []
    with torch.inference_mode():
        for uid in tqdm(test.StudyInstanceUID, desc=config['id'] + ' test studies'):
            probs = []
            for encoder, head, transform, member_config in members:
                tokens, mask, _ = weak_tokens(uid, series, data / 'test_series', encoder, transform, member_config, device)
                probs.append(head(tokens[None].to(device), mask[None].to(device)).sigmoid().cpu().numpy()[0])
            predictions.append(np.mean(probs, axis=0))
    out.mkdir(parents=True, exist_ok=True)
    frame = save_submission(test.StudyInstanceUID.tolist(), predictions, sample.columns, out / 'submission.csv')
    summary = {'experiment': config, 'studies': len(test), 'labels': 12, 'seconds': time.monotonic()-started,
               'manifest_sha256': file_hash(bundle / 'assets.json'), 'internet_required': False}
    (out / 'inference_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return frame
