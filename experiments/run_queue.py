"""Run the existing train/submit notebooks without an AI client."""
import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.build_notebooks import ROOT, build_all, experiment_configs, all_experiment_configs
from experiments.core import LABELS, file_hash

DEFAULT_EXPERIMENTS = ['e03_attention', 'e04_position', 'e05_plane']
TRAIN_FILES = r'(metrics\.json|folds\.csv|history\.csv|timings\.json|assets\.json|oof_predictions\.csv|label_audit\.json|backbone_audit\.json|selection_audit\.csv)$'
SUBMIT_FILES = r'(submission\.csv|inference_summary\.json)$'


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def latest_version(api, ref):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = ref.split('/')
    with api.build_kaggle_client() as client:
        return retry_kaggle_read(lambda: client.kernels.kernels_api_client.get_kernel(request)).metadata.current_version_number


def retry_kaggle_read(call, *, attempts=5, sleep=time.sleep):
    from requests.exceptions import ConnectionError as NetworkError, Timeout
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', getattr(exc, 'status', None))
            transient = isinstance(exc, (NetworkError, Timeout)) or status in (429, 502, 503, 504)
            if not transient or attempt + 1 == attempts:
                raise
            delay = min(60, 5 * 2**attempt)
            try:
                delay = min(300, max(delay, float(getattr(response, 'headers', {}).get('Retry-After', 0))))
            except (TypeError, ValueError):
                pass
            logging.warning('Kaggle read %s; retry %s/%s in %ss', type(exc).__name__, attempt+2, attempts, delay)
            sleep(delay)


def check_version(api, ref, version):
    # The installed CLI parses version suffixes but omits them from status/output requests.
    if latest_version(api, ref) != version:
        raise RuntimeError(f'{ref}: remote version changed; stop other writers and inspect state.json.')


def wait_for_job(api, ref, version, output, poll_seconds):
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        check_version(api, ref, version)
        response = retry_kaggle_read(lambda: api.kernels_status(ref))
        status = getattr(response.status, 'name', str(response.status).rsplit('.', 1)[-1]).upper()
        logging.info('%s v%s: %s', ref, version, status)
        if status == 'COMPLETE':
            return
        if status not in {'RUNNING', 'QUEUED', 'PENDING', 'INITIALIZING'}:
            check_version(api, ref, version)
            retry_kaggle_read(lambda: api.kernels_output(ref, str(output), file_pattern=r'(?!)', quiet=True))
            raise RuntimeError(f'{ref} v{version}: {status}. {response.failure_message} Logs: {output}')
        time.sleep(poll_seconds)
    raise TimeoutError(f'{ref}: waited 12 hours. The Kaggle job may still run; rerun this command to resume.')


def verify_results(config, train_dir, submit_dir):
    metrics = json.loads((train_dir / 'run/metrics.json').read_text(encoding='utf-8'))
    assets = json.loads((train_dir / 'run/bundle/assets.json').read_text(encoding='utf-8'))
    summary = json.loads((submit_dir / 'inference_summary.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(submit_dir / 'submission.csv')
    if not (metrics['experiment'] == assets['experiment'] == summary['experiment'] == config):
        raise ValueError('Downloaded outputs belong to a different experiment/configuration.')
    if config.get('suite') == 'weak':
        matched = (assets['final_training_studies'] == config['expected_training_studies']
                   and assets['gold_training_studies'] == 58
                   and summary['manifest_sha256'] == file_hash(train_dir / 'run/bundle/assets.json'))
    else:
        matched = (assets['final_training_studies'] == 58
                   and summary['model_sha256'] == assets['files']['model.pt']
                   and summary['encoder_sha256'] == assets['files']['encoder.ts'])
    if not matched or summary['internet_required'] is not False:
        raise ValueError('Submission does not match the final training model.')
    if (frame.columns.tolist() != ['StudyInstanceUID', *LABELS]
            or len(frame) != summary['studies'] or not len(frame)
            or frame.StudyInstanceUID.isna().any() or not frame.StudyInstanceUID.is_unique):
        raise ValueError('Invalid submission columns or study IDs.')
    values = frame[LABELS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError('Submission probabilities must be finite and between 0 and 1.')
    test_csv = ROOT / 'data/raw/test.csv'
    if test_csv.exists() and frame.StudyInstanceUID.tolist() != pd.read_csv(test_csv).StudyInstanceUID.tolist():
        raise ValueError('Submission study IDs differ from the local public test.csv.')
    auc, bce = metrics['macro_auc_mean'], metrics['bce_mean']
    if not np.isfinite([auc, bce]).all() or not 0 <= auc <= 1 or bce < 0:
        raise ValueError('Invalid validation metrics.')
    return {'macro_auc': auc, 'bce': bce}


def run(api, configs, run_dir, poll_seconds=60, builder=None, external_jobs=None, before_upload=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = b''.join((ROOT / 'experiments' / name).read_bytes() for name in
                       ('core.py', 'backbones.py', 'train.py', 'submit.py', 'build_notebooks.py'))
    if any(c.get('suite') == 'weak' for c in configs):
        sources += b''.join((ROOT / 'experiments' / name).read_bytes() for name in
                           ('weak_core.py', 'weak_train.py', 'weak_submit.py', 'weak_configs.py'))
    fingerprint = hashlib.sha256(sources + json.dumps(configs, sort_keys=True).encode()).hexdigest()
    state_path = run_dir / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'fingerprint': fingerprint, 'jobs': {}}
    if state['fingerprint'] != fingerprint:
        raise ValueError('Experiment list or source changed. Use a new run name to preserve previous results.')
    for eid, job in (external_jobs or {}).items():
        if eid in state['jobs'] and state['jobs'][eid] != job:
            raise ValueError('Pinned external input changed.')
        state['jobs'][eid] = job
    write_json(state_path, state)
    (builder or build_all)(configs, run_dir / 'notebooks')

    def stage(config, mode):
        eid = config['id']
        folder = run_dir / 'notebooks' / eid / mode
        metadata = json.loads((folder / 'kernel-metadata.json').read_text(encoding='utf-8'))
        jobs = state['jobs'].setdefault(eid, {})
        if mode == 'train' and config.get('train_dependencies'):
            sources = []
            for dependency in config['train_dependencies']:
                prior = state['jobs'].get(dependency, {})
                if not prior.get('verified'):
                    raise RuntimeError(f'Run dependency {dependency} first in this queue.')
                job = prior['train']
                check_version(api, job['ref'], job['version'])
                sources.append(f"{job['ref']}/{job['version']}")
            metadata['kernel_sources'] = sources
            write_json(folder / 'kernel-metadata.json', metadata)
        if mode == 'submit':
            metadata['kernel_sources'] = [f"{jobs['train']['ref']}/{jobs['train']['version']}"]
            write_json(folder / 'kernel-metadata.json', metadata)
        if mode not in jobs:
            if before_upload:
                before_upload(config, mode, metadata)
            jobs[mode] = {'ref': metadata['id'], 'version': None, 'downloaded': False}
            write_json(state_path, state)
            logging.info('Uploading %s', metadata['id'])
            result = api.kernels_push(str(folder))
            jobs[mode]['version'] = result.version_number or None
            invalid = any(getattr(result, field) for field in ('invalid_kernel_sources',
                          'invalid_dataset_sources', 'invalid_competition_sources', 'invalid_model_sources'))
            jobs[mode]['error'] = result.error or ('Kaggle rejected an input source.' if invalid else '')
            write_json(state_path, state)
        job = jobs[mode]
        if job.get('error'):
            raise RuntimeError(job['error'])
        if not job['version']:
            raise RuntimeError(f"Unknown upload outcome for {job['ref']}. Check Kaggle before retrying; see state.json.")
        output = run_dir / eid / mode
        if not job['downloaded']:
            wait_for_job(api, job['ref'], job['version'], output, poll_seconds)
            check_version(api, job['ref'], job['version'])
            retry_kaggle_read(lambda: api.kernels_output(job['ref'], str(output), quiet=True, force=True,
                               file_pattern=TRAIN_FILES if mode == 'train' else SUBMIT_FILES))
            check_version(api, job['ref'], job['version'])
        return output

    rows = []
    for config in configs:
        logging.info('Experiment: %s', config['id'])
        train_dir = stage(config, 'train')
        submit_dir = stage(config, 'submit')
        scores = verify_results(config, train_dir, submit_dir)
        jobs = state['jobs'][config['id']]
        jobs['train']['downloaded'] = jobs['submit']['downloaded'] = True
        jobs['verified'] = True
        write_json(state_path, state)
        rows.append({'experiment': config['id'], **scores,
                     'train_version': jobs['train']['version'], 'submit_version': jobs['submit']['version'],
                     'submit_url': 'https://www.kaggle.com/code/' + jobs['submit']['ref']})
        temporary = run_dir / 'summary.csv.tmp'
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(run_dir / 'summary.csv')
        logging.info('Verified %s: OOF AUC=%.6f; %s', config['id'], scores['macro_auc'], submit_dir / 'submission.csv')
    logging.info('Finished. Results: %s', run_dir / 'summary.csv')


def main():
    configs_by_id = {c['id']: c for c in all_experiment_configs()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=configs_by_id, default=DEFAULT_EXPERIMENTS)
    parser.add_argument('--run-name', default='next-experiments')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run_name):
        parser.error('Run name must use letters, digits, underscores or hyphens (max 80 characters).')
    if args.poll_seconds < 10 or len(set(args.experiments)) != len(args.experiments):
        parser.error('Use a polling interval of at least 10 seconds and unique experiment IDs.')
    configs = [configs_by_id[name] for name in args.experiments]
    run_dir = ROOT / 'outputs/automation' / args.run_name
    if args.dry_run:
        print(f'Run directory: {run_dir}')
        for c in configs:
            print(f"{c['id']} ({c['backbone']}): train upload -> wait -> pin version -> submit upload -> wait -> download/verify")
        print('No uploads. Normal execution creates private GPU notebooks; leaderboard submission is manual.')
        return

    import msvcrt
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (run_dir.parent / 'runner.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            parser.exit(1, 'Another experiment runner is active on this PC.\n')
        run_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
            logging.StreamHandler(), logging.FileHandler(run_dir / 'runner.log', encoding='utf-8')])
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        try:
            run(api, configs, run_dir, args.poll_seconds)
        except KeyboardInterrupt:
            parser.exit(130, '\nLocal runner stopped. Kaggle jobs continue; rerun the same command to resume.\n')
        except Exception as exc:
            logging.error('%s: %s', type(exc).__name__, exc)
            parser.exit(1, f'Stopped; state preserved in {run_dir}.\n')


if __name__ == '__main__':
    main()
