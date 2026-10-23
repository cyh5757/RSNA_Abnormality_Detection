"""After the active local run: full-encoder smoke, fine-tuning and checkpoint averaging."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.core import file_hash
from experiments.run_queue import write_json
from experiments.run_spatial_queue import exclusive_lock, RunnerBusy

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ('config', 'input_hashes', 'source_hashes', 'started_at', 'pid')


def predecessor_ready(record, identity):
    if any(record.get(key) != identity[key] for key in IDENTITY):
        raise ValueError('Predecessor run identity changed.')
    if record.get('status') not in ('running', 'complete'):
        raise ValueError('Predecessor did not complete successfully; queue stopped.')
    return record['status'] == 'complete'


def prepare_queue(root, queue):
    predecessor = root/'outputs/local/earlystop-p3-rtx5090-train/local_run.json'
    record = json.loads(predecessor.read_text(encoding='utf-8'))
    identity = {key: record[key] for key in IDENTITY}
    predecessor_ready(record, identity)
    queue.mkdir(parents=True, exist_ok=False)
    source = queue/'source'
    files = sorted((root/'experiments').glob('*.py'))
    files += [root/'scripts'/name for name in ('__init__.py', 'train_local.py', 'run_local_public_methods.py')]
    hashes = {}
    for path in files:
        if not path.exists():
            if path.name == '__init__.py': continue
            raise FileNotFoundError(path)
        relative = path.relative_to(root)
        target = source/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target); hashes[relative.as_posix()] = file_hash(target)
    inputs = [root/name for name in ('data/raw/train.csv', 'data/raw/train_series.csv',
        'data/weak_labels/report_labels_v2.csv', 'data/local-assets/s11/run/bundle/assets.json')]
    pins = dict(workspace=str(root), predecessor=str(predecessor), predecessor_identity=identity,
                sources=hashes, inputs={str(path): file_hash(path) for path in inputs})
    write_json(queue/'pins.json', pins)
    return pins


def verify_pins(queue):
    pins = json.loads((queue/'pins.json').read_text(encoding='utf-8'))
    if any(file_hash(queue/'source'/name) != digest for name, digest in pins['sources'].items()):
        raise ValueError('Frozen queue source changed.')
    if any(file_hash(Path(name)) != digest for name, digest in pins['inputs'].items()):
        raise ValueError('Training inputs changed while queued.')
    return pins


def run_queue(queue):
    pins = verify_pins(queue); root = Path(pins['workspace'])
    predecessor = Path(pins['predecessor']); shared_lock = root/'outputs/local/r06-training.lock'
    record = dict(pid=os.getpid(), automatic_agent_monitoring=False, official_submission=False)

    def status(phase, **details):
        record.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **details)
        write_json(queue/'status.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    with exclusive_lock(root/'outputs/local/public-finetune-queue.lock'):
        import ctypes
        if os.name == 'nt' and not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError('Could not prevent automatic sleep.')
        try:
            status('waiting_for_predecessor', predecessor=str(predecessor))
            while True:
                current = json.loads(predecessor.read_text(encoding='utf-8'))
                if predecessor_ready(current, pins['predecessor_identity']): break
                try:
                    with exclusive_lock(shared_lock):
                        current = json.loads(predecessor.read_text(encoding='utf-8'))
                        if not predecessor_ready(current, pins['predecessor_identity']):
                            raise RuntimeError('Predecessor says running but no longer owns its training lock.')
                except RunnerBusy:
                    pass
                time.sleep(60)
            base = [str(root/'.venv-5090/Scripts/python.exe'), '-u', '-X', 'utf8', '-m', 'scripts.train_local',
                    '--public-finetuning', '--data-dir', str(root/'data/raw'),
                    '--labels', str(root/'data/weak_labels/report_labels_v2.csv'),
                    '--parent-run', str(root/'data/local-assets/s11/run'), '--training-lock', str(shared_lock)]
            for name, mode in (('smoke', 'smoke'), ('full', 'train'), ('average3', 'train')):
                verify_pins(queue)
                output = queue/name
                if output.exists(): raise FileExistsError('Stage output exists; refusing automatic retraining.')
                status(name, output=str(output))
                command = base + ['--mode', mode, '--output', str(output)]
                if name == 'average3': command += ['--average-from', str(queue/'full')]
                subprocess.run(command, cwd=queue/'source', check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                receipt = json.loads((output/'local_run.json').read_text(encoding='utf-8'))
                if receipt['status'] != 'complete': raise RuntimeError('Missing successful stage receipt.')
            comparison = []
            for name, path in (('partial_patience3', predecessor.parent), ('full_cosine', queue/'full'),
                               ('full_cosine_average3', queue/'average3')):
                metrics = json.loads((path/'run/metrics.json').read_text(encoding='utf-8'))
                comparison.append(dict(experiment=name, output=str(path),
                    macro_auc=metrics['macro_auc_mean'], seed_std=metrics['macro_auc_seed_std'],
                    bce=metrics['bce_mean'], metrics_sha256=file_hash(path/'run/metrics.json')))
            write_json(queue/'comparison.json', dict(results=comparison,
                note='Development OOF, not Public scores. Full scope and schedule change together.'))
            status('complete')
        except Exception as exc:
            status('failed', failed_phase=record.get('phase'), error=f'{type(exc).__name__}: {exc}')
            return 1
        finally:
            if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--queue-dir', type=Path, default=ROOT/'outputs/local/public-finetune-20261023')
    args = parser.parse_args(); queue = args.queue_dir.resolve()
    if args.prepare:
        pins = prepare_queue(ROOT, queue)
        print(f"Prepared {queue}: {len(pins['sources'])} source files; no training started.", flush=True)
        return 0
    return run_queue(queue)


if __name__ == '__main__':
    raise SystemExit(main())
