"""Run local R06 or validation-driven fine-tuning on a CUDA GPU."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

from experiments import spatial_train
from experiments.backbones import swin_t_encoder
from experiments.core import file_hash, PLANES
from experiments.weak_core import make_targets, validate_weak_bundle

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [ROOT/'experiments'/name for name in (
    'core.py', 'backbones.py', 'train.py', 'weak_core.py', 'weak_train.py',
    'spatial_core.py', 'spatial_train.py', 'public_finetune.py')] + [Path(__file__).resolve()]


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in SOURCE_FILES}


def local_config(parent, early_stopping=False, public_finetuning=False):
    if (parent.get('id') != 's11_cross' or parent.get('backbone') != 'swin_t'
            or parent.get('architecture') != 'spatial_v1'
            or parent.get('finetune') or parent.get('ensemble')):
        raise ValueError('Use the original frozen S11 training output.')
    digest = hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()
    config = dict(parent, id='r06_local_warmup', comparison_parent='s12_finetune',
                pipeline_sha256=digest, feature_pipeline_sha256=parent['pipeline_sha256'],
                train_dependencies=['s11_cross'], center_range=[.25, .75],
                head_warmup_epochs=1, backbone_epochs=1, finetune=True, ensemble=False)
    if early_stopping or public_finetuning:
        config.update(id='local_earlystop_p3', comparison_parent='r06_local_warmup',
                      backbone_epochs=30, early_stopping_patience=3, backbone_timeout_hours=72.)
    if public_finetuning:
        config.update(id='local_public_full_cosine', comparison_parent='local_earlystop_p3',
                      backbone_scope='full', backbone_schedule='warmup_cosine',
                      gradient_checkpointing=True, keep_backbone_checkpoints=3)
    return config


def smoke_indices(gold, groups):
    train, valid = spatial_train.weak_finetune_split(gold, groups)
    if len(train) < 2:
        raise ValueError('Smoke training needs two weak training studies and one validation study.')
    return np.concatenate((train[:2], valid[:1]))


def check_images(table, series, image_root):
    required = {'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane',
                'Fat_Suppression', 'Fluid_Sensitive'}
    if not required <= set(series) or not series.SeriesInstanceUID.is_unique:
        raise ValueError('Invalid MRI series metadata.')
    if not set(table.StudyInstanceUID) <= set(series.StudyInstanceUID):
        raise ValueError('Missing study series metadata.')
    count = 0
    for uid, rows in series.loc[series.StudyInstanceUID.isin(table.StudyInstanceUID)].groupby('StudyInstanceUID'):
        present = 0
        for plane in PLANES:
            for fs in (True, False):
                choices = rows.loc[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
                if choices.empty:
                    continue
                sr = choices.sort_values(['Fluid_Sensitive', 'SeriesInstanceUID'], ascending=[False, True]).iloc[0].SeriesInstanceUID
                if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', str(x)) or str(x) in ('.', '..') for x in (uid, sr)):
                    raise ValueError('Unsafe study or series directory name.')
                folder = (image_root/str(uid)/str(sr)).resolve()
                if not folder.is_relative_to(image_root.resolve()):
                    raise ValueError('MRI series path leaves the data directory.')
                if not folder.is_dir() or next(folder.glob('*.dcm'), None) is None:
                    raise FileNotFoundError('Selected MRI series missing or empty; finish the DICOM download.')
                present += 1
        if not present:
            raise ValueError('Study has no usable MRI protocol slots.')
        count += present
    return count


def validate_inputs(data, labels, parent_run, mode, early_stopping=False, public_finetuning=False):
    metadata = json.loads((parent_run/'bundle/assets.json').read_text(encoding='utf-8'))
    config = local_config(metadata['experiment'], early_stopping, public_finetuning)
    validate_weak_bundle(parent_run/'bundle', metadata['experiment'])
    if file_hash(labels) != config['labels_sha256']:
        raise ValueError('Report label hash does not match the S11 training labels.')
    table, y, w, gold, groups, _ = make_targets(pd.read_csv(data/'train.csv'), pd.read_csv(labels), config)
    if len(table) != 4350 or int(gold.sum()) != 58:
        raise ValueError('Expected the original 4,350 training studies including Gold58.')
    series = pd.read_csv(data/'train_series.csv')
    image_table = table.iloc[smoke_indices(gold, groups)] if mode == 'smoke' else table
    selected_series = check_images(image_table, series, data/'train_series')
    if file_hash(parent_run/'tokens.pt') != metadata['tokens_sha256']:
        raise ValueError('S11 feature cache hash mismatch.')
    cache = torch.load(parent_run/'tokens.pt', map_location='cpu', weights_only=True)
    if not spatial_train.compatible_spatial_cache(cache, config, file_hash(parent_run/'bundle/encoder.ts'), table.StudyInstanceUID.tolist()):
        raise ValueError('S11 cache IDs, encoder or preprocessing do not match.')
    if (cache['tokens'].ndim != 4 or cache['mask'].dtype != torch.bool
            or cache['tokens'].shape[:-1] != cache['mask'].shape
            or cache['coords'].shape != (*cache['mask'].shape, 3)
            or len(cache['tokens']) != len(table)
            or cache['tokens'].shape[-1] != config['feature_dim']
            or not cache['mask'].flatten(1).any(1).all()
            or not torch.isfinite(cache['tokens']).all() or not torch.isfinite(cache['coords']).all()):
        raise ValueError('Invalid S11 feature tensors.')
    return dict(config=config, table=table, y=torch.from_numpy(y), w=torch.from_numpy(w),
                gold=gold, groups=groups, series=series, cache=cache, selected_series=selected_series)


@contextmanager
def local_paths(labels, parent_run):
    def find_parent(eid):
        if eid != 's11_cross':
            raise ValueError('Local R06 only accepts the verified S11 parent.')
        return parent_run
    # Resolve two Kaggle-only lookups in this process without changing pinned source files.
    old = spatial_train.find_report_labels, spatial_train.find_training_run
    spatial_train.find_report_labels = lambda config: labels
    spatial_train.find_training_run = find_parent
    try:
        yield
    finally:
        spatial_train.find_report_labels, spatial_train.find_training_run = old


@contextmanager
def run_record(output, details):
    output.mkdir(parents=True, exist_ok=False)
    record = dict(details, pid=os.getpid(), status='running', started_at=datetime.now(timezone.utc).isoformat())
    def save():
        temporary = output/'local_run.json.tmp'
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(output/'local_run.json')
    save()
    try:
        yield
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        record['status'] = 'complete'
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()


@contextmanager
def early_stopping_session(enabled, lock_path=None):
    if not enabled:
        yield
        return
    from experiments.run_spatial_queue import exclusive_lock
    with exclusive_lock(lock_path or ROOT/'outputs/local/r06-training.lock'):
        import ctypes
        if os.name == 'nt' and not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError('Could not prevent automatic sleep during training.')
        try:
            yield
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def smoke_train(inputs, data, parent_run, output):
    indices = smoke_indices(inputs['gold'], inputs['groups'])
    cache = {key: inputs['cache'][key][indices] for key in ('tokens', 'mask', 'coords')}
    encoder = swin_t_encoder(pretrained=False)
    encoder.load_state_dict(torch.jit.load(str(parent_run/'bundle/encoder.ts'), map_location='cpu').state_dict(), strict=True)
    output.mkdir()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    spatial_train.finetune_spatial(encoder, cache, inputs['table'].iloc[indices].reset_index(drop=True),
        inputs['y'][indices], inputs['w'][indices], inputs['gold'][indices], inputs['groups'][indices],
        inputs['series'], data, inputs['config'], output)
    result = dict(training_studies=2, validation_studies=1, gold_studies=0,
                  seconds=time.monotonic()-started, peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  note='Execution smoke test only; not an OOF score or submission model.')
    (output/'smoke_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'smoke', 'train'), default='check')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--labels', type=Path, default=ROOT/'data/weak_labels/report_labels_v2.csv')
    parser.add_argument('--parent-run', type=Path, default=ROOT/'data/local-assets/s11/run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--early-stopping', action='store_true',
                        help='Stop after 3 epochs without a lower weak validation BCE; cap at 30 epochs.')
    parser.add_argument('--public-finetuning', action='store_true',
                        help='Full encoder, 5 percent LR warmup/cosine, patience 3, retain top 3 checkpoints.')
    parser.add_argument('--average-from', type=Path,
                        help='Evaluate the top 3 weak-validation checkpoints from a completed public-method run.')
    parser.add_argument('--training-lock', type=Path, help='Shared GPU lock for an isolated source snapshot.')
    args = parser.parse_args(argv)
    if args.average_from and (not args.public_finetuning or args.mode != 'train'):
        parser.error('--average-from requires --public-finetuning --mode train.')
    args.early_stopping = args.early_stopping or args.public_finetuning
    data, labels, parent = (p.resolve() for p in (args.data_dir, args.labels, args.parent_run))
    report = dict(mode=args.mode, python=sys.version.split()[0], torch=torch.__version__,
                  cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(), errors=[])
    if not report['cuda_available']:
        report['errors'].append('CUDA is unavailable. Use the separate .venv-5090 CUDA environment.')
    else:
        try:
            probe = torch.ones(8, 8, device='cuda', requires_grad=True)
            (probe @ probe).sum().backward()
            torch.cuda.synchronize()
            report.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
            del probe
        except RuntimeError as exc:
            report['errors'].append(f'CUDA forward/backward failed: {exc}')
    try:
        inputs = validate_inputs(data, labels, parent, args.mode, args.early_stopping, args.public_finetuning)
        report.update(training_studies=len(inputs['table']), selected_series=inputs['selected_series'])
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        report['errors'].append(str(exc))
    print(json.dumps(report, indent=2), flush=True)
    if report['errors']:
        return 1
    if args.mode == 'check':
        return 0
    prefix = 'earlystop-p3' if args.early_stopping else 'r06'
    if args.public_finetuning: prefix = 'public-full-cosine-avg3' if args.average_from else 'public-full-cosine'
    output = (args.output or ROOT/'outputs/local'/f'{prefix}-rtx5090-{args.mode}').resolve()
    from experiments.public_finetune import average_backbone, use_averaged_backbone
    averaged, average_audit = None, None
    if args.average_from:
        averaged, average_audit = average_backbone(args.average_from.resolve(), inputs['config'])
        inputs['config'] = dict(inputs['config'], id='local_public_full_cosine_avg3',
                                comparison_parent='local_public_full_cosine')
    receipt = dict(report, config=inputs['config'], source_hashes=source_hashes(),
                   data_dir=str(data), parent_run=str(parent),
                   input_hashes={str(p): file_hash(p) for p in
                                 (data/'train.csv', data/'train_series.csv', labels, parent/'bundle/assets.json')})
    with early_stopping_session(args.early_stopping, args.training_lock), run_record(output, receipt):
        if args.mode == 'smoke':
            print(json.dumps(smoke_train(inputs, data, parent, output/'smoke'), indent=2))
        else:
            config = inputs['config']
            del inputs
            with local_paths(labels, parent), use_averaged_backbone(averaged, average_audit):
                spatial_train.train_spatial_experiment(config, swin_t_encoder, data, output/'run')
            validate_weak_bundle(output/'run/bundle', config)
    print(f'Completed {args.mode}: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
