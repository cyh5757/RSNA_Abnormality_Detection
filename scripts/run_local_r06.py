"""Prepare MRI inputs, then run one local R06 job without an agent or polling service."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zlib
from zipfile import ZipFile

import pandas as pd

from experiments.core import PLANES, file_hash
from experiments.run_queue import write_json
from experiments.run_spatial_queue import exclusive_lock
from experiments.weak_core import make_targets
from scripts.train_local import ROOT, local_config, source_hashes


def extract_series(archive_path, data, wanted):
    data = Path(data).resolve()
    with ZipFile(archive_path) as archive:
        members = []
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            target = (data / member.filename).resolve()
            if '..' in name.parts or '\\' in member.filename or not target.is_relative_to(data):
                raise ValueError('Unsafe archive path.')
            if not member.is_dir() and name.parent.as_posix() in wanted and name.suffix == '.dcm':
                members.append(member)
        if {PurePosixPath(m.filename).parent.as_posix() for m in members} != wanted:
            raise ValueError('Archive does not contain every selected MRI series.')
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate MRI archive entry.')
        missing_bytes = sum(m.file_size for m in members if not (data / m.filename).exists())
        if shutil.disk_usage(archive_path.parent).free < missing_bytes + 10 * 2**30:
            raise OSError('Insufficient space for selected MRI files plus 10 GiB reserve.')
        print(f'Extracting {len(members)} MRI files; new bytes={missing_bytes}', flush=True)
        for index, member in enumerate(members, 1):
            target = data / member.filename
            if target.exists():
                if target.stat().st_size != member.file_size or zlib.crc32(target.read_bytes()) != member.CRC:
                    raise FileExistsError(f'Existing MRI differs from archive: {target.name}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix('.dcm.part')
                with archive.open(member) as source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if partial.stat().st_size != member.file_size:
                    raise ValueError('Incomplete extracted MRI file.')
                partial.replace(target)
            with target.open('rb') as stream:
                stream.seek(128)
                if stream.read(4) != b'DICM':
                    raise ValueError(f'Invalid DICOM header: {target.name}')
            if index % 2000 == 0 or index == len(members):
                print(f'MRI files verified: {index}/{len(members)}', flush=True)
    return len(members)


def main():
    import ctypes
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFilesRequest

    state_dir = ROOT / 'outputs/local/r06-unattended-20261022'
    state_dir.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'outputs/local/r06-rtx5090-train'
    data = ROOT / 'data/raw'
    archive_path = ROOT / 'data/downloads/rsna-knee-abnormality-detection.zip'
    gpu_python = ROOT / '.venv-5090/Scripts/python.exe'
    record = dict(pid=os.getpid(), output=str(output), automatic_agent_monitoring=False)

    def status(phase, **details):
        record.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **details)
        write_json(state_dir / 'status.json', record)
        print(f'{record["updated_at"]} {phase}', flush=True)

    with exclusive_lock(ROOT / 'outputs/local/r06-training.lock'):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            status('preparing')
            if output.exists():
                raise FileExistsError('Training output already exists; refusing to overwrite or retrain.')
            if not gpu_python.is_file():
                raise FileNotFoundError('Missing verified .venv-5090 environment.')
            pins = source_hashes()
            pins['scripts/run_local_r06.py'] = file_hash(Path(__file__))
            inputs = [data / 'train.csv', data / 'train_series.csv',
                      ROOT / 'data/weak_labels/report_labels_v2.csv',
                      ROOT / 'data/local-assets/s11/run/bundle/assets.json']
            input_hashes = {str(p): file_hash(p) for p in inputs}
            meta = json.loads(inputs[-1].read_text(encoding='utf-8'))
            config = local_config(meta['experiment'])
            if input_hashes[str(inputs[2])] != config['labels_sha256']:
                raise ValueError('Weak-label hash mismatch.')
            table, _, _, gold, _, _ = make_targets(pd.read_csv(inputs[0]), pd.read_csv(inputs[2]), config)
            if len(table) != 4350 or int(gold.sum()) != 58:
                raise ValueError('Training population changed.')
            series = pd.read_csv(inputs[1])
            if not series.SeriesInstanceUID.is_unique:
                raise ValueError('Duplicate series metadata.')
            selected = series.loc[series.StudyInstanceUID.isin(table.StudyInstanceUID)
                                  & series.Anatomical_Plane.isin(PLANES)]
            selected = selected.sort_values(['Fluid_Sensitive', 'SeriesInstanceUID'], ascending=[False, True])
            selected = selected.drop_duplicates(['StudyInstanceUID', 'Anatomical_Plane', 'Fat_Suppression'])
            if set(selected.StudyInstanceUID) != set(table.StudyInstanceUID):
                raise ValueError('Missing selected study metadata.')
            wanted = {f'train_series/{row.StudyInstanceUID}/{row.SeriesInstanceUID}'
                      for row in selected.itertuples()}
            write_json(state_dir / 'inputs.json', dict(source_hashes=pins, input_hashes=input_hashes,
                       selected_series=sorted(wanted), experiment=config))
            api = KaggleApi()
            api.authenticate()
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with api.build_kaggle_client() as client:
                    request = ApiDownloadDataFilesRequest()
                    request.competition_name = 'rsna-knee-abnormality-detection'
                    response = client.competitions.competition_api_client.download_data_files(request)
                    size = int(response.headers['Content-Length'])
                    existing = archive_path.stat().st_size if archive_path.exists() else 0
                    if shutil.disk_usage(ROOT).free < max(0, size - existing) + 610 * 2**30:
                        raise OSError('Insufficient space for archive, MRI extraction and training reserve.')
                    status('downloading', archive_bytes=size, selected_series=len(wanted),
                           training_studies=len(table))
                    try:
                        if existing != size or not archive_path.is_file():
                            api.download_file(response, str(archive_path), client.http_client(), quiet=True, resume=True)
                    finally:
                        response.close()
            except Exception as exc:
                raise RuntimeError(f'Kaggle archive preparation failed ({type(exc).__name__}).') from None
            status('extracting')
            with ZipFile(archive_path) as archive:
                for name in ('train.csv', 'train_series.csv'):
                    if hashlib.sha256(archive.read(name)).hexdigest() != input_hashes[str(data / name)]:
                        raise ValueError('Downloaded competition metadata differs from pinned local inputs.')
            count = extract_series(archive_path, data, wanted)
            status('checking', dicom_files=count)
            current = source_hashes()
            current['scripts/run_local_r06.py'] = file_hash(Path(__file__))
            if current != pins or any(file_hash(Path(p)) != digest for p, digest in input_hashes.items()):
                raise ValueError('Code or metadata changed during input preparation; training not started.')
            base = [str(gpu_python), '-u', '-X', 'utf8', '-m', 'scripts.train_local']
            subprocess.run(base + ['--mode', 'check'], cwd=ROOT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            status('training')
            subprocess.run(base + ['--mode', 'train', '--output', str(output)], cwd=ROOT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            result = json.loads((output / 'local_run.json').read_text())
            if result['status'] != 'complete':
                raise RuntimeError('Training did not produce a completion record.')
            status('complete')
        except Exception as exc:
            status('failed', failed_phase=record['phase'], error=f'{type(exc).__name__}: {exc}')
            return 1
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
