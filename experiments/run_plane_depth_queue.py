"""Wait for the shared runner lock, then execute the three fixed position ablations."""
import argparse
import json
import logging
import re
import time
from experiments.build_notebooks import ROOT
from experiments.followup_configs import INPUTS, followup_configs
from experiments.plane_depth_configs import plane_depth_configs
from experiments.plane_depth_notebooks import build_plane_depth
from experiments.run_followup_queue import resolve_verified_inputs, wait_for_gpu
from experiments.run_queue import run, write_json, verify_results
from experiments.run_spatial_queue import exclusive_lock, RunnerBusy


def verify_finished_followup(previous):
    """No lock owner is not enough: a crashed predecessor is not a completed one."""
    state_path = previous / 'state.json'
    if not state_path.exists():
        raise RuntimeError('Follow-up queue has not started. Resume it first.')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    for config in followup_configs():
        eid = config['id']
        job = state.get('jobs', {}).get(eid, {})
        if not job.get('verified') or any(not job.get(m, {}).get('downloaded') or
                not isinstance(job[m].get('version'), int) or job[m]['version'] < 1 or job[m].get('error')
                for m in ('train', 'submit')):
            raise RuntimeError(f'Predecessor {eid} is incomplete. Resume {previous.name} first.')
        folder = previous / eid
        saved = json.loads((folder / 'train/run/bundle/assets.json').read_text(encoding='utf-8'))['experiment']
        if saved['id'] != eid:
            raise ValueError('Wrong predecessor experiment.')
        verify_results(saved, folder / 'train', folder / 'submit')
    decision_path = previous / 'r07_decision.json'
    if not decision_path.exists():
        raise RuntimeError('R07 decision is still pending.')
    decision = json.loads(decision_path.read_text(encoding='utf-8'))['status']
    if decision == 'eligible':
        root = previous.with_name(previous.name + '-r07')
        state = json.loads((root / 'state.json').read_text(encoding='utf-8'))
        eid = 'r07_long_ft'
        job = state.get('jobs', {}).get(eid, {})
        if not job.get('verified') or any(not job.get(m, {}).get('downloaded') or
                not isinstance(job[m].get('version'), int) or job[m]['version'] < 1 or job[m].get('error')
                for m in ('train', 'submit')):
            raise RuntimeError('Eligible R07 has not completed.')
        folder = root / eid
        cfg = json.loads((folder / 'train/run/bundle/assets.json').read_text(encoding='utf-8'))['experiment']
        if cfg['id'] != eid:
            raise ValueError('Wrong conditional predecessor experiment.')
        verify_results(cfg, folder / 'train', folder / 'submit')
    elif decision != 'deferred':
        raise ValueError('Unknown R07 decision.')


def wait_and_execute(lock_path, status_path, poll_seconds, callback, sleep=time.sleep):
    while True:
        try:
            with exclusive_lock(lock_path):
                write_json(status_path, {'status': 'runner_lock_acquired'})
                return callback()
        except RunnerBusy:
            write_json(status_path, {'status': 'waiting_for_existing_runner'})
            logging.info('Waiting for the existing experiment runner; no upload performed.')
            sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='plane-depth-20261019')
    parser.add_argument('--wait-for-run', default='followup-20261019')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,65}', name)
            for name in (args.run_name, args.wait_for_run)) or args.poll_seconds < 10
            or args.run_name == args.wait_for_run):
        parser.error('Invalid run name/poll interval.')
    inputs = resolve_verified_inputs({'s11_cross': INPUTS['s11_cross']})
    configs = plane_depth_configs()
    root = ROOT / 'outputs/automation'
    output = root / args.run_name
    if args.dry_run:
        print('Verified frozen input: S11 train v' + str(inputs['s11_cross']['job']['train']['version']))
        print('Order:', ' -> '.join(c['id'] for c in configs))
        print('Output:', output)
        print('Require completed predecessor:', args.wait_for_run, '(including conditional R07)')
        print('Wait for shared lock; reuse S11 tokens; private GPU train -> offline submit -> verify.')
        print('No official leaderboard submission. No uploads in dry run.')
        return
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(output / 'runner.log', encoding='utf-8')])

    def execute():
        verify_finished_followup(root / args.wait_for_run)
        if plane_depth_configs() != configs:
            raise ValueError('Experiment sources changed while waiting. Restart this queue to review the new configuration.')
        pin = output / 'inputs.json'
        if pin.exists() and json.loads(pin.read_text(encoding='utf-8')) != inputs:
            raise ValueError('Pinned external input changed.')
        write_json(pin, inputs)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        def before_upload(config, mode, metadata):
            # Cache reuse avoids a multi-hour DICOM feature extraction pass.
            wait_for_gpu(api, .75 if mode == 'train' else .5, output / 'quota_wait.json')

        run(api, configs, output, args.poll_seconds, builder=build_plane_depth,
            external_jobs={k: v['job'] for k, v in inputs.items()}, before_upload=before_upload)
        from scripts.analyze_experiments import main as analyze
        analyze(ROOT / 'outputs/analysis' / args.run_name)
        write_json(output / 'queue_status.json', {'status': 'complete'})

    try:
        with exclusive_lock(root / 'plane-depth-waiter.lock'):
            wait_and_execute(root / 'runner.lock', output / 'queue_status.json', args.poll_seconds, execute)
    except RunnerBusy:
        parser.exit(1, 'Another plane-depth queue is already waiting/running.\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Local runner stopped; rerun the same command to resume pinned versions.\n')
    except Exception as exc:
        logging.error('%s: %s', type(exc).__name__, exc)
        write_json(output / 'queue_status.json', {'status': 'stopped', 'error_type': type(exc).__name__})
        parser.exit(1, f'Stopped; state preserved in {output}. No ambiguous upload is retried.\n')


if __name__ == '__main__':
    main()
