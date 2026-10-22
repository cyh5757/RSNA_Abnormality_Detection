"""Finite private Kaggle campaign. Ordinary Python only; no agent/LLM calls."""
import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

from experiments.build_notebooks import ROOT
from experiments.core import file_hash
from experiments.run_queue import run, write_json
from experiments.run_spatial_queue import exclusive_lock, RunnerBusy
from experiments.run_followup_queue import resolve_verified_inputs, wait_for_gpu, r07_decision, GPU_HOURS
from experiments.followup_configs import INPUTS, followup_configs
from experiments.followup_notebooks import build_followup
from experiments.anatomy_configs import anatomy_configs
from experiments.run_anatomy_queue import run_adaptation_stages, checked_builder
from experiments.campaign_configs import CAMPAIGN_INPUTS, campaign_configs
from experiments.campaign_notebooks import build_campaign


def source_snapshot():
    files = list((ROOT/'experiments').glob('*.py'))
    files += [ROOT/'scripts'/name for name in ('analyze_experiments.py', 'compare_confusions.py', 'run_research_campaign.ps1')]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(files)}


def pin_campaign(root, payload):
    path = Path(root)/'inputs.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != payload:
        raise ValueError('Pinned campaign input/source changed; inspect before starting a different campaign.')
    write_json(path, payload)


def campaign_gpu_hours(config, mode):
    if mode == 'submit': return .5
    if config.get('ensemble'): return 0.
    if config['id'] == 'v05_convnext': return 8.
    if config.get('head_seed_offsets'): return 2.
    return 1.25


def guarded_gpu_wait(api, hours, path, check):
    check()
    if hours: wait_for_gpu(api, hours, path)
    check()


def guarded_adaptation_stages(api, config, root, state, poll_seconds, check):
    namespace = run_adaptation_stages.__globals__
    previous = namespace['checked_gpu_wait']
    def guarded_wait(*args):
        check()
        previous(*args)
        check()
    # Preserve the historical anatomy source digest while extending its upload guard.
    namespace['checked_gpu_wait'] = guarded_wait
    try:
        check()
        run_adaptation_stages(api, config, root, state, poll_seconds)
    finally:
        namespace['checked_gpu_wait'] = previous


def verify_followup_resume(configs, root):
    path = root/'state.json'
    if not path.exists(): return
    names = ('core.py', 'backbones.py', 'train.py', 'submit.py', 'build_notebooks.py',
             'weak_core.py', 'weak_train.py', 'weak_submit.py', 'weak_configs.py')
    sources = b''.join((ROOT/'experiments'/name).read_bytes() for name in names)
    expected = hashlib.sha256(sources + json.dumps(configs, sort_keys=True).encode()).hexdigest()
    if json.loads(path.read_text())['fingerprint'] != expected:
        raise ValueError('Historical R queue source changed; cannot resume automatically.')


def refresh_reports(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    paths = list((ROOT/'outputs/experiments').glob('*/train/run/metrics.json'))
    paths += list((ROOT/'outputs/automation').glob('*/*/train/run/metrics.json'))
    partial = [str(p.parent.parent.parent) for p in paths
               if not (p.parent.parent.parent/'submit/inference_summary.json').exists()]
    if partial:
        write_json(output/'analysis_status.json', dict(status='waiting_for_artifacts', partial=partial))
        return False
    from scripts.analyze_experiments import main as analyze
    from scripts.compare_confusions import write_comparisons
    analyze(output); write_comparisons(output)
    write_json(output/'analysis_status.json', dict(status='complete'))
    return True


def execute_campaign(api, root, inputs, v_configs, r_configs, u_configs, snapshot, poll_seconds):
    def check():
        if source_snapshot() != snapshot: raise ValueError('Campaign source changed during execution.')
    jobs = {eid: item['job'] for eid, item in inputs.items()}
    analysis = ROOT/'outputs/analysis'/root.name
    def status(phase, eid=None):
        write_json(root/'queue_status.json', dict(status='running', phase=phase, experiment=eid))
    def one(config, builder, hours):
        check(); status('V' if config['id'].startswith('v') else 'U', config['id'])
        jobroot = root.with_name(root.name+'-'+config['id'])
        def before_upload(cfg, mode, metadata):
            guarded_gpu_wait(api, hours(cfg, mode) if metadata['enable_gpu'] else 0,
                             root/'quota_wait.json', check)
        def build(configs, output_root, training_version=None):
            check(); builder(configs, output_root, training_version); check()
        run(api, [config], jobroot, poll_seconds, builder=build,
            external_jobs={eid: jobs[eid] for eid in config['train_dependencies']}, before_upload=before_upload)
        jobs[config['id']] = json.loads((jobroot/'state.json').read_text())['jobs'][config['id']]
        write_json(root/'completed_jobs.json', jobs)
        refresh_reports(analysis)
    for cfg in v_configs:
        one(cfg, build_campaign, campaign_gpu_hours)

    check(); status('R')
    r_root = ROOT/'outputs/automation/followup-20261019'
    verify_followup_resume(r_configs, r_root)
    def r_before(cfg, mode, metadata):
        status('R', cfg['id'])
        write_json(r_root/'queue_status.json', dict(status='running', campaign=root.name, experiment=cfg['id']))
        guarded_gpu_wait(api, (GPU_HOURS[cfg['id']] + .5 if mode == 'train' else .5)
                         if metadata['enable_gpu'] else 0, root/'quota_wait.json', check)
    def r_build(configs, output_root, training_version=None):
        check(); build_followup(configs, output_root, training_version); check()
    run(api, r_configs, r_root, poll_seconds, builder=r_build,
        external_jobs={eid: inputs[eid]['job'] for eid in INPUTS}, before_upload=r_before)
    decision = r07_decision(r_root/'r06_warmup/train/run', INPUTS['s12_finetune']/'s12_finetune/train/run')
    write_json(r_root/'r07_decision.json', decision)
    if decision['status'] == 'eligible':
        run(api, followup_configs(True)[-1:], r_root.with_name(r_root.name+'-r07'), poll_seconds,
            builder=r_build, external_jobs={eid: inputs[eid]['job'] for eid in INPUTS}, before_upload=r_before)
    write_json(r_root/'queue_status.json', dict(status='complete', campaign=root.name, r07=decision['status']))
    refresh_reports(analysis)

    adapt_root = root.with_name(root.name+'-adapt'); adapt_root.mkdir(exist_ok=True)
    state_path = adapt_root/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(jobs={'s11_cross': jobs['s11_cross']})
    def u_hours(cfg, mode):
        if mode == 'submit': return .5
        return 1.5 if cfg['id'] in ('u03_patch_attention', 'u06_flip_tta') or cfg.get('ensemble') else 7.5
    for cfg in u_configs:
        check(); status('U', cfg['id'])
        if cfg.get('adapt_stage') and not cfg.get('ensemble'):
            guarded_adaptation_stages(api, cfg, adapt_root, state, poll_seconds, check)
            jobs.update(state['jobs'])
        one(cfg, checked_builder, u_hours)
        state['jobs'][cfg['id']] = jobs[cfg['id']]; write_json(state_path, state)
    complete = refresh_reports(analysis)
    write_json(root/'queue_status.json', dict(status='complete' if complete else 'complete_reports_pending',
               analysis=str(analysis), r07=decision['status'], automatic_agent_monitoring=False))
    logging.info('Campaign finished. No automatic agent invocation; results: %s', analysis)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='research-campaign-20261021')
    parser.add_argument('--poll-seconds', type=int, default=120)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', args.run_name) or args.poll_seconds < 10:
        parser.error('Invalid run name/poll interval.')
    root = ROOT/'outputs/automation'/args.run_name
    inputs = resolve_verified_inputs(CAMPAIGN_INPUTS)
    v_configs, r_configs, u_configs = campaign_configs(), followup_configs(), anatomy_configs()
    verify_followup_resume(r_configs, ROOT/'outputs/automation/followup-20261019')
    snapshot = source_snapshot()
    payload = dict(inputs=inputs, v=v_configs, r=r_configs, u=u_configs, sources=snapshot)
    if args.dry_run:
        print('Verified inputs:', ', '.join(inputs))
        print('New V order:', ' -> '.join(cfg['id'] for cfg in v_configs))
        print('Resume R02 -> R03 -> R06 -> conditional R07; then U00 -> U08 (12 adaptation segments).')
        print('Free quota wait; private notebooks; offline CSV; no leaderboard submission; no LLM calls.')
        if (root/'inputs.json').exists() and json.loads((root/'inputs.json').read_text()) != payload:
            raise ValueError('Existing campaign pin differs.')
        return
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(root/'runner.log', encoding='utf-8')])
    try:
        with exclusive_lock(ROOT/'outputs/automation/runner.lock'):
            pin_campaign(root, payload)
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            execute_campaign(api, root, inputs, v_configs, r_configs, u_configs, snapshot, args.poll_seconds)
    except RunnerBusy:
        parser.exit(1, 'Another runner owns the lock; no upload performed.\n')
    except KeyboardInterrupt:
        write_json(root/'queue_status.json', dict(status='stopped', reason='local_interrupt'))
        parser.exit(130, 'Local runner stopped; same command resumes pinned jobs.\n')
    except Exception as exc:
        logging.exception('Campaign stopped; saved versions will not be reuploaded blindly.')
        write_json(root/'queue_status.json', dict(status='stopped', error_type=type(exc).__name__, reason=str(exc)))
        parser.exit(1, f'State preserved: {root}\n')


if __name__ == '__main__': main()
