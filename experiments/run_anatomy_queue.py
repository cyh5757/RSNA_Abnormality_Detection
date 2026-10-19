"""Run U ablations only after R -> T completes; preserve pinned uploads on resume."""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from experiments.build_notebooks import ROOT
from experiments.followup_configs import INPUTS
from experiments.run_followup_queue import resolve_verified_inputs,wait_for_gpu
from experiments.run_queue import run,write_json,verify_results,check_version,wait_for_job,retry_kaggle_read
from experiments.run_spatial_queue import exclusive_lock,RunnerBusy
from experiments.anatomy_configs import anatomy_configs
from experiments.anatomy_notebooks import build_anatomy,build_adaptation
from experiments.anatomy_adapt import adapt_recipe


def assert_current_source(config):
    current={c['id']:c for c in anatomy_configs()}
    if current.get(config['id'])!=config:
        raise ValueError('Experiment source/configuration changed after pinning. Review and start a new run name.')


def checked_builder(configs,output_root,training_version=None):
    for config in configs: assert_current_source(config)
    build_anatomy(configs,output_root,training_version)
    for config in configs: assert_current_source(config)


def checked_gpu_wait(api,config,hours,status_path):
    assert_current_source(config)
    wait_for_gpu(api,hours,status_path)
    # The wait can span days: never upload live edited code under an old digest.
    assert_current_source(config)


def predecessor_ready(root):
    path=root/'queue_status.json'
    if not path.exists(): return False
    status=json.loads(path.read_text(encoding='utf-8'))['status']
    if status=='stopped': raise RuntimeError('Predecessor queue stopped. Resume it before this queue.')
    if status!='complete': return False
    for eid in ('t00_position_control','t01_position_sum','t02_position_joint'):
        resolve_verified_inputs({eid:root})
    return True


def adaptation_stages(config):
    end=config['backbone_epochs'];start=1 if end==1 else 2
    stages=[]
    for epoch in range(start,end+1):
        for segment in range(4):
            previous=None if (epoch,segment)==(1,0) else (f'a{epoch:02d}_s{segment:02d}' if segment else f'a{epoch-1:02d}_s04')
            stages.append(dict(id=f'a{epoch:02d}_s{segment+1:02d}',epoch=epoch,segment=segment,previous=previous))
    return stages


def run_adaptation_stages(api,config,root,state,poll_seconds):
    for stage in adaptation_stages(config):
        assert_current_source(config)
        eid=stage['id'];dependencies=['u03_patch_attention']+([stage['previous']] if stage['previous'] else [])
        sources=[]
        for dependency in dependencies:
            prior=state['jobs'][dependency]
            if not prior.get('verified'): raise ValueError('Unverified adaptation predecessor.')
            job=prior['train'];check_version(api,job['ref'],job['version']);sources.append(f"{job['ref']}/{job['version']}")
        folder=root/'adapt_notebooks'/eid;metadata=build_adaptation(config,stage,folder,sources)
        assert_current_source(config)
        jobs=state.setdefault('adapt_jobs',{})
        if eid not in jobs:
            checked_gpu_wait(api,config,4. if stage['segment']==3 else 3.,root/'quota_wait.json')
            jobs[eid]=dict(ref=metadata['id'],version=None,downloaded=False);write_json(root/'state.json',state)
            result=api.kernels_push(str(folder))
            jobs[eid]['version']=result.version_number or None
            invalid=any(getattr(result,k) for k in ('invalid_kernel_sources','invalid_dataset_sources','invalid_competition_sources','invalid_model_sources'))
            jobs[eid]['error']=result.error or ('Kaggle rejected an input source.' if invalid else '')
            write_json(root/'state.json',state)
        job=jobs[eid]
        if job.get('error') or not job['version']: raise RuntimeError(f'Unresolved adaptation upload {eid}; inspect state before retrying.')
        output=root/'adaptation'/eid
        if not job['downloaded']:
            wait_for_job(api,job['ref'],job['version'],output,poll_seconds);check_version(api,job['ref'],job['version'])
            retry_kaggle_read(lambda:api.kernels_output(job['ref'],str(output),quiet=True,force=True,file_pattern=r'progress\.json$'))
            check_version(api,job['ref'],job['version'])
        progress=json.loads((output/'adapt/progress.json').read_text(encoding='utf-8'))
        if (progress['recipe']!=adapt_recipe(config) or (progress['epoch'],progress['segment'])!=(stage['epoch'],stage['segment'])
            or progress['complete'] is not True or progress['gold_studies_used_for_backbone']!=0):
            raise ValueError('Adaptation output does not match the requested weak-only stage.')
        job['downloaded']=True;state['jobs'][eid]=dict(verified=True,train=dict(job));write_json(root/'state.json',state)


def execute_suite(api,configs,root,inputs,poll_seconds):
    state_path=root/'state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else dict(configs=configs,jobs={k:v['job'] for k,v in inputs.items()})
    if state['configs']!=configs: raise ValueError('Pinned source/configuration changed; use a reviewed new run name.')
    write_json(state_path,state)
    for cfg in configs:
        assert_current_source(cfg)
        write_json(root/'queue_status.json',dict(status='running',experiment=cfg['id']))
        if cfg.get('adapt_stage') and not cfg.get('ensemble'): run_adaptation_stages(api,cfg,root,state,poll_seconds)
        # One ordinary run per experiment permits adaptation between U03 and U04/U05.
        # Flat automation directories remain discoverable by the existing analysis script.
        jobroot=root.with_name(root.name+'-'+cfg['id'])
        def before_upload(config,mode,metadata):
            hours=.5 if mode=='submit' else (1. if config['id'] in ('u03_patch_attention','u06_flip_tta') or config.get('ensemble') else 7.)
            checked_gpu_wait(api,config,hours,root/'quota_wait.json')
        run(api,[cfg],jobroot,poll_seconds,builder=checked_builder,
            external_jobs={eid:state['jobs'][eid] for eid in cfg['train_dependencies']},before_upload=before_upload)
        job=json.loads((jobroot/'state.json').read_text(encoding='utf-8'))['jobs'][cfg['id']]
        state['jobs'][cfg['id']]=job;write_json(state_path,state)
        # Only verified train+submit artifacts are included; no GPU run is launched by analysis.
        from scripts.analyze_experiments import main as analyze
        from scripts.compare_confusions import write_comparisons
        destination=ROOT/'outputs/analysis'/root.name
        analyze(destination);write_comparisons(destination)
    write_json(root/'queue_status.json',dict(status='complete',experiments=[c['id'] for c in configs]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='anatomy-20261019');parser.add_argument('--wait-for-run',default='plane-depth-20261019')
    parser.add_argument('--poll-seconds',type=int,default=60);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',name) for name in (args.run_name,args.wait_for_run)) or args.poll_seconds<10 or args.run_name==args.wait_for_run:
        parser.error('Invalid run names or polling interval.')
    inputs=resolve_verified_inputs({'s11_cross':INPUTS['s11_cross']});configs=anatomy_configs()
    root=ROOT/'outputs/automation'/args.run_name;prior=root.parent/args.wait_for_run
    if args.dry_run:
        print('Order:', ' -> '.join(c['id'] for c in configs));print('Predecessor:',prior)
        print('Full encoder adaptation: 4 + 8 pinned segment jobs, then OOF + offline submit.')
        print('9 private train/submit pairs; no official leaderboard submission; no uploads in dry run.');return
    root.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(root/'runner.log',encoding='utf-8')])
    try:
        with exclusive_lock(root.parent/'anatomy-waiter.lock'):
            pin=root/'inputs.json'
            payload=dict(inputs=inputs,configs=configs)
            if pin.exists() and json.loads(pin.read_text(encoding='utf-8'))!=payload: raise ValueError('Pinned inputs or sources changed.')
            write_json(pin,payload)
            while True:
                if predecessor_ready(prior):
                    try:
                        with exclusive_lock(root.parent/'runner.lock'):
                            if not predecessor_ready(prior): raise ValueError('Predecessor state changed.')
                            if configs!=anatomy_configs(): raise ValueError('Sources changed while waiting; review and restart with a new run name.')
                            from kaggle.api.kaggle_api_extended import KaggleApi
                            api=KaggleApi();api.authenticate()
                            execute_suite(api,configs,root,inputs,args.poll_seconds);return
                    except RunnerBusy: pass
                write_json(root/'queue_status.json',dict(status='waiting_for_predecessor',predecessor=args.wait_for_run,
                    order=[c['id'] for c in configs]))
                logging.info('Waiting for verified R -> T completion; no Kaggle upload yet.')
                time.sleep(args.poll_seconds)
    except RunnerBusy:
        parser.exit(1,'Another U queue already owns the waiter lock.\n')
    except KeyboardInterrupt:
        parser.exit(130,'Local runner stopped; rerun to resume pinned jobs.\n')
    except Exception as exc:
        logging.exception('U queue stopped')
        write_json(root/'queue_status.json',dict(status='stopped',error_type=type(exc).__name__))
        parser.exit(1,f'Stopped; state preserved in {root}.\n')


if __name__=='__main__': main()
