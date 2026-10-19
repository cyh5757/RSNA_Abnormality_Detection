"""Verified historical inputs -> fixed R ablations -> offline CSV, with free-quota waiting."""
import argparse
import json
import logging
import re
import time
from pathlib import Path

from experiments.build_notebooks import ROOT
from experiments.core import file_hash
from experiments.followup_configs import INPUTS, followup_configs
from experiments.followup_notebooks import build_followup
from experiments.run_queue import run, verify_results, retry_kaggle_read, write_json
from experiments.run_spatial_queue import exclusive_lock, RunnerBusy

GPU_HOURS={'r04_confidence':.75,'r05_label_loss':.75,'r02_wide':6.,'r03_dense':7.,'r06_warmup':11.,'r07_long_ft':11.}


def resolve_verified_inputs(inputs=None):
    result={}
    for eid,root in (INPUTS if inputs is None else inputs).items():
        root=Path(root); state=json.loads((root/'state.json').read_text(encoding='utf-8'))
        job=state.get('jobs',{}).get(eid,{})
        if not job.get('verified') or any(not job.get(m,{}).get('downloaded') or
            not isinstance(job[m].get('version'),int) or job[m]['version']<1 or job[m].get('error') for m in ('train','submit')):
            raise ValueError(f'Unverified external input: {eid}')
        folder=root/eid; assets=folder/'train/run/bundle/assets.json'
        config=json.loads(assets.read_text(encoding='utf-8'))['experiment']
        if config['id']!=eid: raise ValueError('Wrong external configuration.')
        verify_results(config,folder/'train',folder/'submit')
        result[eid]=dict(job=job,config=config,assets_sha256=file_hash(assets),
            oof_sha256=file_hash(folder/'train/run/oof_predictions.csv'))
    return result


def wait_for_gpu(api,needed_hours,status_path,sleep=time.sleep):
    while True:
        quota=retry_kaggle_read(api.quota_view); gpu=quota.gpu_quota
        remaining=(gpu.total_time_allowed-gpu.time_used-gpu.time_reserved).total_seconds()/3600
        ready=remaining>=needed_hours
        write_json(status_path,dict(status='ready' if ready else 'waiting_for_free_gpu_quota',
            required_hours=needed_hours,remaining_hours=remaining,refresh_time=str(quota.quota_refresh_time)))
        if ready: return
        logging.info('Waiting for free GPU quota: %.2fh available, %.2fh required; refresh %s',remaining,needed_hours,quota.quota_refresh_time)
        sleep(600)


def r07_decision(result_dir,parent_dir):
    current=json.loads((result_dir/'metrics.json').read_text());parent=json.loads((parent_dir/'metrics.json').read_text())
    timing=json.loads((result_dir/'timings.json').read_text())
    # One R06 epoch measured end-to-end, including validation. Reserve head/export margin.
    projected_ft=3*timing['backbone_seconds']/3600
    estimate=projected_ft+timing['feature_seconds']/3600+1.
    wins=sum(a['macro_auc']>b['macro_auc'] for a,b in zip(current['runs'],parent['runs']))
    delta=current['macro_auc_mean']-parent['macro_auc_mean']
    drops={label:sum(r['per_label_auc'][label] for r in current['runs'])/3-
        sum(r['per_label_auc'][label] for r in parent['runs'])/3 for label in ('MCL','Lateral Meniscus','Fracture')}
    eligible=delta>=.005 and wins>=2 and min(drops.values())>=-.03 and estimate<=10.5 and projected_ft<=7.
    return dict(status='eligible' if eligible else 'deferred',estimated_hours=estimate,
        projected_ft_hours=projected_ft,mean_auc_gain=delta,seed_wins=wins,priority_category_deltas=drops,
        criteria='gain>=.005; >=2 seed wins; priority category drop<=.03; projected session<=10.5h; FT<=7h')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='followup-20261019')
    parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,65}',args.run_name) or args.poll_seconds<10:
        parser.error('Invalid run name/poll interval.')
    output=ROOT/'outputs/automation'/args.run_name
    inputs=resolve_verified_inputs(); configs=followup_configs()
    if args.dry_run:
        print('Verified inputs:',', '.join(f"{k} v{v['job']['train']['version']}" for k,v in inputs.items()))
        print('Order:', ' -> '.join(c['id'] for c in configs), '-> conditional R07')
        print('Output:',output)
        print('Private notebooks; offline submission.csv; free-quota wait; no official leaderboard submission.')
        return
    output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(output/'runner.log',encoding='utf-8')])
    try:
        with exclusive_lock(ROOT/'outputs/automation/runner.lock'):
            pin=output/'inputs.json'
            if pin.exists() and json.loads(pin.read_text())!=inputs: raise ValueError('Pinned external inputs changed.')
            write_json(pin,inputs)
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate()
            def before_upload(config,mode,metadata):
                if metadata['enable_gpu']:
                    wait_for_gpu(api,GPU_HOURS[config['id']] if mode=='train' else .5,output/'quota_wait.json')
            run(api,configs,output,args.poll_seconds,builder=build_followup,
                external_jobs={k:v['job'] for k,v in inputs.items()},before_upload=before_upload)
            decision=r07_decision(output/'r06_warmup/train/run',INPUTS['s12_finetune']/'s12_finetune/train/run')
            write_json(output/'r07_decision.json',decision)
            if decision['status']=='eligible':
                run(api,followup_configs(True)[-1:],output.with_name(output.name+'-r07'),args.poll_seconds,
                    builder=build_followup,external_jobs={k:v['job'] for k,v in inputs.items()},before_upload=before_upload)
            logging.info('Follow-up queue complete; R07 decision: %s',decision)
            from scripts.analyze_experiments import main as analyze
            analyze(ROOT/'outputs/analysis/2026-10-19')
    except RunnerBusy:
        parser.exit(1,'Another experiment runner owns the lock. No upload performed.\n')
    except KeyboardInterrupt:
        parser.exit(130,'Stopped locally; rerun the same shell to resume pinned Kaggle versions.\n')
    except Exception as exc:
        logging.error('%s: %s',type(exc).__name__,exc)
        parser.exit(1,f'Stopped; state preserved in {output}. No automatic reupload after ambiguous writes.\n')


if __name__=='__main__': main()
