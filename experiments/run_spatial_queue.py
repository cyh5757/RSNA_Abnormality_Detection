"""Wait for verified E06-E12 completion, then run the spatial suite exclusively."""
import argparse
import contextlib
import json
import logging
import re
import time
from pathlib import Path

from experiments.build_notebooks import ROOT
from experiments.run_queue import run, verify_results
from experiments.spatial_configs import spatial_configs
from experiments.spatial_notebooks import build_spatial
from experiments.weak_configs import WEAK_IDS, weak_configs


class RunnerBusy(Exception):
    pass


@contextlib.contextmanager
def exclusive_lock(path):
    import msvcrt
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc: raise RunnerBusy(str(path)) from exc
        try: yield
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def predecessor_complete(run_dir):
    path=Path(run_dir)/'state.json'
    if not path.is_file(): return False
    state=json.loads(path.read_text(encoding='utf-8'))
    for eid in WEAK_IDS:
        job=state.get('jobs',{}).get(eid,{})
        if job.get('verified') is not True: return False
        for mode in ('train','submit'):
            stage=job.get(mode,{})
            if stage.get('downloaded') is not True or not isinstance(stage.get('version'),int) or stage['version']<1 or stage.get('error'):
                return False
    return True


def verify_predecessor(run_dir):
    if not predecessor_complete(run_dir): raise RuntimeError('Predecessor is incomplete.')
    for config in weak_configs():
        verify_results(config,Path(run_dir)/config['id']/'train',Path(run_dir)/config['id']/'submit')


def wait_and_run(previous,lock_path,poll_seconds,callback):
    while True:
        try:
            with exclusive_lock(lock_path):
                if not predecessor_complete(previous):
                    raise RuntimeError(f'Predecessor is incomplete and no runner is active: {previous}. Resume the weak-label queue first.')
                verify_predecessor(previous)
                logging.info('Predecessor verified. Starting spatial queue.')
                return callback()
        except RunnerBusy:
            logging.info('Waiting for predecessor completion and the experiment runner lock: %s',previous)
            time.sleep(poll_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='spatial')
    parser.add_argument('--wait-for-run',default='weak-labels')
    parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    for value in (args.run_name,args.wait_for_run):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value): parser.error('Invalid run name.')
    if args.run_name==args.wait_for_run or args.poll_seconds<10: parser.error('Use different run names and poll interval >=10s.')
    root=ROOT/'outputs/automation'; previous=root/args.wait_for_run; output=root/args.run_name
    configs=spatial_configs()
    if args.dry_run:
        print(f'Wait for ALL E06-E12 verified: {previous}; currently complete={predecessor_complete(previous)}')
        print(f'Results: {output}')
        for cfg in configs:
            print(f"{cfg['id']}: {cfg['representation']} neighbors={cfg['neighbors']} position={cfg['position']} fusion={cfg['fusion']} -> {cfg['train_dependencies']}")
        print('No uploads or waiting. Normal run: private GPU train -> offline submit -> verify; leaderboard submission is manual.')
        return
    output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[
        logging.StreamHandler(),logging.FileHandler(output/'runner.log',encoding='utf-8')])

    def execute():
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi(); api.authenticate()
        run(api,configs,output,args.poll_seconds,builder=build_spatial)
    try:
        with exclusive_lock(root/'spatial-waiter.lock'):
            wait_and_run(previous,root/'runner.lock',args.poll_seconds,execute)
    except KeyboardInterrupt:
        parser.exit(130,'Local runner stopped; saved Kaggle jobs can be resumed with the same command.\n')
    except RunnerBusy:
        parser.exit(1,'Another spatial queue is already waiting/running.\n')
    except Exception as exc:
        logging.error('%s: %s',type(exc).__name__,exc)
        parser.exit(1,f'Stopped. Review {output}/runner.log; existing results are preserved.\n')


if __name__=='__main__': main()
