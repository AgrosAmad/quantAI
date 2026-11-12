"""Z7 durable two-worker GPU0 queue with complete post-training analysis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .hires_study import assert_sources,atomic_json


def run(root,workers):
    root=Path(root).resolve();assert_sources(root)
    jobs=json.loads((root/'runs.json').read_text())
    # Launch primary comparisons first, then matched controls within each seed.
    order=['coherent_routing','real_routing','local_capacity9','classical_attention16']
    jobs.sort(key=lambda j:(j['seed'],order.index(j['architecture'])))
    pending=list(jobs);running=[];finished=[];failed=[];start=time.time()
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    while pending or running:
        while pending and len(running)<workers and not failed:
            job=pending.pop(0);directory=Path(job['directory'])
            log=(directory/'run.log').open('x')
            command=[sys.executable,'-m','experiments.qvf_project_z.routing_study','--root',str(root),'--architecture',job['architecture'],'--seed',str(job['seed'])]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
            running.append((process,log,job));print(f'START {job["architecture"]} seed={job["seed"]} pid={process.pid}',flush=True)
        remaining=[]
        for process,log,job in running:
            code=process.poll()
            if code is None:remaining.append((process,log,job));continue
            log.close();item={**job,'exit_code':code,'pid':process.pid}
            (finished if code==0 else failed).append(item)
            print(f'EXIT {job["architecture"]} seed={job["seed"]} code={code}',flush=True)
        running=remaining
        atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':'training','workers':workers,
            'active':[{**j,'pid':proc.pid} for proc,_,j in running],'pending':pending,
            'finished':finished,'failed':failed,'elapsed_seconds':time.time()-start})
        if failed and not running:raise RuntimeError(f'Queue stopped after failed runs: {failed}')
        time.sleep(5)
    for module in ('routing_verify','routing_audit','routing_inversion','routing_report'):
        atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':module,'finished':finished,'failed':[], 'elapsed_seconds':time.time()-start})
        with (root/(module+'.log')).open('x') as log:
            process=subprocess.run([sys.executable,'-m','experiments.qvf_project_z.'+module,'--root',str(root)],env=environment,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':'failed','failed_stage':module,'exit_code':process.returncode,'elapsed_seconds':time.time()-start})
            raise RuntimeError(f'{module} failed; inspect log')
        print(f'FINISHED {module}',flush=True)
    atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':'complete','finished':finished,'failed':[],'elapsed_seconds':time.time()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.workers<1 or a.workers>2:raise ValueError('One or two GPU0 workers')
    run(a.root,a.workers)
