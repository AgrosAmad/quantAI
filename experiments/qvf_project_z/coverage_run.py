"""Detached GPU0 queue; waits for Z7 completion before starting Z8."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .hires_study import atomic_json,assert_sources


def run(root,workers=2):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    for panel in m['panels']:assert_sources(root/panel)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg')
    start=time.time()
    if not m['discarded_synthetic_smoke']:
        parent=Path(m['wait_for_study'])
        while True:
            q=json.loads((parent/'queue_status.json').read_text())
            if q['phase']=='complete':break
            if q.get('failed') or q['phase']=='failed' or not Path('/proc',str(q['runner_pid'])).exists():
                atomic_json(root/'queue_status.json',{'phase':'failed_dependency','dependency':str(parent),'parent_status':q})
                raise RuntimeError('Z7 did not finish successfully; existing work was not interrupted')
            atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':'waiting_for_z7',
                'dependency':str(parent),'fresh_runs':len(m['fresh_runs']),'elapsed_seconds':time.time()-start})
            time.sleep(15)
        for folder in m['reused_roots']:
            assert_sources(Path(folder))
            for filename in ('verification.json','quantum_audit.json'):
                assert json.loads((Path(folder)/filename).read_text())['complete']
    jobs=sorted(m['fresh_runs'],key=lambda j:(0 if j['panel']=='dogs256' else 1,j['seed'],m['methods'].index(j['architecture'])))
    pending=list(jobs);active=[];finished=[];failed=[]
    def update_report():
        with (root/'partial_report.log').open('a') as log:
            subprocess.run([sys.executable,'-m','experiments.qvf_project_z.coverage_report','--root',str(root),'--partial'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    while pending or active:
        while pending and len(active)<workers and not failed:
            job=pending.pop(0);log=(Path(job['directory'])/'run.log').open('x')
            command=[sys.executable,'-m','experiments.qvf_project_z.coverage_study','--root',job['protocol_root'],'--architecture',job['architecture'],'--seed',str(job['seed'])]
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((proc,log,job));print('START',job['panel'],job['architecture'],job['seed'],proc.pid,flush=True)
        remaining=[];changed=False
        for proc,log,job in active:
            code=proc.poll()
            if code is None:remaining.append((proc,log,job));continue
            log.close();item={**job,'pid':proc.pid,'exit_code':code};(failed if code else finished).append(item);changed=True
            print('EXIT',job['panel'],job['architecture'],job['seed'],code,flush=True)
        active=remaining
        atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':'training','workers':workers,
            'active':[{**j,'pid':p.pid} for p,_,j in active],'pending':pending,'finished':finished,'failed':failed,
            'elapsed_seconds':time.time()-start})
        if changed:update_report()
        if failed and not active:raise RuntimeError('Failed runs preserved; queue stopped')
        time.sleep(1 if m['discarded_synthetic_smoke'] else 5)
    for panel in m['panels']:
        for stage in ('coverage_verify','coverage_audit'):
            atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':stage,'panel':panel,
                'finished':finished,'failed':[],'elapsed_seconds':time.time()-start})
            with (root/(panel+'_'+stage+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-m','experiments.qvf_project_z.'+stage,'--root',str(root/panel)],env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(root/'queue_status.json',{'phase':'failed','failed_stage':stage,'panel':panel,'exit_code':result.returncode})
                raise RuntimeError(stage+' failed; logs preserved')
    with (root/'final_report.log').open('x') as log:
        subprocess.run([sys.executable,'-m','experiments.qvf_project_z.coverage_report','--root',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(root/'queue_status.json',{'runner_pid':os.getpid(),'phase':'complete','finished':finished,
        'failed':[],'elapsed_seconds':time.time()-start,'discarded_synthetic_smoke':m['discarded_synthetic_smoke']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if not 1<=args.workers<=2:raise ValueError('One or two GPU0 workers')
    run(args.root,args.workers)
