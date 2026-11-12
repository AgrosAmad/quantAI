"""Bounded GPU0 job queue followed by independent Z5 audits and reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_state(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2)+'\n')
    os.replace(tmp, path)


def execute(root, module, args, label):
    for attempt in range(3):
        path = root / (label + (f'_retry{attempt}' if attempt else '') + '.log')
        print(f'Starting {label}, attempt {attempt+1}', flush=True)
        with path.open('x') as log:
            result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            print(f'Completed {label}', flush=True)
            return
        if 'CUDA was requested but is unavailable' not in path.read_text():
            raise RuntimeError(f'{label} failed: {path}')
        print(f'Transient CUDA initialization failure; preserving {path}', flush=True)
        time.sleep(3)
    raise RuntimeError(f'{label} failed CUDA initialization three times')


def run(root, audit, workers=6):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not 1 <= workers <= 6:
        raise ValueError('GPU0 only; at most six concurrent processes')
    root, audit = Path(root).resolve(), Path(audit).resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['kind'] != 'qvf_project_z_scale_v1':
        raise ValueError('Expected the frozen Z5 protocol')
    priority = ['attention_chain_q5', 'spatial_star_q5', 'attention_chain_q7',
        'spatial_star_q7', 'attention_star_q7', 'center_star_q7',
        'attention_star_q5', 'center_star_q5', 'attention_chain_wide_q5']
    queue = sorted(protocol['runs'], key=lambda r: (r['config']['seed'], priority.index(r['name'])))
    complete = [r['directory'] for r in queue if (root / r['directory'] / 'COMPLETE.json').exists()]
    queue = [r for r in queue if r['directory'] not in complete]
    if any((root / r['directory']).exists() for r in queue):
        raise RuntimeError('Partial runs are preserved; inspect them rather than silently restarting')
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    active, failures, attempts = {}, [], {}
    started = time.time()
    try:
        while queue or active:
            for key, entry in list(active.items()):
                proc, log, spec, path = entry
                code = proc.poll()
                if code is None:
                    continue
                log.close(); del active[key]
                if code == 0 and (root / key / 'COMPLETE.json').exists():
                    complete.append(key)
                    print(f'Completed {key}: {len(complete)}/{len(protocol["runs"])}', flush=True)
                elif not (root / key).exists() and 'CUDA was requested but is unavailable' in path.read_text() and attempts[key] < 3:
                    queue.insert(0, spec)
                    print(f'Retrying CUDA initialization for {key}; preserving {path}', flush=True)
                else:
                    failures.append({'run': key, 'exit_code': code, 'log': str(path)})
                    print(f'FAILED {key}: see {path}', flush=True)
            while queue and len(active) < workers:
                spec = queue.pop(0); key = spec['directory']
                attempts[key] = attempts.get(key, 0)+1
                path = logs / (key + (f'_attempt{attempts[key]}' if attempts[key] > 1 else '') + '.log')
                log = path.open('x')
                args = [sys.executable, '-m', 'experiments.qvf_project_z.scale_study', 'run',
                    '--root', str(root), '--seed', str(spec['config']['seed']), '--stage', 'selected',
                    '--name', spec['name'], '--device', 'cuda:0']
                proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                active[key] = (proc, log, spec, path)
                print(f'Started {key}, PID {proc.pid}', flush=True)
                time.sleep(1)
            write_state(root / 'runner_state.json', {'started_unix': started, 'updated_unix': time.time(),
                'workers': workers, 'complete': complete, 'failures': failures,
                'running': {k: {'pid': v[0].pid, 'log': str(v[3])} for k, v in active.items()},
                'queued': [r['directory'] for r in queue]})
            if queue or active:
                time.sleep(5)
        if failures:
            raise RuntimeError(f'{len(failures)} training runs failed; see runner_state.json')
        execute(root, 'experiments.qvf_project_z.scale_verify', ['--root', root], 'verify')
        execute(root, 'experiments.qvf_project_z.scale_report', ['--root', root], 'report')
        execute(root, 'experiments.qvf_project_z.scale_feasibility', ['--root', audit, '--study-root', root], 'feasibility')
        execute(root, 'experiments.qvf_project_z.scale_audit_verify', ['--root', audit], 'audit_verify')
        execute(root, 'experiments.qvf_project_z.scale_decision', ['--root', root, '--audit', audit], 'decision')
        execute(root, 'experiments.qvf_project_z.scale_comparison', ['--root', root], 'pilot_comparison')
        write_state(root / 'PIPELINE_COMPLETE.json', {'training_runs': len(complete),
            'independent_verification': 'passed', 'quantum_audit': str(audit)})
    except BaseException as error:
        write_state(root / 'PIPELINE_FAILED.json', {'error': str(error), 'type': type(error).__name__})
        raise
    finally:
        # On an external interruption, only children launched by this runner are
        # stopped; user GPU1 jobs are neither selected nor signalled.
        for proc, log, _, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
            log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    run(args.root, args.audit, args.workers)
