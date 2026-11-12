"""Export compact, explicit research artifacts for Git; raw runs stay local."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from experiments.qvf_project_a.train import REPO,sha256,write_json


def export(root):
    root=Path(root).resolve();destination=REPO/'reports/qvf_benchmarks'
    destination.mkdir(parents=True,exist_ok=True);copied=[]
    def copy(source,relative):
        source=Path(source)
        if not source.exists():return
        if source.stat().st_size>5_000_000:raise ValueError('Oversized curated artifact: '+str(source))
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        copied.append({'source':str(source.relative_to(REPO)),'export':relative,'sha256':sha256(source),'bytes':source.stat().st_size})
    z=REPO/'experiments/qvf_project_z/results'
    copy(REPO/'experiments/qvf_repro/results/20261108_vanilla/verified_results.json','vanilla_reproduction/results.json')
    copy(z/'20261110_z5_full/verified/summary.json','z5/summary.json')
    copy(z/'20261110_z6_hires/comparison.json','z6/comparison.json')
    copy(z/'20261110_z6_hires/protocol.json','z6/protocol.json')
    copy(z/'20261110_z6_hires/source_manifest.json','z6/source_manifest.json')
    copy(z/'20261110_z6_hires/figures/primary_comparison.png','z6/comparison.png')
    for name in ('protocol.json','source_manifest.json','STATUS_20261112_0411UTC.json',
                 'STATUS_20261112_0411UTC_with_QVF.json','STATUS_20261112_0411UTC_with_QVF.png'):
        copy(z/'20261111_z7_coherent'/name,'z7/'+name)
    for name in ('manifest.json','comparison.json','comparison.csv','REPORT.md',
                 'cifar32_comparison.png','dogs256_comparison.png','cifar32_comparison.pdf','dogs256_comparison.pdf',
                 'launch_validation.json','launch.json'):
        copy(root/name,'z8/'+name)
    for panel in ('cifar32','dogs256'):
        for name in ('protocol.json','source_manifest.json'):
            copy(root/panel/name,'z8/'+panel+'/'+name)
    for name in ('coverage_tests_20261112.log','coverage_preflight_20261112.json'):
        copy(z/name,'validation/'+name)
    queue=json.loads((root/'queue_status.json').read_text()) if (root/'queue_status.json').exists() else {'phase':'prepared'}
    write_json(destination/'z8/status_at_export.json',{'exported_utc':datetime.now(timezone.utc).isoformat(),
        'phase':queue['phase'],'completed_fresh_runs':len(queue.get('finished',[])),
        'active_fresh_runs':len(queue.get('active',[])),'failed_runs':len(queue.get('failed',[])),
        'note':'Static Git snapshot. Live state is in the local study queue_status.json.'})
    write_json(destination/'export_manifest.json',{'exported_utc':datetime.now(timezone.utc).isoformat(),'artifacts':copied})
    (destination/'README.md').write_text('''# QVF benchmark records

This directory contains compact research artifacts exported from the local
experiments. Source code and experiment instructions are in
[experiments](../../experiments/README.md). Datasets, checkpoints, raw arrays,
source snapshots, live queues and large galleries remain local.

- [Original QVF reproduction](vanilla_reproduction/results.json): verified
  single-image and 50-image fits. These are not the current larger-data benchmark.
- [Z5](z5/summary.json): completed 32×32 spatial/circuit study.
- [Z6](z6/comparison.json): completed 256×256 quantum-design study.
- [Z7 historical snapshot](z7/STATUS_20261112_0411UTC.json): provisional first-two-seed
  routing results. Do not treat this dated snapshot as the final Z7 result.
- [Z8 complete-method comparison](z8/REPORT.md): all13 Z6/Z7 methods plus vanilla
  QVF at both resolutions. Missing measurements are explicitly marked pending.
  The matched protocol uses50,000 updates and three seeds per method/dataset.
- [Status when exported](z8/status_at_export.json) and
  [artifact source hashes](export_manifest.json).

The current training recipe is common within each dataset. Vanilla QVF retains
its64-value global code; spatial methods have336 values at32px and21,504 at256px.
The two datasets differ, so cross-panel score differences are not a controlled
resolution comparison. No quantum advantage or hardware execution is claimed.

Run `python -m experiments.qvf_project_z.coverage_export --root LOCAL_Z8_ROOT`
after the full queue finishes to refresh these static records before committing
new results. Exporting does not commit or push automatically.
''')
    print(json.dumps({'exported_files':len(copied),'total_bytes':sum(x['bytes'] for x in copied),'directory':str(destination)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);export(parser.parse_args().root)
