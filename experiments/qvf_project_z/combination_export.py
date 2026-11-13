"""Compact Z9 comparison and completed Z7 records; raw arrays stay local."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from experiments.qvf_project_a.train import REPO,sha256,write_json
from .coverage_export import export as export_parent


def export(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest['discarded_synthetic_smoke']:raise ValueError('Do not publish discarded smoke results')
    export_parent(Path(manifest['wait_for_study']))
    destination=REPO/'reports/qvf_benchmarks'
    record=json.loads((destination/'export_manifest.json').read_text())
    def copy(source,relative):
        if not source.exists():return
        if source.stat().st_size>5_000_000:raise ValueError('Oversized compact artifact: '+str(source))
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        record['artifacts']=[a for a in record['artifacts'] if a['export']!=relative]
        record['artifacts'].append({'source':str(source.relative_to(REPO)),'export':relative,
                                   'sha256':sha256(source),'bytes':source.stat().st_size})
    for name in ('manifest.json','comparison.json','comparison.csv','REPORT.md','plot_style.json',
                 'cifar32_comparison.png','cifar32_comparison.pdf','dogs256_comparison.png','dogs256_comparison.pdf',
                 'launch_validation.json','launch.json'):
        copy(root/name,'z9/'+name)
    for panel in manifest['panels']:
        for name in ('protocol.json','source_manifest.json','verification.json','quantum_audit.json'):
            copy(root/panel/name,'z9/'+panel+'/'+name)
    z7=REPO/'experiments/qvf_project_z/results/20261111_z7_coherent'
    for name in ('comparison.json','comparison.csv','REPORT.md','verification.json','extended_development.json',
                 'comparison.png','development_code_fitting.png'):
        copy(z7/name,'z7/'+name)
    for name in ('combination_tests_20261113.log','combination_preflight_20261113.json'):
        copy(root.parent/name,'validation/'+name)
    q=json.loads((root/'queue_status.json').read_text()) if (root/'queue_status.json').exists() else {'phase':'prepared'}
    write_json(destination/'z9/status_at_export.json',{'exported_utc':datetime.now(timezone.utc).isoformat(),
        'phase':q['phase'],'completed_fresh_runs':len(q.get('finished',[])),
        'active_fresh_runs':len(q.get('active',[])),'failed_runs':len(q.get('failed',[])),
        'note':'Static snapshot. Inspect the local queue for live progress.'})
    record['exported_utc']=datetime.now(timezone.utc).isoformat();write_json(destination/'export_manifest.json',record)
    index=destination/'README.md'
    index.write_text(index.read_text()+'''

- [Z7 completed comparison](z7/REPORT.md): all12 fresh runs and independent
  numerical/quantum checks finished; the older dated Z7 snapshot is historical.
- [Z9 full attention combination](z9/REPORT.md): adds per-scale attention to
  three quantum branches, comparing mean fusion, classical output attention,
  and coherent quantum scale routing. Both plots include all17 methods with
  vanilla QVF in red. Missing results remain pending.
- [Z9 export status](z9/status_at_export.json).

The Z9 queue waits for Z8, then trains18 new runs (three methods, two datasets,
three seeds) at50,000 updates. Its local5x5-token attention keeps per-query
retrieval bounded. The prior dense-attention5q reference has different encoding
and circuits, so that comparison does not isolate head count.

Refresh all current compact records with
`python -m experiments.qvf_project_z.combination_export --root LOCAL_Z9_ROOT`.
Exporting does not commit or push. Generated source snapshots and heavy data
remain local. The final comparison requires the Z8 and Z9 checks to pass.
''')
    print(json.dumps({'exported_files':len(record['artifacts']),'directory':str(destination)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    export(parser.parse_args().root)
