"""Compact Z10 larger-code control and all prior records; raw arrays stay local."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from experiments.qvf_project_a.train import REPO,sha256,write_json
from .combination_export import export as export_parent


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
        copy(root/name,'z10/'+name)
    for panel in manifest['panels']:
        for name in ('protocol.json','source_manifest.json','verification.json','quantum_audit.json'):
            copy(root/panel/name,'z10/'+panel+'/'+name)
    z8=REPO/'experiments/qvf_project_z/results/20261112_z8_complete_methods'
    for panel in ('cifar32','dogs256'):
        for name in ('verification.json','quantum_audit.json'):
            copy(z8/panel/name,'z8/'+panel+'/'+name)
    copy(z8/'plot_style.json','z8/plot_style.json')
    for name in ('code_budget_tests_20261113.log','code_budget_preflight_20261113.json'):
        copy(root.parent/name,'validation/'+name)
    q=json.loads((root/'queue_status.json').read_text()) if (root/'queue_status.json').exists() else {'phase':'prepared'}
    write_json(destination/'z10/status_at_export.json',{'exported_utc':datetime.now(timezone.utc).isoformat(),
        'phase':q['phase'],'completed_fresh_runs':len(q.get('finished',[])),
        'active_fresh_runs':len(q.get('active',[])),'failed_runs':len(q.get('failed',[])),
        'note':'Static snapshot. Inspect the local queue for live progress.'})
    record['exported_utc']=datetime.now(timezone.utc).isoformat();write_json(destination/'export_manifest.json',record)
    index=destination/'README.md'
    index.write_text(index.read_text()+"""

- [Z10 larger-code vanilla QVF](z10/REPORT.md): adds256 global code entries
  with the original five-qubit QVF circuit and RGB readout unchanged.
  Both64-value and256-value QVF controls are red in the18-method figures.
- [Z10 export status](z10/status_at_export.json).

Z8 has completed all84 entries and independent metric/quantum checks.
Z10 preserves the active Z9 study and waits for its full pipeline, then trains
six fresh runs: two datasets, three seeds,50,000 updates each. A larger input
adds24,576 classical first-layer weights. This is a code-size control, not
an equal-storage comparison with the spatial methods.

Refresh all compact records with
`python -m experiments.qvf_project_z.code_budget_export --root LOCAL_Z10_ROOT`.
Missing measurements stay pending; raw arrays, checkpoints and source snapshots
stay local. Exporting does not automatically commit or push.
""")
    print(json.dumps({'exported_files':len(record['artifacts']),'directory':str(destination)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    export(parser.parse_args().root)
