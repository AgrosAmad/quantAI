"""Both resolution panels always list all14 methods, including vanilla QVF."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.qvf_project_a.train import write_json,sha256
from .routing_report import LABELS

LABELS={**LABELS,'vanilla_qvf':'Vanilla QVF (64-value code)'}


def report(root,partial=False):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    if not partial:
        for panel in m['panels']:
            for file in ('verification.json','quantum_audit.json'):
                assert json.loads((root/panel/file).read_text())['complete']
        if not m['discarded_synthetic_smoke']:
            for folder,digest in m['reused_protocol_hashes'].items():
                source=Path(folder);assert sha256(source/'protocol.json')==digest
                assert json.loads((source/'queue_status.json').read_text())['phase']=='complete'
                for file in ('verification.json','quantum_audit.json'):
                    assert json.loads((source/file).read_text())['complete']
    rows=[];raw=[]
    for item in m['coverage']:
        if m['discarded_synthetic_smoke'] and item['reused']:continue
        path=Path(item['directory'])/'summary.json'
        if not path.exists():continue
        s=json.loads(path.read_text());assert s['complete']
        metrics=s['new_image_evaluations'];d=Path(item['directory'])
        raw.append({**item,'completion_psnr_db':metrics['test_completion']['primary_metrics']['psnr_db'],
            'completion_edge_mse':metrics['test_completion']['primary_metrics']['gradient_mse'],
            'fully_observed_psnr_db':metrics['test_fully_observed']['primary_metrics']['psnr_db'],
            'training_probe_psnr_db':json.loads((d/'final/train_probe_metrics.json').read_text())['psnr_db'],
            'training_seconds':s['training']['elapsed_seconds'],
            'code_entries':64 if item['architecture']=='vanilla_qvf' else 336 if item['panel']=='cifar32' else 21504,
            'resources':s['training']['resources']})
    for panel in m['panels']:
        for name in m['methods']:
            group=[r for r in raw if r['panel']==panel and r['architecture']==name]
            row={'panel':panel,'architecture':name,'label':LABELS[name],
                 'completed_seeds':[r['seed'] for r in group],'seed_count':len(group),
                 'complete':len(group)==len(m['seeds'])}
            for key in ('completion_psnr_db','completion_edge_mse','fully_observed_psnr_db','training_probe_psnr_db'):
                values=[r[key] for r in group]
                row[key+'_values']=values
                row[key+'_mean']=float(np.mean(values)) if values else None
                row[key+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            rows.append(row)
    full=all(r['complete'] for r in rows) and not partial and not m['discarded_synthetic_smoke']
    differences=[]
    for panel in m['panels']:
        reference={r['seed']:r for r in raw if r['panel']==panel and r['architecture']=='vanilla_qvf'}
        for name in m['methods']:
            if name=='vanilla_qvf':continue
            group={r['seed']:r for r in raw if r['panel']==panel and r['architecture']==name}
            common=sorted(reference.keys()&group.keys())
            if not common:continue
            differences.append({'panel':panel,'architecture':name,'paired_seeds':common,
                'psnr_differences':[group[s]['completion_psnr_db']-reference[s]['completion_psnr_db'] for s in common],
                'edge_mse_differences':[group[s]['completion_edge_mse']-reference[s]['completion_edge_mse'] for s in common]})
    write_json(root/'comparison.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),
        'rows':rows,'per_seed':raw,'paired_vs_vanilla':differences,'complete':full,
        'audits_complete':not partial,'smoke':m['discarded_synthetic_smoke']})
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if not k.endswith('_values')])
        writer.writeheader();writer.writerows({k:v for k,v in r.items() if k in writer.fieldnames} for r in rows)
    for panel in m['panels']:
        panel_rows=[r for r in rows if r['panel']==panel]
        fig,axes=plt.subplots(1,2,figsize=(14,8),layout='constrained')
        for ax,key,title in zip(axes,('completion_psnr_db','completion_edge_mse'),
                               ('Completion PSNR · higher is better','Completion edge MSE · lower is better')):
            numeric=[v for r in panel_rows for v in r[key+'_values']]
            lo=min(numeric) if numeric else 0.;hi=max(numeric) if numeric else 1.
            span=max(hi-lo,abs(hi)*.02,1e-6);ax.set_xlim(lo-.12*span,hi+.38*span)
            for i,r in enumerate(panel_rows):
                color='#b65c19' if r['architecture']=='vanilla_qvf' else '#007e9a'
                if r[key+'_values']:
                    ax.scatter(r[key+'_values'],[i]*r['seed_count'],s=20,facecolors='white',edgecolors=color,zorder=3)
                    if r['complete']:ax.scatter([r[key+'_mean']],[i],marker='D',color=color,s=35,zorder=4)
                    text=f'{r[key+"_mean"]:.3f}' if 'psnr' in key else f'{r[key+"_mean"]:.6f}'
                    if not r['complete']:text+=f' ({r["seed_count"]}/3)'
                else:text='pending'
                ax.text(hi+.34*span,i,text,ha='right',va='center',fontsize=8)
            ax.set_yticks(range(len(panel_rows)),[r['label'] for r in panel_rows],fontsize=8)
            ax.set_ylim(len(panel_rows)-.5,-.7);ax.grid(axis='x',alpha=.2);ax.set_title(title,fontsize=10)
        dataset='CIFAR-10 dogs · 32×32' if panel=='cifar32' else 'Stanford Dogs · 256×256'
        state='COMPLETE' if full else 'PROVISIONAL · measurements or audits pending'
        fig.suptitle(f'{dataset} | 4,744 training images | 50,000 updates | 3 seeds\n{state}',fontsize=12)
        fig.savefig(root/(panel+'_comparison.png'),dpi=160);fig.savefig(root/(panel+'_comparison.pdf'));plt.close(fig)
    lines=['# Z8: complete QVF-method comparison','',
        f'Status: {"complete and verified" if full else "in progress; no complete comparison yet"}.',
        'Both plots include the same14 methods. Pending entries are measurements not yet completed; no historical scores substitute for missing runs.','',
        'Within each dataset, all methods use the same images,50,000 updates,paired seeds,queries,RGB-plus-edge objective and1,000-step frozen-decoder evaluation. Vanilla QVF keeps its original5q circuit,amplitude map,RGB readout and64-value code. The training recipe is matched to Z,not a replay of the paper training schedule.','',
        'Spatial codes contain336 values at32px and21,504 at256px. This is a comparison of architecture packages,not an equal-storage or equal-compute ablation. The two datasets differ; do not interpret cross-panel score changes as a resolution-only effect.','']
    for panel in m['panels']:
        lines.extend([f'## {panel}','',f'![All methods]({panel}_comparison.png)','',
            '| Method | Seeds | Completion PSNR | Completion edge MSE |','|---|---:|---:|---:|'])
        for r in rows:
            if r['panel']!=panel:continue
            score='pending' if not r['seed_count'] else f'{r["completion_psnr_db_mean"]:.3f}'
            edge='pending' if not r['seed_count'] else f'{r["completion_edge_mse_mean"]:.7f}'
            lines.append(f'| {r["label"]} | {r["seed_count"]}/3 | {score} | {edge} |')
        lines.append('')
    lines+=['Source data,per-seed values and paired differences versus vanilla QVF are in`comparison.json`. Final completion requires independent checkpoint/metric checks and native quantum/measurement audits. No hardware or quantum-advantage claim.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args();report(args.root,args.partial)
