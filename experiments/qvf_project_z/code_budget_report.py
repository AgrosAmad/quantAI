"""Z10: all previous methods plus the original QVF with256 global code entries."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from experiments.qvf_project_a.train import write_json,sha256
from .combination_report import LABELS as OLD_LABELS
from .code_budget_models import LABELS as NEW_LABELS
from .code_budget_plot import render as plot

LABELS={**OLD_LABELS,**NEW_LABELS}


def report(root,partial=False):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    identities=[(x['panel'],x['architecture'],x['seed']) for x in m['coverage']]
    assert len(identities)==len(set(identities))
    if not partial:
        for panel in m['panels']:
            for file in ('verification.json','quantum_audit.json'):
                assert json.loads((root/panel/file).read_text())['complete']
        if not m['discarded_synthetic_smoke']:
            assert json.loads((Path(m['wait_for_study'])/'queue_status.json').read_text())['phase']=='complete'
            for folder,digest in m['reused_protocol_hashes'].items():
                source=Path(folder);assert sha256(source/'protocol.json')==digest
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
            'completion_mse':metrics['test_completion']['primary_metrics']['mse'],
            'fully_observed_psnr_db':metrics['test_fully_observed']['primary_metrics']['psnr_db'],
            'training_probe_psnr_db':json.loads((d/'final/train_probe_metrics.json').read_text())['psnr_db'],
            'training_seconds':s['training']['elapsed_seconds'],
            'code_entries':s['training']['resources']['image_code_entries'],
            'resources':s['training']['resources']})
    for panel in m['panels']:
        for name in m['methods']:
            group=[r for r in raw if r['panel']==panel and r['architecture']==name]
            row={'panel':panel,'architecture':name,'label':LABELS[name],
                 'completed_seeds':[r['seed'] for r in group],'seed_count':len(group),
                 'complete':len(group)==len(m['seeds'])}
            for key in ('completion_psnr_db','completion_edge_mse','completion_mse','fully_observed_psnr_db','training_probe_psnr_db'):
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
    contrasts=[]
    planned={
        'vanilla_qvf256':['vanilla_qvf','spatial_reference','attention_chain5'],
        'scale_attention_mean':['separate_scales','attention_chain5'],
        'scale_attention_fusion':['scale_attention_mean','classical_attention16'],
        'scale_attention_coherent':['scale_attention_mean','scale_attention_fusion','coherent_routing']}
    for panel in m['panels']:
        for name,references in planned.items():
            group={r['seed']:r for r in raw if r['panel']==panel and r['architecture']==name}
            for reference in references:
                other={r['seed']:r for r in raw if r['panel']==panel and r['architecture']==reference}
                common=sorted(group.keys()&other.keys())
                if common:
                    contrasts.append({'panel':panel,'architecture':name,'reference':reference,'paired_seeds':common,
                        'psnr_differences':[group[s]['completion_psnr_db']-other[s]['completion_psnr_db'] for s in common],
                        'edge_mse_differences':[group[s]['completion_edge_mse']-other[s]['completion_edge_mse'] for s in common]})
    write_json(root/'comparison.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),
        'rows':rows,'per_seed':raw,'paired_vs_vanilla':differences,'paired_planned_contrasts':contrasts,'complete':full,
        'audits_complete':not partial,'smoke':m['discarded_synthetic_smoke']})
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if not k.endswith('_values')])
        writer.writeheader();writer.writerows({k:v for k,v in r.items() if k in writer.fieldnames} for r in rows)
    plot(root)
    lines=['# Z10: larger-code vanilla QVF control','',
        f'Status: {"complete and verified" if full else "in progress; no complete comparison yet"}.',
        'Both plots include all18 methods. The64-value and256-value vanilla QVF controls are red; the latter uses a triangle for its complete three-seed mean. Missing measurements stay pending.','',
        'The new arm uses the original QVFImage with256 learned global code entries instead of64. Its five qubits, five repetitions, all-to-all controlled rotations, amplitude output size32, three128-wide hidden layers and fixed RGB measurement map are unchanged. The widened input adds24,576 classical first-layer weights. All runs start fresh.','',
        'Within each dataset, images,50,000 training updates, three seeds, coordinate streams, Adam settings, RGB-plus-edge objective and1,000-step frozen-decoder code fits are unchanged. The original initializer is used at each width, so paired seed labels do not imply identical shared initial tensors. Code initialization standard deviation and the L2-norm penalty coefficient are unchanged; expected norms depend on dimension.','',
        'This tests a larger global code, including its larger MLP input. It does not equalize representation storage or classical compute: spatial methods still use336 entries at32px and21,504 at256px. The two panels use different datasets. Prior test results informed this extension, so it is not a new confirmatory holdout.','',
        'The18 Z9 per-scale-attention runs continue unchanged. Z10 waits for that complete pipeline before training its six new runs on GPU0. The combined report reuses earlier runs and marks unmeasured rows pending.','']
    if m['discarded_synthetic_smoke']:
        lines=['# DISCARDED SYNTHETIC SMOKE — not quality measurements','',
               'Each disposable run used8 synthetic images at8px,6 training updates and3 code-fitting updates. Production protocol descriptions below are not the smoke budget.','']+lines
    for panel in m['panels']:
        lines.extend([f'## {panel}','',f'![All methods]({panel}_comparison.png)','',
            '| Method | Seeds | Completion PSNR | Completion edge MSE |','|---|---:|---:|---:|'])
        for r in rows:
            if r['panel']!=panel:continue
            score='pending' if not r['seed_count'] else f'{r["completion_psnr_db_mean"]:.3f}'
            edge='pending' if not r['seed_count'] else f'{r["completion_edge_mse_mean"]:.7f}'
            lines.append(f'| {r["label"]} | {r["seed_count"]}/3 | {score} | {edge} |')
        lines.append('')
    lines+=['Source data,per-seed values, arithmetic mean pixel MSE, paired differences versus QVF64 and planned code-size contrasts are in`comparison.json`. Final completion requires independent checkpoint/metric checks and native quantum/measurement audits. No hardware or quantum-advantage claim.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args();report(args.root,args.partial)
