"""Matched Z7/Z6 comparisons with per-seed results and static figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.qvf_project_a.train import write_json
from .hires_report import LABELS as OLD_LABELS

LABELS={**OLD_LABELS,'coherent_routing':'Coherent routing, 9q','real_routing':'Coherent routing without phase gates',
    'local_capacity9':'Local capacity control, 9q','classical_attention16':'Classical attention over three outputs'}


def report(root):
    root=Path(root).resolve();p=json.loads((root/'protocol.json').read_text())
    for filename in ('verification.json','quantum_audit.json','extended_development.json'):
        assert json.loads((root/filename).read_text())['complete']
    parent=Path(p['parent_benchmark']);names=p['architectures']+p['parent_reference_architectures'];rows=[];raw={}
    for name in names:
        source=root if name in p['architectures'] else parent;raw[name]={}
        for seed in p['seeds']:
            directory=source/f'{name}_seed{seed}';summary=json.loads((directory/'summary.json').read_text())
            fit=json.loads((directory/'final/train_probe_metrics.json').read_text());new=summary['new_image_evaluations']
            raw[name][seed]={'training_probe_psnr_db':fit['psnr_db'],'training_probe_gradient_mse':fit['gradient_mse'],
                **{split+'_'+metric:new[split]['primary_metrics'][metric] for split in ('development_completion','test_fully_observed','test_completion') for metric in ('psnr_db','gradient_mse')},
                'training_seconds':summary['training']['elapsed_seconds']}
        row={'architecture':name,'source':'Z7 fresh training' if source==root else 'Z6 completed benchmark'}
        for key in raw[name][p['seeds'][0]]:
            values=[raw[name][s][key] for s in p['seeds']]
            row[key+'_mean']=float(np.mean(values));row[key+'_sample_sd']=float(np.std(values,ddof=1))
        rows.append(row)
    comparisons=[]
    for variant,controls in p['comparisons'].items():
        for control in controls:
            for metric in ('training_probe_psnr_db','test_fully_observed_psnr_db','test_completion_psnr_db','test_completion_gradient_mse'):
                delta=[raw[variant][s][metric]-raw[control][s][metric] for s in p['seeds']]
                comparisons.append({'variant':variant,'control':control,'metric':metric,'paired_differences':delta,
                    'mean_difference':float(np.mean(delta)),'wins_of_three':sum(d>0 if 'psnr' in metric else d<0 for d in delta)})
    write_json(root/'comparison.json',{'rows':rows,'per_seed':raw,'paired_comparisons':comparisons,'complete':True})
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selected=p['architectures']+['separate_scales','three_mixed','spatial_reference','attention_chain5']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,key,title in zip(axes,('test_completion_psnr_db','test_completion_gradient_mse'),('Held-out completion PSNR (higher is better)','Held-out completion edge MSE (lower is better)')):
        for i,name in enumerate(selected):
            values=[raw[name][s][key] for s in p['seeds']]
            color='#157aa0' if name in p['architectures'] else '#707a8a'
            ax.scatter(values,[i]*len(values),s=15,color=color,alpha=.55)
            ax.scatter([np.mean(values)],[i],s=40,color=color,marker='D')
        ax.set_yticks(range(len(selected)),[LABELS[x] for x in selected]);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=170);fig.savefig(root/'comparison.pdf');plt.close(fig)
    extended=json.loads((root/'extended_development.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for name in p['extended_development_models']:
        curves=[x for x in extended['runs'] if x['architecture']==name]
        values=np.array([[r['metrics']['psnr_db'] for r in x['records']] for x in curves])
        ax.errorbar(p['extended_development_code_steps'],values.mean(0),yerr=values.std(0,ddof=1),marker='o',label=LABELS[name],capsize=4)
    ax.set_xlabel('Updates of new-image codes; decoder frozen');ax.set_ylabel('Development completion PSNR (dB)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'development_code_fitting.png',dpi=170);fig.savefig(root/'development_code_fitting.pdf');plt.close(fig)
    original=np.load(Path(p['data_root'])/'test.npy',mmap_mode='r')
    gallery=p['architectures']+['separate_scales','attention_chain5']
    for index in (0,1):
        fig,axes=plt.subplots(2,len(gallery)+1,figsize=(18,5.5),layout='constrained')
        images=[original[index].astype(np.float32)/255.]
        for name in gallery:
            source=root if name in p['architectures'] else parent
            images.append(np.load(source/f'{name}_seed0/final/test_fully_observed_reconstructions.npy',mmap_mode='r')[index])
        for col,(image,label) in enumerate(zip(images,['Original']+[LABELS[x] for x in gallery])):
            lo=image.shape[0]*3//8;hi=image.shape[0]*5//8
            axes[0,col].imshow(np.clip(image,0,1));axes[0,col].set_title(label,fontsize=8)
            axes[1,col].imshow(np.clip(image[lo:hi,lo:hi],0,1));axes[1,col].set_title('Fixed central detail crop',fontsize=8)
            for row in (0,1):axes[row,col].axis('off')
        fig.suptitle(f'Reserved benchmark image {index}, seed0; frozen decoder, 1,000 fully-observed code updates')
        fig.savefig(root/f'native_image_{index}.png',dpi=160);plt.close(fig)
    lines=['# Z7: coherent routing benchmark against Z6','',
        'Four new conditions were trained from scratch with three paired seeds. Data, updates, queried coordinates, losses, code budgets and primary evaluation are matched to the completed Z6 study. Z6 scores are reused from verified artifacts; those models were not retrained.','',
        '| Model | Training probe PSNR | New-image full-fit PSNR | Completion PSNR | Completion edge MSE |',
        '|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {LABELS[row["architecture"]]} | {row["training_probe_psnr_db_mean"]:.3f} | {row["test_fully_observed_psnr_db_mean"]:.3f} | {row["test_completion_psnr_db_mean"]:.3f} | {row["test_completion_gradient_mse_mean"]:.7f} |')
    lines += ['', 'Values are three-seed means. Paired differences and sample SDs are saved in `comparison.json`; three seeds do not establish broad statistical significance.','', '## Paired completion comparisons','']
    for x in comparisons:
        if x['metric']=='test_completion_psnr_db':lines.append(f'- Coherent routing versus {LABELS[x["control"]]}: {x["mean_difference"]:+.3f}dB; higher PSNR in {x["wins_of_three"]}/3 seeds.')
    lines += ['', '## Interpretation and limits','']+[f'- {x}' for x in p['limits']]
    lines += ['', '- Longer code-fitting curves use development images only. The primary test comparison retains 1,000 updates. A recovery after longer fitting supports an optimization-budget explanation; it is not proof that a decoder has unlimited capacity.',
        '- Local-capacity control has the same nine-qubit register and 16 extra trainable quantum parameters but no non-additive communication between scales. Its primitive gate count differs. Classical attention has 16 extra stored classical parameters and three independent seven-qubit circuits.',
        '- Circuit coherence diagnostics establish sensitivity to coherence for the trained model, not an advantage over classical computation or over a separately retrained dephased model.',
        '- Quantum circuits, gradients, fresh initializations, source/checkpoint hashes, training/query streams and independent reconstruction metrics have been checked. Noise is synthetic and excludes physical routing.',
        '', '![Comparison](comparison.png)', '', '![Development code fitting](development_code_fitting.png)', '', '![Native reconstructions](native_image_0.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
