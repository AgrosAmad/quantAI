"""Static Z6 comparison tables and native-resolution reconstruction figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.qvf_project_a.train import write_json
from .hires_models import PRIMARY

LABELS={'spatial_reference':'Shared spatial reference','joint_statistics':'Joint measurement statistics',
    'sequential_scales':'Successive scales, one circuit','complex_phases':'Complex phase gates',
    'separate_scales':'Three circuits, one per scale','marginal_products':'Products of marginal means',
    'extra_real_gates':'Extra real gates','three_mixed':'Three circuits, all scales each',
    'attention_chain5':'Five-qubit attention context baseline'}


def report(root):
    root=Path(root).resolve();p=json.loads((root/'protocol.json').read_text())
    verification=json.loads((root/'verification.json').read_text());quantum=json.loads((root/'quantum_audit.json').read_text())
    assert verification['complete'] and quantum['complete']
    rows=[];raw={}
    for name in p['architectures']:
        raw[name]={}
        for seed in p['seeds']:
            directory=root/f'{name}_seed{seed}';summary=json.loads((directory/'summary.json').read_text())
            fit=json.loads((directory/'final/train_probe_metrics.json').read_text())
            new=summary['new_image_evaluations'];train=summary['training']
            raw[name][seed]={'train_probe_psnr_db':fit['psnr_db'],'train_probe_gradient_mse':fit['gradient_mse'],
                **{split+'_'+metric:new[split]['primary_metrics'][metric] for split in ('development_completion','test_fully_observed','test_completion') for metric in ('psnr_db','gradient_mse')},
                'training_seconds':train['elapsed_seconds'],'total_seconds':summary['elapsed_seconds']}
        row={'architecture':name}
        for key in raw[name][p['seeds'][0]]:
            values=[raw[name][s][key] for s in p['seeds']]
            row[key+'_mean']=float(np.mean(values));row[key+'_sample_sd']=float(np.std(values,ddof=1))
        rows.append(row)
    with (root/'comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[]
    for name,controls in p['comparisons'].items():
        if name=='context_baseline':continue
        for control in controls:
            for metric in ('train_probe_psnr_db','test_fully_observed_psnr_db','test_completion_psnr_db','test_completion_gradient_mse'):
                delta=[raw[name][s][metric]-raw[control][s][metric] for s in p['seeds']]
                higher_better='psnr' in metric
                paired.append({'variant':name,'control':control,'metric':metric,'difference_by_seed':delta,
                    'mean_difference':float(np.mean(delta)),
                    'wins_of_three':sum(x>0 if higher_better else x<0 for x in delta)})
    write_json(root/'comparison.json',{'means_and_sd':rows,'per_seed':raw,'paired_comparisons':paired,
        'all_seeds_complete':True,'source_verification_passed':True,'quantum_audit_passed':True})
    selected=['spatial_reference',*PRIMARY]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric,title in zip(axes,('test_completion_psnr_db','test_completion_gradient_mse'),('Unseen-image completion: PSNR (higher is better)','Unseen-image completion: edge MSE (lower is better)')):
        values=np.array([[raw[n][s][metric] for s in p['seeds']] for n in selected])
        ax.barh(np.arange(len(selected)),values.mean(1),color=['#6d7386','#3388b9','#22a884','#cf8d36','#9c6bc0'],alpha=.75)
        for i,v in enumerate(values):ax.scatter(v,np.full(3,i),c='#172b4d',s=15,zorder=3)
        ax.set_yticks(np.arange(len(selected)),[LABELS[n] for n in selected]);ax.invert_yaxis();ax.set_title(title,fontsize=10)
        ax.grid(axis='x',alpha=.2)
    fig.savefig(root/'primary_comparison.png',dpi=170);fig.savefig(root/'primary_comparison.pdf');plt.close(fig)
    # Images are selected before examining quality: first two reserved test rows.
    actual=np.load(Path(p['data_root'])/'test.npy',mmap_mode='r')
    gallery_names=['spatial_reference',*PRIMARY,'attention_chain5']
    for index in (0,1):
        fig,axes=plt.subplots(2,len(gallery_names)+1,figsize=(18,5.5),layout='constrained')
        images=[actual[index].astype(np.float32)/255.]
        images += [np.load(root/f'{n}_seed0/final/test_fully_observed_reconstructions.npy',mmap_mode='r')[index] for n in gallery_names]
        for col,(img,label) in enumerate(zip(images,['Original']+[LABELS[n] for n in gallery_names])):
            axes[0,col].imshow(np.clip(img,0,1));axes[0,col].set_title(label,fontsize=8)
            lo=img.shape[0]*3//8;hi=img.shape[0]*5//8
            axes[1,col].imshow(np.clip(img[lo:hi,lo:hi],0,1));axes[1,col].set_title(f'Fixed central {hi-lo}×{hi-lo} crop',fontsize=8)
            for row in (0,1):axes[row,col].axis('off')
        fig.suptitle(f'256×256 held-out image {index}; frozen decoder, 1,000 new-code updates; seed 0',fontsize=12)
        fig.savefig(root/f'test_image_{index}_native.png',dpi=160);plt.close(fig)
    lines=['# Z6: native 256×256 quantum-INR study','',
        'Four quantum modifications were trained separately, with matched controls, on 4,744 Stanford Dogs images. All models use 50,000 updates and three paired seeds. Targets are downsampled original photographs, never enlarged CIFAR images.','',
        '| Architecture | Training probe PSNR | New-image full-fit PSNR | Completion PSNR | Completion edge MSE |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {LABELS[r["architecture"]]} | {r["train_probe_psnr_db_mean"]:.3f} | {r["test_fully_observed_psnr_db_mean"]:.3f} | {r["test_completion_psnr_db_mean"]:.3f} | {r["test_completion_gradient_mse_mean"]:.7f} |')
    lines += ['', 'Values are means over three seeds. Full per-seed values, sample standard deviations and paired differences are in `comparison.json`; three seeds do not establish broad statistical significance.', '',
        '## Controlled comparisons','']
    for x in paired:
        if x['metric']=='test_completion_psnr_db':lines.append(f'- {LABELS[x["variant"]]} versus {LABELS[x["control"]]}: {x["mean_difference"]:+.3f} dB; higher PSNR in {x["wins_of_three"]}/3 seeds.')
    lines += ['', '## Scope and limitations','']+[f'- {v}' for v in p['limits']]
    lines += ['', '- Completion fitting observes a fixed 25% pixel mask. Scores use only withheld pixels and edges whose two endpoints are withheld. Full-fit codes observe all pixels, so that test measures inversion capacity, not prediction of unobserved data.',
        '- Readout, state and gradient checks, finite-shot estimates and synthetic density-matrix noise are recorded in `quantum_audit.json`. Noise uses seed 0, four images and 256 sampled coordinates per image; it is not a full-image noise benchmark.',
        '- Same shots per circuit and equal total shots per coordinate are both reported. Three independent seven-qubit circuits can reuse seven physical qubits sequentially; simultaneous execution needs 21. The three registers are not entangled together.',
        '- Code fitting and training remain classical optimization with differentiable statevector simulation. Small registers do not eliminate the cost of one or three circuit evaluations per queried coordinate.',
        '', '![Primary comparison](primary_comparison.png)', '', '![Native reconstruction and fixed crops](test_image_0_native.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
