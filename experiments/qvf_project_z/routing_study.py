"""Z7 matched 256px benchmark of coherent scale routing and controls."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,save_checkpoint,sha256,write_json,REPO
from experiments.qvf_project_a.edge_qvf import neighbour_pairs,rgb_importance_weights,edge_objective
from experiments.qvf_project_a.models import pixel_coordinates
from .hires_study import targets,metrics,render,fit_new_codes,state_hash,atomic_json,assert_sources
from .scale_study import snapshot_sources
from .routing_models import NAMES,build_model,resource_summary

HERE=Path(__file__).resolve().parent
PARENT=HERE/'results/20261110_z6_hires'
KIND='qvf_project_z_routing_v1'
MATCH_KEYS=('data_root','data_manifest_sha256','data_arrays_sha256','seeds','model_config','n_train','steps',
    'batch_size','pairs_per_orientation','model_lr','code_lr','code_init_std','code_norm_weight',
    'edge_weight','evaluation_steps','checkpoint_steps','training_probe_indices',
    'render_image_batch','render_query_batch','new_image_indices','new_code_steps','new_code_queries_per_image',
    'completion_observed_indices','quantum_audit_query_indices','quantum_audit_image_indices',
    'quantum_noise_image_indices','shots','equal_total_shot_budgets','measurement_seeds','noise_levels')


def create_protocol(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    parent=json.loads((PARENT/'protocol.json').read_text());p={k:parent[k] for k in MATCH_KEYS}
    assert_sources(PARENT)
    assert json.loads((PARENT/'queue_status.json').read_text())['phase']=='complete'
    for file in ('verification.json','quantum_audit.json'):
        verified=json.loads((PARENT/file).read_text())
        assert verified['complete'] and len(verified['runs'])==27 and all(x['passed'] for x in verified['runs'])
    p.update({'kind':KIND,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'parent_benchmark':str(PARENT),'parent_protocol_sha256':sha256(PARENT/'protocol.json'),
        'architectures':list(NAMES),'primary_variant':'coherent_routing',
        'comparisons':{'coherent_routing':['real_routing','local_capacity9','classical_attention16',
            'separate_scales','three_mixed','spatial_reference','attention_chain5','joint_statistics']},
        'parent_reference_architectures':parent['architectures'],
        'noise_query_indices':parent['quantum_audit_query_indices'][::8],
        'noise_parent_rechecks':['separate_scales','attention_chain5'],
        'extended_development_code_steps':[1000,5000,10000],
        'extended_development_models':['coherent_routing','separate_scales','attention_chain5'],
        'training_device':'physical GPU0 only, two concurrent jobs',
        'limits':parent['limits']+[
            'Z6 test scores were inspected before Z7 design. This is reuse of a benchmark, not a previously unseen confirmatory test set.',
            'All four variants are freshly trained; no Z6 weights/codes are used for initialization.',
            'Stored total parameters match except real_routing freezes eight phase entries; independence of all parameter entries is not claimed.',
            'The classical 16-parameter softmax gate is output fusion, not feedback or attention over grid cells.',
            'Nine-qubit coherent variants are multiplexed branches, not three independent seven-qubit registers.',
            'Synthetic gate-noise scope is four test images x 32 common withheld coordinates, seed0 only; Z6 references are rechecked on that same subset.',
            'Finite-shot classical softmax fusion is nonlinear; its expected loss is estimated by measurement replicates, not the linear readout formula.',
            'Coherence-removal checks use a tiny trained-checkpoint diagnostic; the dephased model is not separately retrained.',
            'No hardware execution, quantum advantage, or research-novelty claim.']})
    # Remove obsolete Z6-specific readout statements from the inherited limits.
    p['limits']=[x for x in p['limits'] if not x.startswith(('Three independent heads use','Marginal-product and joint readouts','Extra-real gate control'))]
    write_json(root/'protocol.json',p);snapshot_sources(root)
    references={}
    for filename in ('protocol.json','comparison.json','verification.json','quantum_audit.json','source_manifest.json'):
        references[filename]=sha256(PARENT/filename)
    for run in json.loads((PARENT/'runs.json').read_text()):
        directory=Path(run['directory'])
        for rel in ('summary.json','training_summary.json','final/train_probe_metrics.json',
            'final/development_completion_metrics.json','final/test_fully_observed_metrics.json','final/test_completion_metrics.json'):
            references[str((directory/rel).relative_to(PARENT))]=sha256(directory/rel)
    write_json(root/'parent_reference_hashes.json',references)
    runs=[]
    for seed in p['seeds']:
        for name in NAMES:
            d=root/f'{name}_seed{seed}';d.mkdir();item={'architecture':name,'seed':seed,'directory':str(d)}
            write_json(d/'run.json',item);runs.append(item)
    write_json(root/'runs.json',runs)
    return p


def load_model(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['kind']!=KIND:raise ValueError('Wrong Z7 checkpoint')
    cfg=dict(saved['model_config']);cfg.pop('architecture')
    model=build_model(saved['variant'],saved['seed'],**cfg).to(device)
    model.load_state_dict(saved['decoder_state'],strict=True)
    return model.eval(),saved['codes'].to(device),saved


# The matched training loop below is copied from the frozen Z6 harness; only
# model construction, variant checkpoint metadata and resource accounting differ.

def train_one(root,name,seed):
    root=Path(root).resolve();assert_sources(root)
    p=json.loads((root/'protocol.json').read_text());directory=root/f'{name}_seed{seed}'
    if name not in p['architectures'] or seed not in p['seeds']:raise ValueError('Undeclared run')
    lock=directory/'started.json'
    if lock.exists():raise FileExistsError(f'Fresh-only runner refuses to overwrite {directory}')
    write_json(lock,{'pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    device=configure_runtime('cuda:0',seed)
    data=Path(p['data_root']);images=np.load(data/'train.npy',mmap_mode='r')
    assert len(images)==p['n_train'] and sha256(data/'manifest.json')==p['data_manifest_sha256']
    cfg=dict(p['model_config']);cfg.pop('architecture')
    model=build_model(name,seed,**cfg).to(device)
    generator=torch.Generator(device=device).manual_seed(seed)
    codes=torch.nn.Parameter(torch.randn(p['n_train'],model.cfg.latent_dim,generator=generator,device=device)*p['code_init_std'])
    code_initial_hash=hashlib.sha256(codes.detach().cpu().numpy().tobytes()).hexdigest()
    opt_model=torch.optim.Adam(model.parameters(),lr=p['model_lr']);opt_codes=torch.optim.Adam([codes],lr=p['code_lr'])
    order_rng=torch.Generator().manual_seed(seed+11);query_rng=torch.Generator().manual_seed(seed+12)
    coords=pixel_coordinates(model.cfg.image_size,model.cfg.image_size,device=device)
    pair_tables=neighbour_pairs(model.cfg.image_size);weights=rgb_importance_weights(model.cfg.image_size).to(device)
    visits=torch.zeros(p['n_train'],dtype=torch.int64);stream=hashlib.sha256()
    probe=np.array(p['training_probe_indices']);probe_gpu=torch.tensor(probe,device=device)
    fit=directory/'fit';fit.mkdir();(fit/'checkpoints').mkdir();final=directory/'final';final.mkdir()
    start=time.perf_counter();step=0;epoch=0;last_terms=None
    def checkpoint(label):
        payload={'kind':KIND,'model_config':asdict(model.cfg),'variant':name,'seed':seed,'step':step,'epoch':epoch,
            'decoder_state':model.state_dict(),'codes':codes,'visits':visits,
            'optimizer_model':opt_model.state_dict(),'optimizer_codes':opt_codes.state_dict(),
            'rng':{'order':order_rng.get_state(),'query':query_rng.get_state()},
            'initial_codes_sha256':code_initial_hash,'stream_sha256':stream.hexdigest(),
            'protocol_sha256':sha256(root/'protocol.json')}
        save_checkpoint(fit/'checkpoints'/f'{label}.pt',payload)
    checkpoint('step0')
    with (fit/'history.jsonl').open('x') as history:
        while step<p['steps']:
            epoch+=1
            for rows in torch.randperm(p['n_train'],generator=order_rng).split(p['batch_size']):
                if len(rows)!=p['batch_size']:continue
                points=torch.cat([table[torch.randint(len(table),(p['pairs_per_orientation'],),generator=query_rng)] for table in pair_tables]).flatten()
                stream.update(rows.numpy().tobytes());stream.update(points.numpy().tobytes())
                selected=codes[rows.to(device)];gpu_points=points.to(device)
                model.train();model.clear_cache();opt_model.zero_grad(set_to_none=True);opt_codes.zero_grad(set_to_none=True)
                y=model(selected,coords[gpu_points])
                terms=edge_objective(y,targets(images,rows.numpy(),points.numpy(),device),weights[gpu_points],selected,p['edge_weight'],p['code_norm_weight'])
                if not torch.isfinite(terms[0]):raise FloatingPointError(f'{name} seed {seed} step {step+1}')
                terms[0].backward();opt_model.step();opt_codes.step();step+=1;visits[rows]+=1
                if step==1 or step%500==0:
                    status={'architecture':name,'seed':seed,'step':step,'epoch':epoch,'elapsed_seconds':time.perf_counter()-start,
                        'loss_components':[x.detach().item() for x in terms]}
                    atomic_json(directory/'progress.json',status)
                    print(f'{name} seed={seed} step={step}/{p["steps"]} loss={status["loss_components"][0]:.6f} elapsed={status["elapsed_seconds"]:.1f}s',flush=True)
                if step in p['evaluation_steps']:
                    prediction=render(model,codes[probe_gpu],p);scores=metrics(prediction,images[probe])
                    record={'step':step,'epoch':epoch,'training_probe':scores,'elapsed_seconds':time.perf_counter()-start}
                    history.write(json.dumps(record,allow_nan=False)+'\n');history.flush()
                    print(f'PROBE {name} seed={seed} step={step} PSNR={scores["psnr_db"]:.3f} edge={scores["gradient_mse"]:.7f}',flush=True)
                    if step==p['steps']:
                        np.save(final/'train_probe_reconstructions.npy',prediction);write_json(final/'train_probe_metrics.json',scores)
                if step in p['checkpoint_steps']:checkpoint(f'step{step}')
                if step==p['steps']:break
    training_seconds=time.perf_counter()-start
    checkpoint_path=fit/'checkpoints'/f'step{p["steps"]}.pt'
    training={'optimizer_steps':step,'epochs':epoch,'visits_per_image':visits.tolist(),
        'initial_codes_sha256':code_initial_hash,'stream_sha256':stream.hexdigest(),
        'training_coordinate_queries':step*p['batch_size']*4*p['pairs_per_orientation'],
        'training_quantum_queries':step*p['batch_size']*4*p['pairs_per_orientation']*resource_summary(model)['logical_circuits_per_coordinate'],
        'resources':resource_summary(model),'elapsed_seconds':training_seconds,'checkpoint_sha256':sha256(checkpoint_path)}
    write_json(directory/'training_summary.json',training)
    # Release full-table optimizer memory before new-image code optimization.
    del codes,opt_codes,opt_model,y,terms,selected
    torch.cuda.empty_cache()
    new={}
    for split,mode in [('development','completion'),('test','fully_observed'),('test','completion')]:
        selected_images=np.load(data/(split+'.npy'),mmap_mode='r')[p['new_image_indices']]
        new[split+'_'+mode]=fit_new_codes(model,selected_images,p,seed,split,mode,final,device)
    summary={'architecture':name,'seed':seed,'training':training,'new_image_evaluations':new,
        'elapsed_seconds':time.perf_counter()-start,'complete':True}
    write_json(directory/'summary.json',summary)
    atomic_json(directory/'progress.json',{'architecture':name,'seed':seed,'step':step,'phase':'complete','elapsed_seconds':time.perf_counter()-start})



def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--create',action='store_true');p.add_argument('--architecture',choices=NAMES);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.create:create_protocol(a.root)
    else:train_one(a.root,a.architecture,a.seed)

if __name__=='__main__':main()
