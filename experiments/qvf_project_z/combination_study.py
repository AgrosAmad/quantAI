"""Z9 matched extension: per-scale attention with three quantum branches.

The training/load bodies are copied unchanged from frozen Z8; only the model
registry, checkpoint kind and protocol preparation are new.
"""
import argparse
from copy import deepcopy
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
from .combination_models import NAMES,build_model,resources

HERE=Path(__file__).resolve().parent
PARENT=HERE/'results/20261112_z8_complete_methods'
KIND='qvf_project_z_combination_v1'


def create_protocol(root,smoke=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    parent=json.loads((PARENT/'manifest.json').read_text())
    parents=[PARENT/panel for panel in parent['panels']]+[Path(p) for p in parent['reused_roots']]
    for folder in parents:assert_sources(folder)
    entries=[{**item,'reused':True} for item in parent['coverage']];jobs=[]
    methods=parent['methods']+list(NAMES)
    for panel in parent['panels']:
        d=root/panel;d.mkdir()
        p=deepcopy(json.loads((PARENT/panel/'protocol.json').read_text()))
        p.update({'kind':KIND,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'architectures':list(NAMES),'all_architectures':methods,'parent_study':str(root),
            'parent_protocol_path':str(PARENT/panel/'protocol.json'),
            'parent_protocol_sha256':sha256(PARENT/panel/'protocol.json'),
            'within_scale_attention':{'type':'classical query/key softmax','window_radius':2,
                'max_tokens_per_slot':25,'slots_per_scale':5,'key_dimension':16,
                'distance_logit_penalty':2.,'out_of_bounds_tokens':'masked, no duplicate border tokens'},
            'comparisons':{'scale_attention_mean':['separate_scales','attention_chain5'],
                'scale_attention_fusion':['scale_attention_mean','classical_attention16'],
                'scale_attention_coherent':['scale_attention_mean','scale_attention_fusion','coherent_routing']}})
        p['limits'] += [
            'New arms use the same seven-qubit spatial circuits, encoding stages and measurement map as their corresponding Z6/Z7 references.',
            'Attention is classical, within a fixed 5x5-token window around each of five spatial slots at each scale. It is not global attention.',
            'The prior five-qubit attention reference uses dense attention, amplitude encoding and a different circuit. That contrast does not isolate circuit count.',
            'Mean and output-attention arms execute three independent seven-qubit circuits; coherent fusion uses one nine-qubit multiplexed register, not a 21-qubit state.',
            'Output attention is a 16-parameter classical softmax over measured outputs. Coherent fusion uses the existing quantum routing operations.',
            'Attention adds classical parameters and work; circuit and shot budgets differ between fusion strategies.',
            'Z8 and earlier benchmark results were inspected before this extension; this is not an untouched confirmatory test set.']
        if smoke:
            data=d/'synthetic';data.mkdir();rng=np.random.default_rng(182)
            for split,count in [('train',8),('development',4),('test',4)]:
                np.save(data/(split+'.npy'),rng.integers(0,256,(count,8,8,3),dtype=np.uint8))
            write_json(data/'manifest.json',{'discarded_synthetic_smoke':True})
            observed=list(range(0,64,4));hidden=np.setdiff1d(np.arange(64),observed)
            p.update({'discarded_synthetic_smoke':True,'device':'cpu','data_root':str(data),
                'data_manifest_sha256':sha256(data/'manifest.json'),
                'data_arrays_sha256':{s:sha256(data/(s+'.npy')) for s in ('train','development','test')},
                'n_train':8,'steps':6,'batch_size':2,'pairs_per_orientation':2,
                'evaluation_steps':[3,6],'checkpoint_steps':[0,3,6],
                'training_probe_indices':[0,1,2,3],'new_image_indices':[0,1,2,3],
                'new_code_steps':3,'new_code_queries_per_image':8,'render_image_batch':2,'render_query_batch':16,
                'model_config':{**p['model_config'],'image_size':8,'grid_sizes':[2,4,8],'frequencies':3},
                'completion_observed_indices':observed,'quantum_audit_query_indices':hidden[:4].tolist(),
                'quantum_audit_image_indices':[0,1],'quantum_noise_image_indices':[0],
                'noise_query_indices':hidden[:1].tolist()})
        write_json(d/'protocol.json',p);snapshot_sources(d);panel_runs=[]
        for seed in p['seeds']:
            for name in NAMES:
                directory=d/f'{name}_seed{seed}';directory.mkdir()
                item={'panel':panel,'architecture':name,'seed':seed,'directory':str(directory),
                      'protocol_root':str(d),'reused':False}
                write_json(directory/'run.json',item);entries.append(item);jobs.append(item);panel_runs.append(item)
        write_json(d/'runs.json',panel_runs)
    manifest={'kind':KIND,'methods':methods,'new_methods':list(NAMES),'seeds':parent['seeds'],
        'panels':parent['panels'],'wait_for_study':str(PARENT),'reused_roots':[str(p) for p in parents],
        'parent_manifest_sha256':sha256(PARENT/'manifest.json'),
        'reused_protocol_hashes':{str(p):sha256(p/'protocol.json') for p in parents},
        'coverage':entries,'fresh_runs':jobs,'discarded_synthetic_smoke':smoke,
        'authorization':'User requested adding the full per-scale attention and multi-circuit combination to the existing comparison. Existing 50000-update, three-seed, two-resolution protocol retained.'}
    write_json(root/'manifest.json',manifest);write_json(root/'runs.json',jobs)
    return manifest


def load_model(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['kind']!=KIND:raise ValueError('Wrong Z8 checkpoint')
    cfg=dict(saved['model_config']);cfg.pop('architecture',None)
    model=build_model(saved['variant'],saved['seed'],**cfg).to(device)
    model.load_state_dict(saved['decoder_state'],strict=True)
    return model.eval(),saved['codes'].to(device),saved


def train_one(root,name,seed):
    root=Path(root).resolve();assert_sources(root)
    p=json.loads((root/'protocol.json').read_text());directory=root/f'{name}_seed{seed}'
    if name not in p['architectures'] or seed not in p['seeds']:raise ValueError('Undeclared run')
    lock=directory/'started.json'
    if lock.exists():raise FileExistsError(f'Fresh-only runner refuses to overwrite {directory}')
    write_json(lock,{'pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    device=configure_runtime(p.get('device','cuda:0'),seed)
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
        payload={'kind':KIND,'model_config':dict(p['model_config']),'variant':name,'seed':seed,'step':step,'epoch':epoch,
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
        'training_quantum_queries':step*p['batch_size']*4*p['pairs_per_orientation']*resources(model,name)['logical_circuits_per_coordinate'],
        'resources':resources(model,name),'elapsed_seconds':training_seconds,'checkpoint_sha256':sha256(checkpoint_path)}
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
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--create',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--architecture',choices=NAMES);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.create:create_protocol(args.root,args.smoke)
    else:train_one(args.root,args.architecture,args.seed)


if __name__=='__main__':main()
