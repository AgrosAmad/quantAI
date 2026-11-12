"""Z6 native-resolution training and fixed-decoder new-image evaluation.

Every long run starts fresh. Dataset pixels stay uint8 on the CPU; only queried
targets are transferred. Training, evaluation and test streams are independent.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from experiments.qvf_project_a.edge_qvf import neighbour_pairs,rgb_importance_weights,edge_objective
from experiments.qvf_project_a.models import pixel_coordinates
from experiments.qvf_project_a.train import configure_runtime,save_checkpoint,sha256,write_json,REPO
from .hires_models import NAMES,PRIMARY,HiresConfig,HiresField,build_model,resources
from .scale_study import snapshot_sources

HERE=Path(__file__).resolve().parent
DATA=HERE/'data/stanford_dogs_maurice/prepared_256'
KIND='qvf_project_z_hires_v1'


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp.json');write_json(temporary,value);temporary.replace(path)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def create_protocol(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    assert (DATA/'manifest.json').exists()
    manifest=json.loads((DATA/'manifest.json').read_text())
    snapshot_sources(root)
    rng=np.random.default_rng(20261110)
    probe=np.sort(rng.choice(4744,256,replace=False)).tolist()
    observed=np.sort(rng.choice(256**2,256**2//4,replace=False)).tolist()
    hidden=np.setdiff1d(np.arange(256**2),observed)
    audit_points=np.sort(rng.choice(hidden,256,replace=False)).tolist()
    protocol={'kind':KIND,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'data_root':str(DATA),'data_manifest_sha256':sha256(DATA/'manifest.json'),
        'data_arrays_sha256':{split:sha256(DATA/(split+'.npy')) for split in ('train','development','test')},
        'architectures':list(NAMES),'primary_four':list(PRIMARY),'seeds':[0,1,2],
        'model_config':asdict(HiresConfig()),'n_train':4744,'steps':50000,
        'batch_size':16,'pairs_per_orientation':512,'model_lr':.001,'code_lr':.001,
        'code_init_std':.01,'code_norm_weight':1e-4/math.sqrt(HiresConfig().latent_dim/336),
        'code_norm_scaling':'1e-4 / sqrt(latent_dim / 336), multiplying per-image L2 norm',
        'edge_weight':.5,'evaluation_steps':[2000,10000,20000,30000,40000,50000],
        'checkpoint_steps':[0,10000,50000],'training_probe_indices':probe,
        'render_image_batch':16,'render_query_batch':512,
        'new_image_indices':list(range(64)),'new_code_steps':1000,'new_code_queries_per_image':128,
        'completion_observed_indices':observed,'quantum_audit_query_indices':audit_points,
        'quantum_audit_image_indices':list(range(16)),
        'quantum_noise_image_indices':list(range(4)),
        'shots':[256,1024,4096],'equal_total_shot_budgets':[768,3072,12288],
        'measurement_seeds':[101,211,307,401,503],
        'noise_levels':[{'name':'ideal','p1':0.,'p2':0.,'readout':0.},
                        {'name':'low','p1':1e-4,'p2':1e-3,'readout':.001},
                        {'name':'moderate','p1':1e-3,'p2':.01,'readout':.01}],
        'training_device':'physical GPU0 only; CUDA_VISIBLE_DEVICES=0',
        'frozen_decoder_tests':['development_completion','test_fully_observed','test_completion'],
        'comparisons':{'joint_statistics':['spatial_reference','marginal_products'],
            'sequential_scales':['spatial_reference'],
            'complex_phases':['spatial_reference','extra_real_gates'],
            'separate_scales':['spatial_reference','three_mixed'],
            'context_baseline':['attention_chain5']},
        'limits':['No hardware experiment or quantum advantage claim.',
            '256px targets are centre crops downsampled from originals with both dimensions >=256; no upsampling.',
            'Training reconstruction metrics use a predeclared 256-image probe, not all 4744 images.',
            'Dataset, latent budget, coordinate budget and positional frequencies differ from Z5; raw metrics are not a controlled resolution ablation.',
            'New-image codes are optimized with a frozen field; no learned encoder.',
            'Three independent heads use three circuit executions and fixed RGB averaging; no cross-head entanglement.',
            'Same 50000 updates and sampled coordinates; no equal wall-clock or equal total quantum-query claim.',
            '2048 coordinate endpoints/image/update: approximately 5.27 pixel presentations per training pixel on average, not 50000 full-image passes.',
            'Marginal-product and joint readouts have the same 30 learned classical parameters.',
            'Extra-real gate control matches 30 added stored parameters and added CNOT count, not every one-qubit gate count.',
            'Original Stanford server unavailable: pinned mirror split membership not independently checked against original MATLAB files.',
            'Exact selected image duplicates removed; near duplicates not independently excluded.']}
    write_json(root/'protocol.json',protocol);write_json(root/'data_manifest.json',manifest)
    runs=[]
    for seed in protocol['seeds']:
        for name in NAMES:
            directory=root/f'{name}_seed{seed}';directory.mkdir()
            item={'architecture':name,'seed':seed,'directory':str(directory)}
            write_json(directory/'run.json',item);runs.append(item)
    write_json(root/'runs.json',runs)
    return protocol


def assert_sources(root):
    for rel,digest in json.loads((Path(root)/'source_manifest.json').read_text()).items():
        if sha256(REPO/rel)!=digest:raise RuntimeError(f'Frozen source changed: {rel}')


def load_model(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['kind']!=KIND:raise ValueError('Wrong checkpoint kind')
    cfg=HiresConfig(**saved['model_config']);model=HiresField(cfg).to(device)
    model.load_state_dict(saved['decoder_state'],strict=True)
    return model.eval(),saved['codes'].to(device),saved


def targets(images,rows,points,device):
    # Advanced indexing touches only selected RGB pixels, not full image batches.
    values=np.asarray(images).reshape(len(images),-1,3)[np.asarray(rows)[:,None],np.asarray(points)[None]]
    return torch.from_numpy(values.astype(np.float32)).to(device).div_(255.)


def metrics(prediction,original,mask=None):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(original,dtype=np.float64)/255.
    assert p.shape==y.shape and p.ndim==4 and p.shape[-1]==3
    e=p-y;n,h,w,_=e.shape
    chosen=np.arange(h*w) if mask is None else np.asarray(mask)
    mse=np.mean(e.reshape(n,-1,3)[:,chosen]**2,axis=(1,2))
    allowed=np.zeros(h*w,dtype=bool);allowed[chosen]=True;allowed=allowed.reshape(h,w)
    hm=allowed[:,:-1]&allowed[:,1:];vm=allowed[:-1]&allowed[1:]
    horizontal=(e[:,:,1:]-e[:,:,:-1])[:,hm]
    vertical=(e[:,1:]-e[:,:-1])[:,vm]
    edge=.5*(np.mean(horizontal**2,axis=(1,2))+np.mean(vertical**2,axis=(1,2)))
    psnr=-10*np.log10(np.maximum(mse,1e-30))
    return {'psnr_db':float(psnr.mean()),'mse':float(mse.mean()),'gradient_mse':float(edge.mean()),
        'per_image_psnr_db':psnr.tolist(),'per_image_mse':mse.tolist(),
        'per_image_gradient_mse':edge.tolist(),'images':n,'scored_pixels_per_image':len(chosen),
        'horizontal_edges':int(hm.sum()),'vertical_edges':int(vm.sum()),'rgb_clipped_before_metrics':False}


@torch.no_grad()
def render(model,codes,p):
    size=model.cfg.image_size;coords=pixel_coordinates(size,size,device=codes.device)
    result=np.empty((len(codes),size*size,3),dtype=np.float32)
    model.eval();model.cache_quantum()
    for start in range(0,len(codes),p['render_image_batch']):
        chunk=codes[start:start+p['render_image_batch']]
        for offset in range(0,size*size,p['render_query_batch']):
            y=model(chunk,coords[offset:offset+p['render_query_batch']])
            result[start:start+len(chunk),offset:offset+y.shape[1]]=y.cpu().numpy()
    model.clear_cache()
    return result.reshape(len(codes),size,size,3)


def fit_new_codes(model,images,p,seed,split,mode,directory,device):
    name=f'{split}_{mode}';before=state_hash(model)
    for param in model.parameters():param.requires_grad_(False)
    model.eval();model.cache_quantum()
    rng_offset=100000 if split=='development' else 200000
    generator=torch.Generator(device=device).manual_seed(seed+rng_offset)
    codes=torch.nn.Parameter(torch.randn(len(images),model.cfg.latent_dim,generator=generator,device=device)*p['code_init_std'])
    initial_hash=hashlib.sha256(codes.detach().cpu().numpy().tobytes()).hexdigest()
    optimizer=torch.optim.Adam([codes],lr=p['code_lr'])
    query_rng=torch.Generator().manual_seed(seed+rng_offset+1)
    allowed=torch.tensor(p['completion_observed_indices']) if mode=='completion' else torch.arange(model.cfg.image_size**2)
    coords=pixel_coordinates(model.cfg.image_size,model.cfg.image_size,device=device)
    stream=hashlib.sha256();start=time.perf_counter();rows=np.arange(len(images))
    for step in range(p['new_code_steps']):
        points=allowed[torch.randint(len(allowed),(p['new_code_queries_per_image'],),generator=query_rng)]
        stream.update(points.numpy().tobytes());optimizer.zero_grad(set_to_none=True)
        y=model(codes,coords[points.to(device)])
        loss=(y-targets(images,rows,points.numpy(),device)).square().mean()+p['code_norm_weight']*codes.norm(dim=1).mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'{name} step {step}')
        loss.backward();optimizer.step()
    model.clear_cache();prediction=render(model,codes,p)
    hidden=np.setdiff1d(np.arange(model.cfg.image_size**2),p['completion_observed_indices'])
    scored=hidden if mode=='completion' else None
    result={'primary_metrics':metrics(prediction,images,scored),'whole_image_metrics':metrics(prediction,images),
        'code_steps':p['new_code_steps'],'training_coordinates_per_image':p['new_code_steps']*p['new_code_queries_per_image'],
        'initial_codes_sha256':initial_hash,'query_stream_sha256':stream.hexdigest(),
        'decoder_before_sha256':before,'decoder_after_sha256':state_hash(model),'elapsed_seconds':time.perf_counter()-start}
    assert result['decoder_before_sha256']==result['decoder_after_sha256']
    save_checkpoint(directory/(name+'_codes.pt'),{'codes':codes,'initial_codes_sha256':initial_hash,'query_stream_sha256':stream.hexdigest()})
    np.save(directory/(name+'_reconstructions.npy'),prediction);write_json(directory/(name+'_metrics.json'),result)
    print(f'{name}: PSNR {result["primary_metrics"]["psnr_db"]:.3f}, edge {result["primary_metrics"]["gradient_mse"]:.7f}',flush=True)
    return result


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
        payload={'kind':KIND,'model_config':asdict(model.cfg),'seed':seed,'step':step,'epoch':epoch,
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
        'training_quantum_queries':step*p['batch_size']*4*p['pairs_per_orientation']*len(model.heads),
        'resources':resources(model),'elapsed_seconds':training_seconds,'checkpoint_sha256':sha256(checkpoint_path)}
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
    parser.add_argument('--create',action='store_true');parser.add_argument('--architecture',choices=NAMES);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.create:create_protocol(args.root)
    else:train_one(args.root,args.architecture,args.seed)

if __name__=='__main__':main()
