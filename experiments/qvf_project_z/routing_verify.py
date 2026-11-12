"""Z7 independent verification, adapted from the frozen Z6 verifier."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,sha256,write_json
from experiments.qvf_project_a.edge_qvf import neighbour_pairs
from .hires_study import render,assert_sources,state_hash
from .routing_study import load_model,MATCH_KEYS
from .routing_models import build_model


def independent_metrics(prediction,original,mask=None):
    # Per-image, per-orientation loop independent of the training metric helper.
    n,h,w,_=original.shape;indices=np.arange(h*w) if mask is None else np.array(mask)
    keep=np.zeros((h,w),bool);keep.flat[indices]=True
    mse=[];edge=[]
    for predicted,actual in zip(prediction,original):
        error=np.asarray(predicted,dtype=np.float64)-actual.astype(np.float64)/255.
        mse.append(float(np.mean(error.reshape(-1,3)[indices]**2)))
        horizontal=[];vertical=[]
        for row in range(h):
            allowed=keep[row,1:]&keep[row,:-1]
            horizontal.append((error[row,1:][allowed]-error[row,:-1][allowed]).reshape(-1))
        for col in range(w):
            allowed=keep[1:,col]&keep[:-1,col]
            vertical.append((error[1:,col][allowed]-error[:-1,col][allowed]).reshape(-1))
        edge.append(float(.5*(np.mean(np.concatenate(horizontal)**2)+np.mean(np.concatenate(vertical)**2))))
    return {'psnr_db':float(np.mean(-10*np.log10(np.maximum(mse,1e-30)))),
        'mse':float(np.mean(mse)),'gradient_mse':float(np.mean(edge)),
        'per_image_mse':mse,'per_image_gradient_mse':edge}


def stream_replay(p,seed):
    order=torch.Generator().manual_seed(seed+11);query=torch.Generator().manual_seed(seed+12)
    tables=neighbour_pairs(p['model_config']['image_size']);visits=torch.zeros(p['n_train'],dtype=torch.int64)
    digest=hashlib.sha256();step=0
    while step<p['steps']:
        for rows in torch.randperm(p['n_train'],generator=order).split(p['batch_size']):
            if len(rows)!=p['batch_size']:continue
            points=torch.cat([t[torch.randint(len(t),(p['pairs_per_orientation'],),generator=query)] for t in tables]).flatten()
            digest.update(rows.numpy().tobytes());digest.update(points.numpy().tobytes());visits[rows]+=1;step+=1
            if step==p['steps']:break
    return digest.hexdigest(),visits


def verify(root):
    root=Path(root).resolve();p=json.loads((root/'protocol.json').read_text());assert_sources(root)
    device=configure_runtime('cuda:0',0);data=Path(p['data_root']);results=[]
    if not p.get('discarded_synthetic_smoke'):
        parent=Path(p['parent_benchmark']);original=json.loads((parent/'protocol.json').read_text())
        assert all(p[k]==original[k] for k in MATCH_KEYS)
        for rel,digest in json.loads((root/'parent_reference_hashes.json').read_text()).items():assert sha256(parent/rel)==digest
    for split,digest in p['data_arrays_sha256'].items():assert sha256(data/(split+'.npy'))==digest
    replay={seed:stream_replay(p,seed) for seed in p['seeds']}
    hidden=np.setdiff1d(np.arange(p['model_config']['image_size']**2),p['completion_observed_indices'])
    for item in json.loads((root/'runs.json').read_text()):
        directory=Path(item['directory']);summary=json.loads((directory/'summary.json').read_text())
        train=summary['training'];assert train['optimizer_steps']==p['steps']
        digest,visits=replay[item['seed']]
        assert digest==train['stream_sha256'] and visits.tolist()==train['visits_per_image']
        checkpoint=directory/'fit/checkpoints'/f'step{p["steps"]}.pt'
        assert sha256(checkpoint)==train['checkpoint_sha256']
        model,codes,saved=load_model(checkpoint,device)
        assert saved['variant']==item['architecture']
        assert saved['step']==p['steps'] and saved['protocol_sha256']==sha256(root/'protocol.json')
        beginning=torch.load(directory/'fit/checkpoints/step0.pt',map_location='cpu',weights_only=False)
        config=dict(beginning['model_config']);config.pop('architecture')
        fresh=build_model(item['architecture'],item['seed'],**config)
        assert beginning['step']==0
        for key,value in fresh.state_dict().items():assert torch.equal(value,beginning['decoder_state'][key]),key
        assert hashlib.sha256(beginning['codes'].numpy().tobytes()).hexdigest()==train['initial_codes_sha256']
        del beginning,fresh
        initial=torch.Generator(device=device).manual_seed(item['seed'])
        initial_codes=torch.randn(p['n_train'],model.cfg.latent_dim,generator=initial,device=device)*p['code_init_std']
        assert hashlib.sha256(initial_codes.cpu().numpy().tobytes()).hexdigest()==train['initial_codes_sha256']
        del initial_codes
        maximum_reload=0.;maximum_metric=0.;final=directory/'final'
        entries=[('train_probe',np.load(data/'train.npy',mmap_mode='r')[p['training_probe_indices']],codes[torch.tensor(p['training_probe_indices'],device=device)],None)]
        for split,mode in [('development','completion'),('test','fully_observed'),('test','completion')]:
            name=split+'_'+mode
            fitted=torch.load(final/(name+'_codes.pt'),map_location='cpu',weights_only=False)
            details=json.loads((final/(name+'_metrics.json')).read_text())
            assert details['decoder_before_sha256']==details['decoder_after_sha256']==state_hash(model)
            offset=100000 if split=='development' else 200000
            generator=torch.Generator(device=device).manual_seed(item['seed']+offset)
            init=torch.randn(len(p['new_image_indices']),model.cfg.latent_dim,generator=generator,device=device)*p['code_init_std']
            assert hashlib.sha256(init.cpu().numpy().tobytes()).hexdigest()==details['initial_codes_sha256']
            query=torch.Generator().manual_seed(item['seed']+offset+1);stream=hashlib.sha256()
            allowed=torch.tensor(p['completion_observed_indices']) if mode=='completion' else torch.arange(model.cfg.image_size**2)
            for _ in range(p['new_code_steps']):
                points=allowed[torch.randint(len(allowed),(p['new_code_queries_per_image'],),generator=query)]
                stream.update(points.numpy().tobytes())
                if mode=='completion':assert not np.isin(points.numpy(),hidden).any()
            assert stream.hexdigest()==details['query_stream_sha256']==fitted['query_stream_sha256']
            entries.append((name,np.load(data/(split+'.npy'),mmap_mode='r')[p['new_image_indices']],fitted['codes'].to(device),hidden if mode=='completion' else None))
        for name,original,reconstruction_codes,mask in entries:
            reference=np.load(final/(name+'_reconstructions.npy'))
            fresh=render(model,reconstruction_codes,p)
            difference=float(np.max(np.abs(reference-fresh)));maximum_reload=max(maximum_reload,difference)
            assert difference<=3e-6,(directory,name,difference)
            independent=independent_metrics(reference,original,mask)
            stored=json.loads((final/(name+'_metrics.json')).read_text())
            if name!='train_probe':stored=stored['primary_metrics']
            for key,value in independent.items():
                error=float(np.max(np.abs(np.asarray(value)-np.asarray(stored[key]))));maximum_metric=max(maximum_metric,error)
                assert error<2e-12,(name,key,error)
        result={**item,'checkpoint_replay_max_abs':maximum_reload,'independent_metric_max_abs':maximum_metric,
            'training_stream_replayed':True,'new_code_streams_replayed':True,'withheld_pixels_excluded_from_fit':True,'passed':True}
        results.append(result);print(json.dumps(result),flush=True);write_json(root/'verification.json',{'runs':results,'complete':len(results)==len(json.loads((root/'runs.json').read_text()))})
        del model,codes,saved,entries;torch.cuda.empty_cache()
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);verify(p.parse_args().root)
