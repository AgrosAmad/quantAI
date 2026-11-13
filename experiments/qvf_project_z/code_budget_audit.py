"""Z10 original-QVF quantum checks, including finite shots and noisy gate execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from experiments.qvf_project_a.models import pixel_coordinates
from .hires_study import assert_sources
from .hires_models import probabilities
from .hires_gates import (pennylane_state, density, sample_probabilities,
                         expected_mse, RGB, readout_numpy)
from .routing_gates import native_qnode
from .routing_audit import noise_run
from .spatial_gates import shot_mse_moments
from .code_budget_study import load_model
from .coverage_models import (Z6,Z7,trace,readout,vanilla_native,vanilla_primitives)
from .code_budget_models import resources as budget_resources

def resources(model, name):
    return budget_resources(model, model.cfg.architecture)


def native_check(model,name,arrays):
    model.clear_cache();model=model.cpu().double()
    if name in Z7 and name!='classical_attention16':
        xs=[torch.tensor(x[:1],dtype=torch.float64,requires_grad=True) for x in arrays]
        fast=model.quantum_state(xs)[0];native=native_qnode(model,[x[0] for x in xs])()
        params=[*xs,*[v for k,v in model.named_parameters() if v.requires_grad and
                (k.startswith(('routing','phase')) or k.startswith('base.heads'))]]
        cases=[(fast,native,params)]
    else:
        cases=[]
        if name=='vanilla_qvf':
            x=torch.tensor(arrays[0][0],dtype=torch.float64,requires_grad=True);xn=x/x.norm()
            q=model.field.quantum
            cases.append((xn@q.circuit_matrix(),vanilla_native(model,xn),[x,*q.parameters()]))
        else:
            heads=model.base.heads if name=='classical_attention16' else model.heads
            amplitude=name=='attention_chain5'
            for head,arr in zip(heads,arrays):
                x=torch.tensor(arr[0],dtype=torch.float64,requires_grad=True);xn=x/x.norm() if amplitude else x
                cases.append((head.state(xn[None],amplitude)[0],pennylane_state(head,xn,amplitude),[x,*head.parameters()]))
    errors=[]
    for fast,native,params in cases:
        obs=torch.arange(len(fast),dtype=torch.float64).sin()
        ga=torch.autograd.grad((probabilities(fast)*obs).sum(),params,retain_graph=True)
        gb=torch.autograd.grad((probabilities(native)*obs).sum(),params)
        e=float((fast-native).abs().max());g=max(float((a-b).abs().max()) for a,b in zip(ga,gb))
        assert e<1e-11 and g<1e-10,(name,e,g)
        errors.append((e,g))
    return model,max(x[0] for x in errors),max(x[1] for x in errors)


def audit(root):
    root=Path(root).resolve();assert_sources(root);p=json.loads((root/'protocol.json').read_text())
    device=configure_runtime(p.get('device','cuda:0'),0)
    images=np.load(Path(p['data_root'])/'test.npy',mmap_mode='r')[p['new_image_indices']]
    coords=pixel_coordinates(p['model_config']['image_size'],p['model_config']['image_size'],device=device)
    rows=p['quantum_audit_image_indices'];ids=p['quantum_audit_query_indices'];xy=coords[ids].cpu().double().numpy()
    target=images.reshape(len(images),-1,3)[np.asarray(rows)[:,None],np.asarray(ids)[None]].reshape(-1,3).astype(np.float64)/255
    lookup=[ids.index(i) for i in p['noise_query_indices']];noise_rows=p['quantum_noise_image_indices']
    assert noise_rows==rows[:len(noise_rows)]
    noise_index=np.array([r*len(ids)+j for r in range(len(noise_rows)) for j in lookup])
    runs=json.loads((root/'runs.json').read_text());results=[]
    for run in runs:
        name='vanilla_qvf';d=Path(run['directory'])
        model,unused,saved=load_model(d/'fit/checkpoints'/f'step{p["steps"]}.pt',device);del unused,saved
        codes=torch.load(d/'final/test_completion_codes.pt',map_location='cpu',weights_only=False)['codes'][rows].to(device)
        with torch.no_grad():
            model.cache_quantum();inputs,ps,rgb=trace(model,name,codes,coords[ids])
        arrays=[x.cpu().numpy() for x in inputs];probs=[x.cpu().numpy() for x in ps]
        independent=readout(model,name,probs,xy,len(rows));readout_error=float(np.max(np.abs(independent-rgb.cpu().flatten(0,1).numpy())))
        assert readout_error<2e-4
        calls=resources(model,name)['logical_circuits_per_coordinate'];shots=[]
        settings=[('per_circuit',n,n*calls) for n in p['shots']]+[('equal_total',n//calls,n) for n in p['equal_total_shot_budgets']]
        for scope,n,total in settings:
            fixed_routing=name in Z7 and name!='classical_attention16'
            if fixed_routing:
                marginal=probs[0].reshape(-1,4,128).sum(1);marginal/=marginal.sum(-1,keepdims=True)
                expectation,_=shot_mse_moments(marginal,target,n)
            elif name!='classical_attention16':expectation=expected_mse(model,probs,target,n)
            measurements=[]
            for seed in p['measurement_seeds']:
                if fixed_routing:y=sample_probabilities(marginal,n,seed)@RGB
                else:
                    samples=[sample_probabilities(x,n,seed+100003*i) for i,x in enumerate(probs)]
                    y=readout(model,name,samples,xy,len(rows))
                measurements.append(float(np.mean((y-target)**2)))
            shots.append({'comparison':scope,'shots_per_circuit':n,'total_shots_per_coordinate':total,
                'sampled_mse_by_measurement_seed':measurements,'sampled_mean_mse':float(np.mean(measurements)),
                'exact_expected_mse':None if name=='classical_attention16' else float(np.mean(expectation)),
                'nonlinear_softmax_requires_replicates':name=='classical_attention16'})
        model,state_error,grad_error=native_check(model,name,arrays);noise=[]
        if run['seed']==p['seeds'][0]:
            aa=[x[noise_index] for x in arrays];pp=[x[noise_index] for x in probs]
            if name!='vanilla_qvf':
                noise=noise_run(model,aa,pp,target[noise_index],xy[lookup],len(noise_rows),p['noise_levels'],device,is_parent=name in Z6)
            else:
                for level in p['noise_levels']:
                    parts=[]
                    for offset in range(0,len(noise_index),16):
                        x=aa[0][offset:offset+16]
                        parts.append(density(vanilla_primitives(model,x),len(x),5,device=device,
                                             **{k:level[k] for k in ('p1','p2','readout')}))
                    prob=np.concatenate(parts);error=float(np.max(np.abs(prob-pp[0]/pp[0].sum(-1,keepdims=True))))
                    if level['name']=='ideal':assert error<3e-5
                    pred=readout_numpy(model,[prob])
                    noise.append({**level,'mse':float(np.mean((pred-target[noise_index])**2)),
                        'coordinates':len(noise_index),'ideal_probability_max_abs_error':error if level['name']=='ideal' else None})
        row={**run,'native_state_max_abs':state_error,'native_gradient_max_abs':grad_error,
             'readout_max_abs':readout_error,'shots':shots,'noise_seed0':noise,
             'resources':resources(model,name),'passed':True,'hardware_execution':False}
        results.append(row);write_json(root/'quantum_audit.json',{'runs':results,'complete':len(results)==len(runs)})
        print('AUDITED',run['architecture'],run['seed'],flush=True)
        del model,codes,inputs,ps,rgb;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);audit(parser.parse_args().root)
