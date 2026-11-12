"""Trained Z7 quantum, measurement and common-subset noise comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json,sha256
from experiments.qvf_project_a.models import pixel_coordinates
from .hires_study import assert_sources,load_model as parent_load
from .hires_gates import (primitives as head_primitives,density as head_density,
    pennylane_state,sample_probabilities,RGB,readout_numpy as parent_readout)
from .spatial_gates import shot_mse_moments,gate_resources
from .hires_models import probabilities
from .routing_study import load_model
from .routing_models import readout_numpy,resource_summary
from .routing_gates import native_qnode,primitives,density


def noise_run(model,inputs,reference,target,points,batch,levels,device,is_parent=False):
    results=[];is_fusion=is_parent or model.variant=='classical_attention16'
    for level in levels:
        noise_kwargs={k:level[k] for k in ('p1','p2','readout')};outputs=[]
        heads=model.heads if is_parent else model.base.heads if is_fusion else [None]
        for h,head in enumerate(heads):
            parts=[]
            for offset in range(0,len(target),8):
                end=min(offset+8,len(target))
                if is_fusion:
                    data=inputs[h][offset:end]
                    ops=head_primitives(head,data,model.is_attention if is_parent else False)
                    parts.append(head_density(ops,end-offset,head.n_qubits,device=device,**noise_kwargs))
                else:
                    data=[x[offset:end] for x in inputs];ops=primitives(model,data)
                    parts.append(density(ops,end-offset,device=device,**noise_kwargs))
            outputs.append(np.concatenate(parts))
        predicted=parent_readout(model,outputs) if is_parent else readout_numpy(model,outputs,points,batch)
        error=None
        if level['name']=='ideal':
            error=max(float(np.max(np.abs(a-b/b.sum(-1,keepdims=True)))) for a,b in zip(outputs,reference))
            assert error<3e-5
        result={**level,'mse':float(np.mean((predicted-target)**2)),
            'coordinates':len(target),'ideal_probability_max_abs_error':error,
            'includes_preparation':True,'includes_routing_noise':False}
        results.append(result);print('NOISE',getattr(model,'variant','parent'),level['name'],result['mse'],flush=True)
    return results


def audit(root):
    root=Path(root).resolve();assert_sources(root);p=json.loads((root/'protocol.json').read_text())
    device=configure_runtime('cuda:0',0);output=root/'quantum_audit';output.mkdir(exist_ok=False)
    images=np.load(Path(p['data_root'])/'test.npy',mmap_mode='r')[p['new_image_indices']]
    size=p['model_config']['image_size'];coords=pixel_coordinates(size,size,device=device)
    point_ids=p['quantum_audit_query_indices'];rows=p['quantum_audit_image_indices']
    point_coords=coords[torch.tensor(point_ids,device=device)];xy=point_coords.cpu().double().numpy()
    target=images.reshape(len(images),-1,3)[np.asarray(rows)[:,None],np.asarray(point_ids)[None]].reshape(-1,3).astype(np.float64)/255.
    lookup=[point_ids.index(i) for i in p['noise_query_indices']]
    noise_rows=p['quantum_noise_image_indices'];assert noise_rows==rows[:len(noise_rows)]
    flat_noise=np.array([r*len(point_ids)+j for r in range(len(noise_rows)) for j in lookup])
    noise_xy=xy[lookup];results=[]
    for run in json.loads((root/'runs.json').read_text()):
        directory=Path(run['directory']);model,unused,saved=load_model(directory/'fit/checkpoints'/f'step{p["steps"]}.pt',device)
        del unused,saved
        codes=torch.load(directory/'final/test_completion_codes.pt',map_location='cpu',weights_only=False)['codes'][rows].to(device)
        fusion=model.variant=='classical_attention16'
        with torch.no_grad():
            model.cache_quantum();trace=model.trace(codes,point_coords)
            inputs=trace[0];ps=trace[1] if fusion else [trace[2]];rgb=trace[-1]
        arrays=[x.cpu().numpy() for x in inputs];probs=[x.cpu().numpy() for x in ps]
        independently=readout_numpy(model,probs,xy,len(rows))
        readout_error=float(np.max(np.abs(independently-rgb.cpu().flatten(0,1).numpy())))
        assert readout_error<2e-4
        shots=[];heads=3 if fusion else 1
        settings=[('per_circuit',n,n*heads) for n in p['shots']]+[('equal_total',n//heads,n) for n in p['equal_total_shot_budgets']]
        for scope,n,total in settings:
            estimates=[]
            if not fusion:
                marginal=probs[0].reshape(-1,4,128).sum(1);marginal=marginal/marginal.sum(-1,keepdims=True)
                expectation,variance=shot_mse_moments(marginal,target,n)
            for seed in p['measurement_seeds']:
                if fusion:
                    samples=[sample_probabilities(x,n,seed+100003*i) for i,x in enumerate(probs)]
                    y=readout_numpy(model,samples,xy,len(rows))
                else:y=sample_probabilities(marginal,n,seed)@RGB
                estimates.append(float(np.mean((y-target)**2)))
            row={'comparison':scope,'shots_per_circuit':n,'total_shots_per_coordinate':total,
                'sampled_mse_by_measurement_seed':estimates,'sampled_mean_mse':float(np.mean(estimates)),
                'exact_expected_mse':None if fusion else float(expectation.mean()),
                'analytic_standard_error_of_replicate_mean':None if fusion else float(np.sqrt(variance.sum()/len(target)**2/len(estimates))),
                'expectation_method':'Measurement replicates; nonlinear softmax gate' if fusion else 'Exact joint-bitstring moments for fixed linear RGB'}
            if not fusion:
                assert abs(row['sampled_mean_mse']-row['exact_expected_mse'])<8*row['analytic_standard_error_of_replicate_mean']+1e-10
            shots.append(row)
        model.clear_cache();model=model.cpu().double();state_error=0.;gradient_error=0.;coherence=[]
        if fusion:
            for head,arr in zip(model.base.heads,arrays):
                x=torch.tensor(arr[0],dtype=torch.float64,requires_grad=True)
                fast=head.state(x[None])[0];native=pennylane_state(head,x)
                state_error=max(state_error,float((fast-native).abs().max()))
                obs=torch.arange(128,dtype=torch.float64).sin();params=[x,*head.parameters()]
                ga=torch.autograd.grad((probabilities(fast)*obs).sum(),params,retain_graph=True)
                gb=torch.autograd.grad((probabilities(native)*obs).sum(),params)
                gradient_error=max(gradient_error,*[float((a-b).abs().max()) for a,b in zip(ga,gb)])
        else:
            xs=[torch.tensor(x[:1],dtype=torch.float64,requires_grad=True) for x in arrays]
            fast=model.quantum_state(xs)[0];native=native_qnode(model,[x[0] for x in xs])()
            state_error=float((fast-native).abs().max());obs=torch.arange(512,dtype=torch.float64).sin()
            params=[*xs,*[v for k,v in model.named_parameters() if v.requires_grad and (k.startswith(('routing','phase')) or k.startswith('base.heads'))]]
            ga=torch.autograd.grad((probabilities(fast)*obs).sum(),params,retain_graph=True)
            gb=torch.autograd.grad((probabilities(native)*obs).sum(),params)
            gradient_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb))
            for index in range(min(2,len(target))):
                data=[torch.tensor(x[index],dtype=torch.float64) for x in arrays]
                with torch.no_grad():
                    ideal=probabilities(model.quantum_state([x[None] for x in data]))
                    dephased=np.asarray(native_qnode(model,data,dephase=True)())[None]
                a=readout_numpy(model,[ideal.numpy()],xy[index:index+1],1);b=readout_numpy(model,[dephased],xy[index:index+1],1)
                coherence.append({'flat_query_index':index,'coherence_removed_rgb_max_abs':float(np.max(np.abs(a-b))),
                    'ideal_mse':float(np.mean((a-target[index:index+1])**2)),
                    'dephased_mse':float(np.mean((b-target[index:index+1])**2))})
                if model.variant=='local_capacity9':assert coherence[-1]['coherence_removed_rgb_max_abs']<2e-10
        assert state_error<1e-11 and gradient_error<1e-10
        noise=[]
        if run['seed']==p['seeds'][0]:
            noise=noise_run(model,[x[flat_noise] for x in arrays],[x[flat_noise] for x in probs],
                target[flat_noise],noise_xy,len(noise_rows),p['noise_levels'],device)
        result={**run,'native_state_max_abs':state_error,'native_gradient_max_abs':gradient_error,
            'readout_max_abs':readout_error,'coordinate_subset_mse':float(np.mean((independently-target)**2)),
            'shots':shots,'noise_seed0':noise,'coherence_removal_diagnostic':coherence,
            'resources':resource_summary(model),'passed':True,'hardware_execution':False}
        write_json(output/f'{run["architecture"]}_seed{run["seed"]}.json',result);results.append(result)
        write_json(root/'quantum_audit.json',{'runs':results,'complete':False})
        print('AUDITED',run['architecture'],run['seed'],flush=True)
        del model,codes,trace,ps,inputs,rgb;torch.cuda.empty_cache()
    parent_results=[]
    for name in p['noise_parent_rechecks']:
        directory=Path(p['parent_benchmark'])/f'{name}_seed{p["seeds"][0]}'
        checkpoint=directory/'fit/checkpoints'/f'step{p["steps"]}.pt'
        summary=json.loads((directory/'summary.json').read_text());assert sha256(checkpoint)==summary['training']['checkpoint_sha256']
        model,unused,saved=parent_load(checkpoint,device);del unused,saved
        codes=torch.load(directory/'final/test_completion_codes.pt',map_location='cpu',weights_only=False)['codes'][rows].to(device)
        with torch.no_grad():
            model.cache_quantum();inputs,ps,_=model.trace(codes,point_coords)
        arrays=[x.cpu().numpy()[flat_noise] for x in inputs];probs=[x.cpu().numpy()[flat_noise] for x in ps]
        model.clear_cache();model=model.cpu().double()
        records=noise_run(model,arrays,probs,target[flat_noise],noise_xy,len(noise_rows),p['noise_levels'],device,is_parent=True)
        parent_results.append({'architecture':name,'seed':p['seeds'][0],'noise':records})
        del model,codes,inputs,ps;torch.cuda.empty_cache()
    write_json(root/'quantum_audit.json',{'runs':results,'parent_common_subset_noise':parent_results,'complete':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
