"""Trained Z6 quantum feasibility: independent states, gradients, shots, noise.

These are coordinate-subset simulation diagnostics, not hardware results or
full-image shot/noise renders. Joint statistics share the same measured shots.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from experiments.qvf_project_a.models import pixel_coordinates
from .hires_study import assert_sources,load_model,targets
from .hires_models import probabilities,resources
from .hires_gates import (pennylane_state,primitives,density,readout_numpy,
    sampled_readout,expected_mse,gate_resources)


def audit(root):
    root=Path(root).resolve();assert_sources(root);p=json.loads((root/'protocol.json').read_text())
    device=configure_runtime('cuda:0',0);data=np.load(Path(p['data_root'])/'test.npy',mmap_mode='r')
    coordinates=pixel_coordinates(p['model_config']['image_size'],p['model_config']['image_size'],device=device)
    selected=p['quantum_audit_query_indices'];rows=p['quantum_audit_image_indices'];out=[]
    audit_dir=root/'quantum_audit';audit_dir.mkdir(exist_ok=False)
    for item in json.loads((root/'runs.json').read_text()):
        directory=Path(item['directory']);model,_,saved=load_model(directory/'fit/checkpoints'/f'step{p["steps"]}.pt',device)
        del saved,_
        fitted=torch.load(directory/'final/test_completion_codes.pt',map_location='cpu',weights_only=False)['codes']
        codes=fitted[rows].to(device)
        with torch.no_grad():
            model.cache_quantum();inputs,ps,rgb=model.trace(codes,coordinates[torch.tensor(selected,device=device)])
        probs=[x.cpu().numpy() for x in ps];input_arrays=[x.cpu().numpy() for x in inputs]
        target=targets(data,rows,selected,'cpu').flatten(0,1).double().numpy()
        # Reconstruct target independently at full float64 precision.
        target=np.asarray(data).reshape(len(data),-1,3)[np.asarray(rows)[:,None],np.asarray(selected)[None]].reshape(-1,3).astype(np.float64)/255.
        independent_rgb=readout_numpy(model,probs)
        readout_error=float(np.max(np.abs(independent_rgb-rgb.cpu().flatten(0,1).numpy())))
        assert readout_error<2e-4,(item,readout_error)
        shot_rows=[]
        settings=[('per_head',n,n*len(model.heads)) for n in p['shots']]
        settings += [('equal_total',total//len(model.heads),total) for total in p['equal_total_shot_budgets']]
        for scope,n,total in settings:
            expectation=expected_mse(model,probs,target,n)
            sampled=[]
            for seed in p['measurement_seeds']:
                pred=sampled_readout(model,probs,n,seed)
                sampled.append(float(np.mean((pred-target)**2)))
            shot_rows.append({'comparison':scope,'shots_per_head':n,'total_shots_per_coordinate':total,
                'exact_expected_mse':float(expectation.mean()),'sampled_mse_by_measurement_seed':sampled,
                'sampled_mean_mse':float(np.mean(sampled)),
                'empirical_standard_error_across_measurement_replicates':float(np.std(sampled,ddof=1)/np.sqrt(len(sampled))),
                'marginal_product_estimator_bias_included':True})
        model.clear_cache();model=model.cpu().double()
        state_max=0.;gradient_max=0.;compiled=[];noise=[]
        for i,head in enumerate(model.heads):
            x=torch.tensor(input_arrays[i][0],dtype=torch.float64,requires_grad=True)
            # Normalize both paths on the same amplitude manifold.
            xnorm=x/x.norm() if model.is_attention else x
            state=head.state(xnorm[None],model.is_attention)[0]
            independent=pennylane_state(head,xnorm,model.is_attention)
            state_error=float((state.to(torch.complex128)-independent).abs().max());state_max=max(state_max,state_error)
            assert state_error<1e-11
            observable=torch.arange(len(state),dtype=torch.float64).sin()
            parameters=[x,*head.parameters()]
            ga=torch.autograd.grad((probabilities(state)*observable).sum(),parameters,retain_graph=True)
            gb=torch.autograd.grad((probabilities(independent)*observable).sum(),parameters)
            grad_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb));gradient_max=max(gradient_max,grad_error)
            assert grad_error<1e-10
            operations=primitives(head,input_arrays[i][:1],model.is_attention)
            compiled.append(gate_resources(operations,head.n_qubits))
            declared=resources(model)
            for gate in ('RY','RZ','CNOT'):
                assert sum(x[0]==gate for x in operations)*len(model.heads)==declared[f'primitive_{gate}_per_coordinate']
        if item['seed']==p['seeds'][0]:
            count=len(p['quantum_noise_image_indices'])*len(selected)
            assert p['quantum_noise_image_indices']==rows[:len(p['quantum_noise_image_indices'])]
            for level in p['noise_levels']:
                noisy=[];ideal_error=0.
                for head,arr,reference in zip(model.heads,input_arrays,probs):
                    parts=[]
                    for offset in range(0,count,32):
                        chunk=arr[offset:min(offset+32,count)]
                        ops=primitives(head,chunk,model.is_attention)
                        parts.append(density(ops,len(chunk),head.n_qubits,device=device,
                            p1=level['p1'],p2=level['p2'],readout=level['readout']))
                    result=np.concatenate(parts);noisy.append(result)
                    if level['name']=='ideal':
                        normalized=reference[:count]/reference[:count].sum(-1,keepdims=True)
                        ideal_error=max(ideal_error,float(np.max(np.abs(result-normalized))))
                pred=readout_numpy(model,noisy)
                if level['name']=='ideal':assert ideal_error<2e-5
                noise.append({**level,'sampled_coordinates':count,'mse':float(np.mean((pred-target[:count])**2)),
                    'ideal_primitive_probability_max_abs_error':ideal_error if level['name']=='ideal' else None,
                    'noise_includes_preparation_and_all_compiled_gates':True,'routing_noise_included':False})
                print(f'NOISE {item["architecture"]} {level["name"]} MSE={noise[-1]["mse"]:.7f}',flush=True)
        result={**item,'sampled_coordinates':len(target),'readout_max_abs':readout_error,
            'native_state_max_abs':state_max,'native_gradient_max_abs':gradient_max,
            'ideal_coordinate_subset_mse':float(np.mean((independent_rgb-target)**2)),
            'gate_resources_per_head':compiled,'shots':shot_rows,'noise_seed0':noise,
            'scope':f'{len(rows)} held-out images x {len(selected)} predeclared unobserved coordinates; noise first {len(p["quantum_noise_image_indices"])} images, seed0 only',
            'hardware_execution':False,'passed':True}
        write_json(audit_dir/f'{item["architecture"]}_seed{item["seed"]}.json',result)
        out.append(result);write_json(root/'quantum_audit.json',{'runs':out,'complete':len(out)==len(json.loads((root/'runs.json').read_text()))})
        print(f'AUDITED {item["architecture"]} seed={item["seed"]}',flush=True)
        del model,ps,inputs,rgb;torch.cuda.empty_cache()
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
