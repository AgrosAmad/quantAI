"""Z9 native circuits/gradients, shared-bitstring measurements and gate noise."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from experiments.qvf_project_a.models import pixel_coordinates
from .hires_study import assert_sources
from .hires_gates import sample_probabilities,expected_mse,RGB
from .routing_audit import noise_run
from .spatial_gates import shot_mse_moments
from .coverage_audit import native_check
from .combination_study import load_model
from .combination_models import ALIASES,trace,readout,resources


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
        name=run['architecture'];alias=ALIASES[name];d=Path(run['directory'])
        fusion=alias=='classical_attention16';coherent=alias=='coherent_routing'
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
            if coherent:
                marginal=probs[0].reshape(-1,4,128).sum(1);marginal/=marginal.sum(-1,keepdims=True)
                expectation,_=shot_mse_moments(marginal,target,n)
            elif not fusion:expectation=expected_mse(model,probs,target,n)
            measurements=[]
            for seed in p['measurement_seeds']:
                if coherent:y=sample_probabilities(marginal,n,seed)@RGB
                else:
                    samples=[sample_probabilities(x,n,seed+100003*i) for i,x in enumerate(probs)]
                    y=readout(model,name,samples,xy,len(rows))
                measurements.append(float(np.mean((y-target)**2)))
            shots.append({'comparison':scope,'shots_per_circuit':n,'total_shots_per_coordinate':total,
                'sampled_mse_by_measurement_seed':measurements,'sampled_mean_mse':float(np.mean(measurements)),
                'exact_expected_mse':None if fusion else float(np.mean(expectation)),
                'nonlinear_softmax_requires_replicates':fusion})
        model,state_error,grad_error=native_check(model,alias,arrays);noise=[]
        if run['seed']==p['seeds'][0]:
            noise=noise_run(model,[x[noise_index] for x in arrays],[x[noise_index] for x in probs],
                target[noise_index],xy[lookup],len(noise_rows),p['noise_levels'],device,
                is_parent=alias=='separate_scales')
        row={**run,'native_state_max_abs':state_error,'native_gradient_max_abs':grad_error,
             'readout_max_abs':readout_error,'shots':shots,'noise_seed0':noise,
             'resources':resources(model,name),'passed':True,'hardware_execution':False}
        results.append(row);write_json(root/'quantum_audit.json',{'runs':results,'complete':len(results)==len(runs)})
        print('AUDITED',name,run['seed'],flush=True)
        del model,codes,inputs,ps,rgb;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);audit(parser.parse_args().root)
