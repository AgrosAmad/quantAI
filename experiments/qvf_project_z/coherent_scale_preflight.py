"""Discarded synthetic GPU0 feasibility; no image-quality training claim."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json,sha256
from .quantum_scale_attention import CoherentScaleField,compiled_resources
from .hires_models import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();device=configure_runtime('cuda:0',20261111);records=[]
    for name in ('independent_three_heads','coherent_scale_routing'):
        model=(build_model('separate_scales',7) if name=='independent_three_heads' else CoherentScaleField(7)).to(device)
        generator=torch.Generator(device=device).manual_seed(7)
        codes=torch.nn.Parameter(torch.randn(4744,model.cfg.latent_dim,device=device,generator=generator)*.01)
        optimizer=torch.optim.Adam([*model.parameters(),codes],lr=.001)
        coordinates=torch.rand(2048,2,device=device,generator=generator)
        target=torch.rand(16,2048,3,device=device,generator=generator)
        durations=[];torch.cuda.reset_peak_memory_stats()
        for step in range(10):
            torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            prediction=model(codes[:16],coordinates);loss=(prediction-target).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic loss')
            loss.backward()
            if name=='coherent_scale_routing':
                assert torch.isfinite(model.routing_angles.grad).all() and torch.isfinite(model.phase_angles.grad).all()
                routing_norm=float(model.routing_angles.grad.norm());phase_norm=float(model.phase_angles.grad.norm())
            optimizer.step();torch.cuda.synchronize();durations.append(time.perf_counter()-start)
        result={'architecture':name,'discarded_synthetic_benchmark':True,'updates':10,
            'batch_size':16,'coordinates_per_image':2048,'images_in_code_table':4744,
            'seconds_per_step_after_two_warmup':sum(durations[2:])/8,
            'individual_seconds':durations,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'no_image_quality_conclusion':True,'physical_gpu':0}
        if name=='coherent_scale_routing':result.update({'last_routing_gradient_norm':routing_norm,'last_phase_gradient_norm':phase_norm})
        print(json.dumps(result),flush=True);records.append(result)
        del model,codes,optimizer,prediction,loss;gc.collect();torch.cuda.empty_cache()
    cpu_model=CoherentScaleField(7).double()
    sample=[torch.randn(5,5,dtype=torch.float64) for _ in range(3)]
    resources=compiled_resources(cpu_model,sample)
    sources=[Path(__file__),Path(__file__).with_name('quantum_scale_attention.py'),Path(__file__).with_name('test_quantum_scale_attention.py')]
    write_json(a.output,{'benchmarks':records,'coherent_circuit_resources':resources,
        'source_sha256':{str(f):sha256(f) for f in sources},'quality_study_launched':False})

if __name__=='__main__':main()
