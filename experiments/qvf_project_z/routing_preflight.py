"""Full-size discarded GPU0 training benchmark for every Z7 condition."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from .routing_models import NAMES,build_model,resource_summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    device=configure_runtime('cuda:0',7);results=[]
    for name in NAMES:
        model=build_model(name,7).to(device);generator=torch.Generator(device=device).manual_seed(7)
        codes=torch.nn.Parameter(torch.randn(4744,model.cfg.latent_dim,generator=generator,device=device)*.01)
        coords=torch.rand(2048,2,generator=generator,device=device);target=torch.rand(16,2048,3,generator=generator,device=device)
        optimizer=torch.optim.Adam([*model.parameters(),codes],lr=.001);times=[];torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            prediction=model(codes[:16],coords);loss=(prediction-target).square().mean()
            assert torch.isfinite(loss);loss.backward();optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        row={'variant':name,'discarded_synthetic':True,'steps':8,'n_codes':4744,'batch_size':16,'queries_per_image':2048,
            'seconds_per_step_after_two_warmup':sum(times[2:])/6,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'resources':resource_summary(model)}
        results.append(row);write_json(a.output,results);print(json.dumps(row),flush=True)
        del model,codes,optimizer,prediction,loss;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
