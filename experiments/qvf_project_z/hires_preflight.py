"""Discarded synthetic full-table training benchmark, restricted to GPU0."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from .hires_models import build_model,resources

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--coordinates',type=int,default=512)
    p.add_argument('--architecture',nargs='+',default=['spatial_reference','complex_phases','separate_scales','attention_chain5'])
    args=p.parse_args();device=configure_runtime('cuda:0',20261110);results=[]
    for name in args.architecture:
        model=build_model(name).to(device)
        codes=torch.nn.Parameter(torch.randn(4744,model.cfg.latent_dim,device=device)*.01)
        opt=torch.optim.Adam([*model.parameters(),codes],lr=.001)
        coordinates=torch.rand(args.coordinates,2,device=device);target=torch.rand(16,args.coordinates,3,device=device)
        durations=[];torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            torch.cuda.synchronize();start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            y=model(codes[torch.arange(16,device=device)],coordinates)
            loss=(y-target).square().mean();loss.backward();opt.step()
            torch.cuda.synchronize();durations.append(time.perf_counter()-start)
        result={'architecture':name,'discarded_synthetic_benchmark':True,'steps':args.steps,
            'seconds_per_step_after_two_warmup':sum(durations[2:])/len(durations[2:]),
            'individual_seconds':durations,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'n_images':4744,'batch_size':16,'query_coordinates_per_image':args.coordinates,'resources':resources(model)}
        results.append(result);print(json.dumps(result),flush=True);write_json(args.output,results)
        del model,codes,opt,y,loss;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
