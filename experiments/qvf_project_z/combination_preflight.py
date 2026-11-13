"""Discarded full-batch GPU0 resource checks; never quality evidence."""
import argparse
import json
from pathlib import Path
import time
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from .combination_models import NAMES,build_model,resources


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();device=configure_runtime('cuda:0',0);results=[]
    for size,grids,freq,queries in [(32,(2,4,8),6,128),(256,(16,32,64),9,2048)]:
        for name in NAMES:
            model=build_model(name,0,image_size=size,grid_sizes=grids,channels=4,frequencies=freq,width=128).to(device)
            codes=torch.nn.Parameter(torch.randn(4744,model.cfg.latent_dim,device=device)*.01)
            optimizer=torch.optim.Adam([codes,*model.parameters()],lr=.001)
            xy=torch.rand(queries,2,device=device);target=torch.rand(16,queries,3,device=device)
            elapsed=[];torch.cuda.reset_peak_memory_stats()
            for step in range(6):
                torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                prediction=model(codes[:16],xy);loss=(prediction-target).square().mean()+1e-4*codes[:16].norm(dim=1).mean()
                loss.backward();optimizer.step();assert torch.isfinite(loss)
                torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
            row={'architecture':name,'resolution':size,'steps':6,'discarded':True,
                'mean_step_seconds_after_warmup':sum(elapsed[2:])/4,
                'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
                'resources':resources(model,name),'concurrent_Z8_training_present':True}
            results.append(row);print(json.dumps(row),flush=True)
            write_json(args.output,{'runs':results,'quality_evidence':False,'complete':len(results)==6})
            del model,codes,optimizer,xy,target,prediction,loss;torch.cuda.empty_cache()


if __name__=='__main__':main()
