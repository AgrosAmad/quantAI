"""Short discarded GPU0 checks of the new vanilla run and a 32px coherent run."""
import json
from pathlib import Path
import time
import torch
from experiments.qvf_project_a.train import configure_runtime,write_json
from .coverage_models import build_model,resources


def main():
    device=configure_runtime('cuda:0',0);results=[]
    for name,size,grids,freq,queries in [('vanilla_qvf',256,(16,32,64),9,2048),
                                       ('coherent_routing',32,(2,4,8),6,128)]:
        model=build_model(name,0,image_size=size,grid_sizes=grids,channels=4,frequencies=freq,width=128).to(device)
        codes=torch.nn.Parameter(torch.randn(4744,model.cfg.latent_dim,device=device)*.01)
        optimizer=torch.optim.Adam([codes,*model.parameters()],lr=.001)
        xy=torch.rand(queries,2,device=device);target=torch.rand(16,queries,3,device=device)
        elapsed=[];torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            prediction=model(codes[:16],xy);loss=(prediction-target).square().mean()+1e-4*codes[:16].norm(dim=1).mean()
            loss.backward();optimizer.step();assert torch.isfinite(loss)
            torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
        results.append({'architecture':name,'resolution':size,'steps':8,'discarded':True,
            'mean_step_seconds_after_warmup':sum(elapsed[2:])/6,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'resources':resources(model,name),
            'concurrent_Z7_training_present':True})
        del model,codes,optimizer,xy,target,prediction,loss;torch.cuda.empty_cache()
    path=Path(__file__).resolve().parent/'results/coverage_preflight_20261112.json'
    write_json(path,{'runs':results,'quality_evidence':False});print(json.dumps(results),flush=True)


if __name__=='__main__':main()
