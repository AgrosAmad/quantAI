"""Fresh 1k/5k/10k development-only code-fitting curves with frozen decoders."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime,save_checkpoint,write_json,sha256
from experiments.qvf_project_a.models import pixel_coordinates
from .hires_study import targets,metrics,render,state_hash,assert_sources,load_model as parent_load
from .routing_study import load_model


def run(root):
    root=Path(root).resolve();p=json.loads((root/'protocol.json').read_text());assert_sources(root)
    output=root/'extended_development';output.mkdir(exist_ok=False)
    device=configure_runtime('cuda:0',0);data=np.load(Path(p['data_root'])/'development.npy',mmap_mode='r')[p['new_image_indices']]
    hidden=np.setdiff1d(np.arange(p['model_config']['image_size']**2),p['completion_observed_indices'])
    all_results=[]
    for name in p['extended_development_models']:
        for seed in p['seeds']:
            directory=(root if name in p['architectures'] else Path(p['parent_benchmark']))/f'{name}_seed{seed}'
            checkpoint=directory/'fit/checkpoints'/f'step{p["steps"]}.pt'
            loader=load_model if name in p['architectures'] else parent_load
            model,unused,saved=loader(checkpoint,device);del unused,saved
            original=json.loads((directory/'final/development_completion_metrics.json').read_text())
            before=state_hash(model)
            for parameter in model.parameters():parameter.requires_grad_(False)
            model.eval();model.cache_quantum()
            generator=torch.Generator(device=device).manual_seed(seed+100000)
            codes=torch.nn.Parameter(torch.randn(len(data),model.cfg.latent_dim,generator=generator,device=device)*p['code_init_std'])
            initial=hashlib.sha256(codes.detach().cpu().numpy().tobytes()).hexdigest()
            assert initial==original['initial_codes_sha256']
            optimizer=torch.optim.Adam([codes],lr=p['code_lr']);query=torch.Generator().manual_seed(seed+100001)
            allowed=torch.tensor(p['completion_observed_indices']);coords=pixel_coordinates(model.cfg.image_size,model.cfg.image_size,device=device)
            rows=np.arange(len(data));stream=hashlib.sha256();records=[]
            target_dir=output/f'{name}_seed{seed}';target_dir.mkdir()
            for step in range(1,max(p['extended_development_code_steps'])+1):
                points=allowed[torch.randint(len(allowed),(p['new_code_queries_per_image'],),generator=query)]
                stream.update(points.numpy().tobytes());optimizer.zero_grad(set_to_none=True)
                y=model(codes,coords[points.to(device)])
                loss=(y-targets(data,rows,points.numpy(),device)).square().mean()+p['code_norm_weight']*codes.norm(dim=1).mean()
                assert torch.isfinite(loss);loss.backward();optimizer.step()
                if step in p['extended_development_code_steps']:
                    image=render(model,codes,p);scores=metrics(image,data,hidden)
                    record={'step':step,'metrics':scores,'query_stream_sha256':stream.hexdigest()}
                    if step==p['new_code_steps']:
                        assert stream.hexdigest()==original['query_stream_sha256']
                        record['psnr_difference_from_primary_fit']=scores['psnr_db']-original['primary_metrics']['psnr_db']
                        assert abs(record['psnr_difference_from_primary_fit'])<.02
                    np.save(target_dir/f'step{step}.npy',image);save_checkpoint(target_dir/f'codes_step{step}.pt',{'codes':codes})
                    records.append(record);print(f'{name} seed={seed} code_steps={step} PSNR={scores["psnr_db"]:.3f}',flush=True)
                    model.cache_quantum()
            assert before==state_hash(model)
            item={'architecture':name,'seed':seed,'initial_codes_sha256':initial,'frozen_decoder_sha256':before,
                'decoder_checkpoint_sha256':sha256(checkpoint),'development_only':True,'records':records}
            write_json(target_dir/'curve.json',item);all_results.append(item)
            write_json(root/'extended_development.json',{'runs':all_results,'complete':len(all_results)==len(p['extended_development_models'])*len(p['seeds'])})
            del model,codes,optimizer,y,loss;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
