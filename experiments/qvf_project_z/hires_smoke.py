"""Disposable synthetic end-to-end test of all nine arms and the full pipeline."""
import argparse
from pathlib import Path
import numpy as np
from experiments.qvf_project_a.train import write_json,sha256
from .hires_models import HiresConfig
from .hires_study import create_protocol
from .hires_run import run
from dataclasses import asdict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    p=create_protocol(a.root);data=a.root.resolve()/'synthetic_data';data.mkdir()
    rng=np.random.default_rng(888)
    for split,count in [('train',8),('development',4),('test',4)]:
        np.save(data/(split+'.npy'),rng.integers(0,256,(count,16,16,3),dtype=np.uint8))
    write_json(data/'manifest.json',{'discarded_synthetic_smoke':True})
    mask=np.sort(rng.choice(256,64,replace=False));hidden=np.setdiff1d(np.arange(256),mask)
    p.update({'discarded_synthetic_smoke':True,'data_root':str(data),
        'data_manifest_sha256':sha256(data/'manifest.json'),
        'data_arrays_sha256':{s:sha256(data/(s+'.npy')) for s in ('train','development','test')},
        'model_config':asdict(HiresConfig(image_size=16,grid_sizes=(2,4,8),frequencies=3)),
        'n_train':8,'steps':12,'batch_size':2,'pairs_per_orientation':2,
        'evaluation_steps':[6,12],'checkpoint_steps':[0,6,12],
        'training_probe_indices':[0,1,2,3],'render_image_batch':2,'render_query_batch':64,
        'new_image_indices':[0,1,2,3],'new_code_steps':5,'new_code_queries_per_image':8,
        'completion_observed_indices':mask.tolist(),'quantum_audit_query_indices':hidden[:4].tolist(),
        'quantum_audit_image_indices':[0,1,2,3],'quantum_noise_image_indices':[0,1]})
    write_json(a.root/'protocol.json',p);run(a.root,6)

if __name__=='__main__':main()
