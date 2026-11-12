"""Disposable full pipeline test, paired with the existing completed Z6 smoke."""
import argparse
import json
from pathlib import Path
from experiments.qvf_project_a.train import write_json
from .routing_study import create_protocol,MATCH_KEYS,HERE
from .routing_run import run

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    p=create_protocol(a.root)
    parent=HERE/'results/20261110_z6_smoke';old=json.loads((parent/'protocol.json').read_text())
    p.update({k:old[k] for k in MATCH_KEYS})
    p.update({'parent_benchmark':str(parent),'discarded_synthetic_smoke':True,
        'noise_query_indices':old['quantum_audit_query_indices'][:2],
        'extended_development_code_steps':[5,10,15]})
    write_json(a.root/'protocol.json',p);run(a.root,2)

if __name__=='__main__':main()
