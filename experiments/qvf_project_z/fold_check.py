"""Check whether same-width grid refinement can be absorbed into fitted codes."""
import argparse
import json
from pathlib import Path
import torch
from experiments.qvf_project_a.train import configure_runtime, write_json, sha256
from .models import build_model
from .study import load_model


def run(root):
    root = Path(root)
    configure_runtime('cpu', 43)
    protocol = json.loads((root / 'protocol.json').read_text())
    results = []
    for spec in protocol['runs']:
        name, seed = spec['name'], spec['config']['seed']
        if name not in ('latent_conv', 'latent_pointwise'):
            continue
        checkpoint = root / spec['directory'] / 'fit/checkpoints/last.pt'
        model, codes, _ = load_model(checkpoint, torch.device('cpu'))
        model = model.double().requires_grad_(False)
        reference = build_model('attention_amp', seed).double().requires_grad_(False)
        reference.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('refiners.')})
        coordinates = torch.rand(79, 2, dtype=torch.float64)
        code = codes[:4].double()
        with torch.no_grad():
            folded = model.refine(code)
            error = (model(code, coordinates) - reference(folded, coordinates)).abs().max().item()
            assert error < 1e-12, error
        results.append({'name': name, 'seed': seed, 'checkpoint_sha256': sha256(checkpoint),
                        'max_rgb_error': error, 'fitted_code_mean_norm': code.norm(dim=1).mean().item(),
                        'folded_code_mean_norm': folded.norm(dim=1).mean().item()})
    out = {'all_passed': True, 'checks': results,
           'interpretation': 'For any fixed refined codes, the same-width grid refiner can be absorbed into new 336-value grids of the baseline. It changes optimization/regularization and the parameterization of new-code fitting, not the unconstrained per-image function family. The original code norm penalty and an eventual encoder distribution are not invariant under folding.'}
    write_json(root / 'verified/folding_check.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    run(parser.parse_args().root)
