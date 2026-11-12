"""GPU0-only synthetic pipeline/reload check, before source and protocol freeze."""
import argparse
import tempfile
import time
from pathlib import Path
import torch
from experiments.qvf_project_a.train import configure_runtime, write_json
from experiments.qvf_project_a.direct_qvf import render_direct
from experiments.qvf_project_a.token_study import fit_completion
from .spatial_models import build_model, model_specs, resources
from .spatial_study import ZTrainConfig, train_model, load_model


def run(output):
    device = configure_runtime('cuda:0', 19)
    records = []
    for name, _ in model_specs():
        configure_runtime('cuda:0', 19)
        model = build_model(name, 19).to(device)
        code = torch.randn(16, 336, device=device, requires_grad=True)
        xy = torch.rand(128, 2, device=device)
        model(code, xy).square().mean().backward()
        for key, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (name, key)
            assert parameter.grad.abs().sum() > 0, (name, key)
        assert torch.isfinite(code.grad).all() and code.grad.abs().sum() > 0
        del model, code
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            target = torch.rand(32, 3, 32, 32)
            cfg = ZTrainConfig(architecture=name, seed=19, n_train=32, steps=100, eval_every=100)
            result = train_model(cfg, target, list(range(32)), folder, device)
            model, codes, _ = load_model(folder / 'fit/checkpoints/last.pt', device)
            payload = torch.load(folder / 'final/train.pt', weights_only=False)
            torch.testing.assert_close(render_direct(model, codes, 16), payload['reconstructions'], atol=0, rtol=0)
            spec = {'observed_pixels': list(range(256)), 'unobserved_pixels': list(range(256, 1024)),
                    'steps': 2, 'code_init_std': .01, 'code_lr': .001, 'code_norm_weight': .0001}
            fit_completion(model, target, list(range(32)), spec, 19, folder / 'final', device)
            payload = torch.load(folder / 'final/completion.pt', weights_only=False)
            torch.testing.assert_close(render_direct(model, payload['codes'].to(device), 16), payload['reconstructions'], atol=0, rtol=0)
            records.append({'name': name, 'elapsed_seconds': time.perf_counter() - start,
                'training_seconds': result['elapsed_seconds'], 'peak_bytes': torch.cuda.max_memory_allocated(),
                'resources': resources(model), 'finite_nonzero_all_gradients': True,
                'exact_train_and_completion_reload': True, 'synthetic_only_no_weights_reused': True})
            del model, codes
        torch.cuda.empty_cache()
    write_json(output, {'device': 'physical GPU0', 'torch': torch.__version__, 'runs': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
