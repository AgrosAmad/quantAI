"""Full-code-table GPU0 preflight; synthetic data and discarded weights."""
import argparse
import tempfile
from pathlib import Path
import torch
from experiments.qvf_project_a.train import configure_runtime, write_json
from experiments.qvf_project_a.direct_qvf import render_direct
from .scale_study import ZTrainConfig, train_model, load_model


def run(output):
    device = configure_runtime('cuda:0', 7101)
    records = []
    for name in ('attention_chain_q5', 'attention_chain_q7', 'spatial_star_q7'):
        configure_runtime('cuda:0', 7101)
        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            images = torch.rand(4744, 3, 32, 32)
            cfg = ZTrainConfig(architecture=name, seed=7101, steps=2, eval_every=2)
            result = train_model(cfg, images, list(range(4744)), folder, device)
            model, codes, saved = load_model(folder / 'fit/checkpoints/last.pt', device)
            prediction = render_direct(model, codes, 16)
            target = torch.load(folder / 'final/train.pt', weights_only=False)
            torch.testing.assert_close(prediction, target['reconstructions'], atol=0, rtol=0)
            assert codes.shape == (4744, 336) and saved['step'] == 2
            assert sum(saved['visits']).item() == 32
            assert torch.isfinite(codes).all()
            records.append({'name': name, 'full_code_table_shape': list(codes.shape),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'exact_reload': True, 'training_seconds': result['elapsed_seconds'],
                'synthetic_only_no_weights_reused': True})
            del model, codes, prediction, target, saved, images
        torch.cuda.empty_cache()
    write_json(output, {'physical_gpu': 0, 'torch': torch.__version__, 'runs': records})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
