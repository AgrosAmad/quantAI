"""Check changed Z5 orchestration while preserving the frozen Z4 computation."""
import tempfile
import unittest
from pathlib import Path
import torch
from experiments.qvf_project_a.train import configure_runtime
from experiments.qvf_project_a.verify_edge_study import expected_stream
from experiments.qvf_project_a.direct_qvf import render_direct
from experiments.qvf_project_z.spatial_models import build_model
from experiments.qvf_project_z import spatial_study as old
from experiments.qvf_project_z import scale_study as new


class ScaleTests(unittest.TestCase):
    def setUp(self):
        configure_runtime('cpu', 71)

    def test_exact_training_and_partial_batch_stream(self):
        for name in ('attention_chain_q5', 'spatial_star_q7'):
            with tempfile.TemporaryDirectory() as tmp:
                folders = [Path(tmp) / label for label in ('old', 'new')]
                for folder in folders:
                    folder.mkdir()
                targets = torch.rand(18, 3, 32, 32)
                values = dict(architecture=name, n_train=18, batch_size=8, steps=5,
                    eval_every=5, pairs_per_orientation=2, seed=4)
                outputs = []
                for module, folder in zip((old, new), folders):
                    cfg = module.ZTrainConfig(**values)
                    module.train_model(cfg, targets, list(range(18)), folder, torch.device('cpu'))
                    model, codes, payload = module.load_model(folder / 'fit/checkpoints/last.pt', torch.device('cpu'))
                    outputs.append(payload)
                    saved = torch.load(folder / 'final/train.pt', weights_only=False)
                    torch.testing.assert_close(render_direct(model, codes, 8), saved['reconstructions'], atol=0, rtol=0)
                for key in ('codes', 'visits'):
                    torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
                for key, value in outputs[0]['decoder_state'].items():
                    torch.testing.assert_close(value, outputs[1]['decoder_state'][key], atol=0, rtol=0)
                digest, visits, rng = expected_stream(values)
                self.assertEqual(digest, outputs[1]['stream_sha256'])
                self.assertEqual(digest, outputs[0]['stream_sha256'])
                torch.testing.assert_close(visits, outputs[1]['visits'], atol=0, rtol=0)
                for key, state in rng.items():
                    torch.testing.assert_close(state, outputs[1]['rng'][key], atol=0, rtol=0)

    def test_corrected_resource_labels_preserve_counts(self):
        for q in (5, 7):
            for family in ('attention_chain', 'spatial_star'):
                model = build_model(f'{family}_q{q}', 1)
                old_counts, counts = old.resources(model), new.resources(model)
                self.assertNotIn('effective_quantum_parameters', counts)
                self.assertEqual(counts['stored_quantum_parameters'], old_counts['effective_quantum_parameters'])
                self.assertEqual(counts['output_relevant_quantum_parameter_entries'], 45 if q == 5 else 53)
                for key in ('primitive_ry_per_query', 'primitive_cnot_per_query', 'shared_parameters', 'classical_parameters'):
                    self.assertEqual(counts[key], old_counts[key])


if __name__ == '__main__':
    unittest.main()
