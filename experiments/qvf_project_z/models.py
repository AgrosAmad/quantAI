"""Controlled latent-grid convolutions and local shared quantum processing.

Original QVF/A sources are imported, never modified. Qubit locality is not image
locality under amplitude encoding; the local circuit is QCNN-inspired without
pooling, not a spatial quantum image convolution.
"""
import hashlib
import torch
from torch import nn
from experiments.qvf_project_a.token_qvf import TokenConfig, TokenQVFDecoder, build_token_decoder
from experiments.qvf_repro.qvf_fast import FastQuantumLayer, _apply_ry, _apply_cry


class ResidualGrid(nn.Module):
    def __init__(self, spatial):
        super().__init__()
        # 1172 versus 1228 parameters/scale: pointwise control is slightly larger.
        kernel, hidden = (3, 16) if spatial else (1, 136)
        self.first = nn.Conv2d(4, hidden, kernel, padding=kernel // 2)
        self.last = nn.Conv2d(hidden, 4, kernel, padding=kernel // 2)

    def forward(self, grid):
        return grid + self.last(torch.nn.functional.silu(self.first(grid)))


class LocalQuantum(FastQuantumLayer):
    """Five brickwork sweeps; shared or edge-specific two-qubit filters.

Each pair: RY on each wire, then CRY in each direction. Even then odd edges;
one shared four-angle filter per sweep or four independent filters. The final
five per-wire RY angles per sweep are retained from the original QVF.
"""
    pairs = ((0, 1), (2, 3), (1, 2), (3, 4))

    def __init__(self, shared):
        super().__init__(5, 5, False)
        del self.cond
        self.shared = shared
        self.filters = nn.Parameter(torch.empty(5, 1 if shared else 4, 4))

    def _run_gates(self, state, weights=None, cond=None):
        weights = self.weights if weights is None else weights
        filters = self.filters if cond is None else cond
        for layer in range(5):
            for index, (a, b) in enumerate(self.pairs):
                x = filters[layer, 0 if self.shared else index]
                state = _apply_ry(state, x[0], a + 1)
                state = _apply_ry(state, x[1], b + 1)
                state = _apply_cry(state, x[2], a + 1, b + 1)
                state = _apply_cry(state, x[3], b + 1, a + 1)
            for wire in range(5):
                state = _apply_ry(state, weights[wire, layer], wire + 1)
        return state

    def circuit_matrix(self):
        basis = torch.eye(32, dtype=torch.float64).reshape(32, 2, 2, 2, 2, 2)
        matrix = self._run_gates(basis, self.weights.to('cpu', torch.float64),
                                 self.filters.to('cpu', torch.float64)).reshape(32, 32)
        return matrix.to(self.weights.device, self.weights.dtype)


class GridQVF(TokenQVFDecoder):
    def __init__(self, spatial):
        super().__init__(TokenConfig())
        self.refiners = nn.ModuleList(ResidualGrid(spatial) for _ in range(3))

    def refine(self, code):
        grids, offset = [], 0
        for layer, size in zip(self.refiners, self.cfg.grid_sizes):
            count = 4 * size * size
            grids.append(layer(code[:, offset:offset + count].reshape(-1, 4, size, size)).flatten(1))
            offset += count
        return torch.cat(grids, -1)

    def retrieve(self, code, coordinates):
        return super().retrieve(self.refine(code), coordinates)


LATENT = ('attention_amp', 'latent_conv', 'latent_pointwise')
ALL = (*LATENT, 'attention_local_shared', 'attention_local_untied',
       'conv_local_shared', 'conv_local_untied')


def build_model(architecture, seed=0):
    if architecture not in ALL:
        raise ValueError(architecture)
    devices = [0] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        reference = build_token_decoder(TokenConfig(), seed)
        use_conv = architecture == 'latent_conv' or architecture.startswith('conv_')
        refine = use_conv or architecture == 'latent_pointwise'
        if refine:
            model = GridQVF(use_conv)
            missing = model.load_state_dict(reference.state_dict(), strict=False)
            assert not missing.unexpected_keys and all(k.startswith('refiners.') for k in missing.missing_keys)
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):
                    torch.manual_seed(seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little'))
                    layer.reset_parameters()
                    if name.endswith('.last'):
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
        else:
            model = reference
        if 'local_' in architecture:
            quantum = LocalQuantum(architecture.endswith('_shared'))
            with torch.no_grad():
                quantum.weights.copy_(reference.quantum.weights)
                torch.manual_seed(seed + 17001)
                common = torch.randn(5, 1, 4) * .1
                quantum.filters.copy_(common.expand_as(quantum.filters))
            model.quantum = quantum
        return model


def resources(model, architecture):
    local = isinstance(model.quantum, LocalQuantum)
    return {'shared_parameters': sum(p.numel() for p in model.parameters()),
            'image_code_entries': 336, 'qubits': 5, 'native_grid_output': False,
            'quantum_queries_per_image_training_update': 128,
            'quantum_queries_per_full_image': 1024,
            'controlled_ry_per_query': 40 if local else 100,
            'processing_ry_per_query': 65 if local else 25,
            'effective_trainable_processing_angles': (45 if model.quantum.shared else 105) if local else 125,
            'arbitrary_amplitude_preparation': True, 'readout': 'original grouped Z RGB',
            'latent_refinement_parameters': sum(p.numel() for p in model.refiners.parameters()) if hasattr(model, 'refiners') else 0,
            'quantum_filter_sharing': model.quantum.shared if local else None}
