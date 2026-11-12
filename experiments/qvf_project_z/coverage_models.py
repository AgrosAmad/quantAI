"""Z8 model registry: unchanged Z6/Z7 implementations and audited vanilla QVF."""
from dataclasses import dataclass
from collections import Counter

import numpy as np
import torch
from torch import nn
import pennylane as qml

from experiments.qvf_repro.qvf_fast import QVFImage, positional_embedding, parity_decode
from .hires_models import NAMES as Z6, build_model as build_z6, resources as z6_resources
from .routing_models import NAMES as Z7, build_model as build_z7, resource_summary as z7_resources

NAMES = ('vanilla_qvf', *Z6, *Z7)


@dataclass(frozen=True)
class VanillaConfig:
    architecture: str = 'vanilla_qvf'
    image_size: int = 256
    frequencies: int = 9
    width: int = 128
    latent_dim: int = 64


class VanillaField(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.field = QVFImage({
            'L_embed': config.frequencies, 'latent_size': 64, 'hid_dim': config.width,
            'out_dim': 32, 'num_hidden_layers_energy': 3, 'skip_connection': False,
            'num_repetations': 5, 'use_identity_block': False, 'gaussian_std': .1,
            'output_channels': 3, 'use_parity': True,
        })
        self._cache = None

    def inputs(self, codes, coordinates):
        if codes.shape[1] != 64:
            raise ValueError('Vanilla QVF retains its original 64-value image code')
        position = positional_embedding(coordinates, self.cfg.frequencies)
        return torch.cat((codes[:, None].expand(-1, len(coordinates), -1),
                          position[None].expand(len(codes), -1, -1)), -1).flatten(0, 1)

    def trace(self, codes, coordinates):
        amplitudes = self.field.embedder(self.inputs(codes, coordinates))
        matrix = self.field.quantum.circuit_matrix() if self._cache is None else self._cache
        state = amplitudes @ matrix
        probs = state.square()
        z = probs @ self.field.quantum.signs.to(probs.dtype)
        rgb = parity_decode(z, 3).reshape(len(codes), len(coordinates), 3)
        return [amplitudes], [probs], rgb

    def forward(self, codes, coordinates):
        # Training calls the original audited QVFImage directly.
        if self._cache is None:
            return self.field(self.inputs(codes, coordinates)).reshape(len(codes), len(coordinates), 3)
        return self.trace(codes, coordinates)[-1]

    @torch.no_grad()
    def cache_quantum(self):
        self._cache = self.field.quantum.circuit_matrix()

    def clear_cache(self):
        self._cache = None


def build_model(name, seed=0, **config):
    config = dict(config)
    config.pop('architecture', None)
    if name in Z6:
        return build_z6(name, seed, **config)
    if name in Z7:
        return build_z7(name, seed, **config)
    if name != 'vanilla_qvf':
        raise ValueError(name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return VanillaField(VanillaConfig(**{k: config[k] for k in ('image_size', 'frequencies', 'width')}))


def trace(model, name, codes, coordinates):
    value = model.trace(codes, coordinates)
    if name in Z7:
        return value[0], value[1] if name == 'classical_attention16' else [value[2]], value[-1]
    return value


def readout(model, name, probabilities, coordinates, batch):
    if name in Z7:
        from .routing_models import readout_numpy
        return readout_numpy(model, probabilities, coordinates, batch)
    from .hires_gates import readout_numpy
    return readout_numpy(model, probabilities)


def vanilla_native(model, amplitudes):
    q = model.field.quantum
    @qml.qnode(qml.device('default.qubit', wires=5), interface='torch', diff_method='backprop')
    def circuit():
        qml.AmplitudeEmbedding(amplitudes, wires=range(5), normalize=True)
        for layer in range(5):
            for a in range(5):
                for b in range(5):
                    if a != b:
                        qml.CRY(q.cond[a, b, layer], wires=[a, b])
            for wire in range(5):
                qml.RY(q.weights[wire, layer], wires=wire)
        return qml.state()
    return circuit()


def vanilla_primitives(model, data):
    # Reuse the already-validated real amplitude preparation, then the original
    # all-to-all ordered CRY circuit (not a shared-filter substitute).
    from .spatial_gates import primitive_ops
    from types import SimpleNamespace
    prepared = primitive_ops('amplitude', np.asarray(data),
        SimpleNamespace(n_qubits=5, weights=torch.zeros(5, 5),
                        filters=torch.zeros(5, 1, 4), pairs=()))
    assert all(name == 'RY' and float(angle) == 0 for name, _, angle in prepared[-25:])
    preparation = prepared[:-25]
    q = model.field.quantum
    operations = list(preparation)
    for layer in range(5):
        for a in range(5):
            for b in range(5):
                if a != b:
                    angle = float(q.cond[a, b, layer].detach())
                    for op in qml.CRY(angle, wires=[a, b]).decomposition():
                        operations.append((op.name, tuple(int(w) for w in op.wires),
                                           np.asarray(op.data[0]) if op.data else None))
        operations.extend(('RY', (wire,), float(q.weights[wire, layer].detach())) for wire in range(5))
    return operations


def resources(model, name):
    if name in Z7:
        return z7_resources(model)
    if name in Z6:
        old = z6_resources(model)
        return {**old, 'stored_shared_parameters': sum(p.numel() for p in model.parameters()),
                'logical_circuits_per_coordinate': len(model.heads),
                'joint_qubits_per_circuit': model.heads[0].n_qubits}
    operations = vanilla_primitives(model, np.full((1, 32), 1 / np.sqrt(32)))
    total = sum(p.numel() for p in model.parameters())
    return {'stored_shared_parameters': total, 'trainable_shared_parameters': total,
            'classical_parameters': total - 150, 'stored_quantum_parameters': 150,
            'output_relevant_quantum_parameters': 125, 'image_code_entries': 64,
            'joint_qubits_per_circuit': 5, 'logical_circuits_per_coordinate': 1,
            'primitive_gates_per_coordinate': dict(Counter(x[0] for x in operations)),
            'original_QVFImage_used': True, 'hardware_execution': False}
