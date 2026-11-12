"""Z4: explicit spatial slots, work-qubit scaling, and matched topology controls.

Original Z1-Z3 sources are frozen. All readouts use the same first-five-wire
RGB rule. In slot models only wires 0..4 receive data, even at seven qubits.
"""
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from experiments.qvf_project_a.token_qvf import TokenConfig, TokenQVFDecoder, build_token_decoder, batched_ry
from experiments.qvf_project_a.conv_qvf import local_features
from experiments.qvf_repro.qvf_fast import FastQuantumLayer, MLPEmbedding, positional_embedding, parity_decode, _apply_ry, _apply_cry
from .models import build_model as build_z3

OFFSETS = ((0., -1.), (-1., 0.), (0., 0.), (1., 0.), (0., 1.))
FAMILIES = ('attention_chain', 'attention_star', 'spatial_star', 'center_star', 'attention_chain_wide')


@dataclass
class SpatialConfig:
    family: str = 'attention_chain'
    qubits: int = 5
    field_width: int = 128

    def validate(self):
        if self.family not in FAMILIES or self.qubits not in (5, 7):
            raise ValueError('Unknown family/qubit count')
        if self.field_width != (148 if self.family == 'attention_chain_wide' else 128):
            raise ValueError('Widths are fixed by this controlled comparison')
        if self.family == 'attention_chain_wide' and self.qubits != 5:
            raise ValueError('Wide control is the five-qubit capacity match')


class FilterCircuit(FastQuantumLayer):
    def __init__(self, qubits, topology):
        super().__init__(qubits, 5, False)
        del self.cond
        self.topology = topology
        self.pairs = tuple((i, i + 1) for parity in (0, 1) for i in range(parity, qubits - 1, 2)) if topology == 'chain' else tuple((2, i) for i in range(qubits) if i != 2)
        self.filters = nn.Parameter(torch.empty(5, 1, 4))

    def block_matrix(self, start=0, stop=5):
        n, d = self.n_qubits, 2 ** self.n_qubits
        state = torch.eye(d, dtype=torch.float64).reshape(d, *([2] * n))
        weights = self.weights.to('cpu', torch.float64)
        filters = self.filters.to('cpu', torch.float64)
        for layer in range(start, stop):
            x = filters[layer, 0]
            for a, b in self.pairs:
                state = _apply_ry(state, x[0], a + 1)
                state = _apply_ry(state, x[1], b + 1)
                state = _apply_cry(state, x[2], a + 1, b + 1)
                state = _apply_cry(state, x[3], b + 1, a + 1)
            for wire in range(n):
                state = _apply_ry(state, weights[wire, layer], wire + 1)
        return state.reshape(d, d).to(self.weights.device, self.weights.dtype)

    def circuit_matrix(self):
        return self.block_matrix()

    def slot_state(self, angles):
        n = self.n_qubits
        if angles.shape[1:] != (5, 5):
            raise ValueError('Five stages, five spatial input slots in both qubit conditions')
        state = angles.new_zeros(len(angles), *([2] * n))
        state.reshape(len(angles), -1)[:, 0] = 1.
        # Fixed |+> preparation of the data wires; two extra work wires stay |0>.
        for wire in range(5):
            state = batched_ry(state, angles.new_full((len(angles),), torch.pi / 2), wire)
        for layer in range(5):
            for wire in range(5):
                state = batched_ry(state, angles[:, layer, wire], wire)
            state = (state.flatten(1) @ self.block_matrix(layer, layer + 1)).reshape(-1, *([2] * n))
        return state.flatten(1)


class AmplitudeField(TokenQVFDecoder):
    def __init__(self, cfg):
        super().__init__(TokenConfig(qubits=cfg.qubits, field_width=cfg.field_width))
        self.spatial_cfg = cfg
        self.quantum = FilterCircuit(cfg.qubits, 'star' if cfg.family == 'attention_star' else 'chain')

    def encoded(self, code, coordinates):
        local, position = self.retrieve(code, coordinates)
        inputs = torch.cat((*local, position[None].expand(len(code), -1, -1)), -1).flatten(0, 1)
        return self.amplitudes(inputs)

    def probabilities(self, code, coordinates):
        amplitudes = self.encoded(code, coordinates)
        return (amplitudes @ self.quantum.circuit_matrix()).square().reshape(len(code), len(coordinates), -1)

    def forward(self, code, coordinates):
        # Keep exactly Z0+Z3, Z1+Z4, Z2 at five AND seven qubits.
        amplitudes = self.encoded(code, coordinates)
        z = self.quantum(amplitudes)
        return parity_decode(z[:, :5], 3).reshape(len(code), len(coordinates), 3)


class SpatialField(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spatial_cfg = cfg
        self.cfg = TokenConfig(qubits=cfg.qubits)
        self.values = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
        # Independent per-slot evaluation, shared weights; no cross-slot mixing here.
        self.slot_encoder = MLPEmbedding(40, 128, 5, 3, amplitude_output=False)
        self.quantum = FilterCircuit(cfg.qubits, 'star')
        self.register_buffer('offsets', torch.tensor(OFFSETS), persistent=False)

    @property
    def circuit(self):
        return self.quantum

    def slot_features(self, code, coordinates):
        if code.shape[1:] != (336,) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError('Expected 336-value codes and row/column queries')
        gathered, start = [], 0
        for index, size in enumerate(self.cfg.grid_sizes):
            count = 4 * size * size
            grid = code[:, start:start + count].reshape(-1, 4, size, size)
            values = self.values[index](grid.flatten(2).transpose(1, 2)).transpose(1, 2).reshape_as(grid)
            offsets = self.offsets.to(code.dtype)
            if self.spatial_cfg.family == 'center_star':
                offsets = torch.zeros_like(offsets)
            positions = coordinates[:, None] + offsets[None] / (size - 1)
            sampled = local_features(values, positions.flatten(0, 1)).reshape(len(code), len(coordinates), 5, 4)
            gathered.append(sampled)
            start += count
        return torch.cat(gathered, -1)

    def encoded(self, code, coordinates):
        features = self.slot_features(code, coordinates)
        position = positional_embedding(coordinates, 6)[None, :, None].expand(len(code), -1, 5, -1)
        roles = self.offsets.to(code.dtype)[None, None].expand(len(code), len(coordinates), -1, -1)
        inputs = torch.cat((features, position, roles), -1)
        angles = torch.pi * self.slot_encoder(inputs.flatten(0, 2)).tanh()
        return angles.reshape(len(code), len(coordinates), 5, 5).transpose(-2, -1).flatten(0, 1)

    def probabilities(self, code, coordinates):
        state = self.quantum.slot_state(self.encoded(code, coordinates))
        return state.square().reshape(len(code), len(coordinates), -1)

    def forward(self, code, coordinates):
        state = self.quantum.slot_state(self.encoded(code, coordinates))
        z = state.square() @ self.quantum.signs.to(state.dtype)
        return parity_decode(z[:, :5], 3).reshape(len(code), len(coordinates), 3)


def build_spatial(cfg, seed=0):
    cfg.validate()
    devices = [0] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        old = build_z3('attention_local_shared', seed)
        if cfg.family.startswith('attention'):
            ref = build_token_decoder(TokenConfig(qubits=cfg.qubits, field_width=cfg.field_width), seed)
            model = AmplitudeField(cfg)
            payload = {k: v for k, v in ref.state_dict().items() if not k.startswith('quantum.')}
            missing = model.load_state_dict(payload, strict=False)
            assert not missing.unexpected_keys and all(k.startswith('quantum.') for k in missing.missing_keys)
        else:
            model = SpatialField(cfg)
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Linear):
                    torch.manual_seed(seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little'))
                    layer.reset_parameters()
                    if name == 'slot_encoder.layers.3':
                        nn.init.normal_(layer.weight, std=.01)
                        nn.init.zeros_(layer.bias)
        with torch.no_grad():
            model.quantum.weights[:5].copy_(old.quantum.weights)
            model.quantum.filters.copy_(old.quantum.filters)
            if cfg.qubits == 7:
                torch.manual_seed(seed + 17002)
                model.quantum.weights[5:].normal_(0, .1)
        return model


def model_specs():
    return [(f'{family}_q{q}', SpatialConfig(family=family, qubits=q))
            for q in (5, 7) for family in FAMILIES[:-1]] + [
                ('attention_chain_wide_q5', SpatialConfig(family='attention_chain_wide', qubits=5, field_width=148))]


def build_model(name, seed=0):
    return build_spatial(dict(model_specs())[name], seed)


def resources(model, name=None):
    cfg, circuit = model.spatial_cfg, model.quantum
    slot = isinstance(model, SpatialField)
    q, pairs = cfg.qubits, len(circuit.pairs)
    cry, ry = 10 * pairs, 10 * pairs + 5 * q
    prep_ry, prep_cnot = (5, 0) if slot else (2 ** q - 1, 2 ** q - 2)
    return {'shared_parameters': sum(p.numel() for p in model.parameters()),
        'classical_parameters': sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in circuit.parameters()),
        'effective_quantum_parameters': sum(p.numel() for p in circuit.parameters()),
        'image_code_entries': 336, 'qubits': q, 'topology': circuit.topology,
        'pairs': [list(p) for p in circuit.pairs], 'spatial_data_wires': 5 if slot else None,
        'work_qubits_without_data_upload': q - 5 if slot else 0,
        'readout_wires': list(range(5)), 'readout': 'Z0+Z3, Z1+Z4, Z2 for both qubit counts',
        'controlled_ry_per_query': cry, 'processing_ry_per_query': ry,
        'data_ry_per_query': 25 if slot else 0, 'preparation_ry_per_query': prep_ry,
        'preparation_cnot_per_query': prep_cnot,
        'primitive_ry_per_query': prep_ry + ry + 2 * cry + (25 if slot else 0),
        'primitive_cnot_per_query': prep_cnot + 2 * cry,
        'quantum_queries_per_image_training_update': 128, 'quantum_queries_per_full_image': 1024,
        'classical_mlp_evaluations_per_query': 5 if slot else 1,
        'arbitrary_amplitude_preparation': not slot, 'native_grid_output': False}
