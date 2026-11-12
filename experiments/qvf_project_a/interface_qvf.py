"""V11 spatial patches, coordinate-driven angle loading, and quantum feature grids.

The original QVF and V9/V10 models are never edited. Five qubits and 336 fitted
image-code values are fixed. A grid decoder has a native 32x32 RGB output rather
than the coordinate-field semantics of the other variants.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from experiments.qvf_repro.qvf_fast import MLPEmbedding, parity_decode, positional_embedding
from .conv_qvf import local_features
from .readout_probe import diagonal_observables
from .token_qvf import TokenConfig, TokenQVFDecoder, ScaleQuantumLayer, batched_ry, build_token_decoder


@dataclass
class InterfaceConfig:
    architecture: str
    latent_dim: int = 336
    image_size: int = 32
    qubits: int = 5
    field_width: int = 128
    grid_sizes: tuple = (2, 4, 8)
    token_width: int = 4
    frequencies: int = 6


def patch_features(grid, coordinates):
    """Offset-major 3x3 features, replicate border, bilinear query at [0,1].

    Concatenated edge slices avoid nondeterministic CUDA padding gradients. This
    expands shared inputs, not the number of fitted values in the image code.
    """
    height, width = grid.shape[-2:]
    padded = torch.cat((grid[:, :, :1], grid, grid[:, :, -1:]), 2)
    padded = torch.cat((padded[:, :, :, :1], padded, padded[:, :, :, -1:]), 3)
    unfolded = torch.cat([padded[:, :, dy:dy + height, dx:dx + width]
                          for dy in range(3) for dx in range(3)], 1)
    return local_features(unfolded, coordinates)


def angle_state(angles, quantum=None):
    """Five data RY uploads, each followed by one original QVF processing block.

    Each of the 25 data angles can depend on features and explicit coordinates.
    With quantum=None all shared processing is removed; data RY gates remain.
    """
    if angles.ndim != 3 or angles.shape[1:] != (5, 5):
        raise ValueError("Expected [queries, five uploads, five wires]")
    state = angles.new_zeros(len(angles), 32)
    state[:, 0] = 1
    for stage in range(5):
        state = state.reshape(-1, 2, 2, 2, 2, 2)
        for wire in range(5):
            state = batched_ry(state, angles[:, stage, wire], wire)
        state = state.flatten(1)
        if quantum is not None:
            state = state @ quantum.block_matrix(stage, stage + 1)
    return state


class ConvImageDecoder(nn.Module):
    """Small shared decoder: 8x8x15 quantum/classical features -> 32x32 RGB."""
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(15, 32, 3, padding=1)
        self.residual = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(32, 32, 3, padding=1))
        self.middle = nn.Conv2d(32, 32, 3, padding=1)
        self.fine = nn.Conv2d(32, 16, 3, padding=1)
        self.rgb = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, features):
        value = F.silu(self.input(features))
        value = F.silu(value + self.residual(value))
        value = value.repeat_interleave(2, 2).repeat_interleave(2, 3)
        value = F.silu(self.middle(value))
        value = value.repeat_interleave(2, 2).repeat_interleave(2, 3)
        return self.rgb(F.silu(self.fine(value)))


def mlp_count(input_width, hidden_width, output_width):
    return (input_width + 1) * hidden_width + 2 * (hidden_width + 1) * hidden_width + (hidden_width + 1) * output_width


# Controls have slightly MORE shared capacity than their target, by the smallest
# integer hidden-width adjustment. Counts include actual stored circuit angles.
PATCH_AMP_PARAMETERS = 60 + mlp_count(134, 128, 32) + 150
PATCH_ANGLE_PARAMETERS = 60 + mlp_count(134, 128, 25) + 50 + 150
WIDE_ATTENTION_WIDTH = next(w for w in range(128, 256)
    if 1842 + mlp_count(38, w, 32) >= PATCH_AMP_PARAMETERS)
CLASSICAL_FIELD_WIDTH = next(w for w in range(128, 256)
    if 60 + mlp_count(134, w, 3) >= PATCH_ANGLE_PARAMETERS)
CLASSICAL_GRID_WIDTH = next(w for w in range(128, 256)
    if 60 + mlp_count(134, w, 15) >= PATCH_ANGLE_PARAMETERS)


class InterfaceDecoder(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        allowed = {"patch_amp", "attention_angle", "patch_angle", "patch_angle_identity",
                   "patch_classical", "quantum_grid_conv", "classical_grid_conv"}
        if architecture not in allowed:
            raise ValueError(architecture)
        width = CLASSICAL_FIELD_WIDTH if architecture == "patch_classical" else (
            CLASSICAL_GRID_WIDTH if architecture == "classical_grid_conv" else 128)
        self.cfg = InterfaceConfig(architecture=architecture, field_width=width)
        self.is_grid = architecture.endswith("grid_conv")
        self.is_classical = architecture in {"patch_classical", "classical_grid_conv"}
        self.is_amplitude = architecture == "patch_amp"
        self.is_attention = architecture == "attention_angle"
        self.values = nn.ModuleList(nn.Linear(4, 4) for _ in self.cfg.grid_sizes)
        if self.is_attention:
            self.queries = nn.ModuleList(nn.Linear(26, 16) for _ in self.cfg.grid_sizes)
            self.keys = nn.ModuleList(nn.Linear(6, 16) for _ in self.cfg.grid_sizes)
        for index, size in enumerate(self.cfg.grid_sizes):
            rows, cols = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
            self.register_buffer(f"positions_{index}", torch.stack((rows, cols), -1).flatten(0, 1), persistent=False)
        output_width = 32 if self.is_amplitude else (15 if self.is_grid else 3) if self.is_classical else 25
        self.amplitudes = MLPEmbedding(38 if self.is_attention else 134, width, output_width, 3,
                                       amplitude_output=self.is_amplitude)
        self.quantum = None if self.is_classical or architecture == "patch_angle_identity" else ScaleQuantumLayer(5, 5, False)
        if not self.is_amplitude and not self.is_classical:
            directions = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.], [.5, .5]])
            frequencies = 2. ** torch.arange(5)
            self.position_weights = nn.Parameter(.1 * torch.pi * frequencies[:, None, None] * directions[None])
        self.register_buffer("observables", diagonal_observables(5, torch.float32), persistent=False)
        if self.is_grid:
            self.image_decoder = ConvImageDecoder()
            self.register_buffer("grid_coordinates", getattr(self, "positions_2").clone(), persistent=False)

    @property
    def circuit(self):
        return self.quantum

    def retrieve(self, code, coordinates):
        features, offset = [], 0
        position = positional_embedding(coordinates, 6)
        for index, size in enumerate(self.cfg.grid_sizes):
            count = 4 * size * size
            grid = code[:, offset:offset + count].reshape(-1, 4, size, size)
            tokens = grid.flatten(2).transpose(1, 2)
            values = self.values[index](tokens)
            if self.is_attention:
                centres = getattr(self, f"positions_{index}").to(code.dtype)
                keys = self.keys[index](torch.cat((tokens, centres[None].expand(len(code), -1, -1)), -1))
                query = self.queries[index](position)
                score = torch.einsum("pd,bnd->bpn", query, keys) / 4.
                distance = ((coordinates[:, None] - centres[None]) * (size - 1)).square().sum(-1)
                selected = (score - 2. * distance).softmax(-1) @ values
            else:
                selected = patch_features(values.transpose(1, 2).reshape_as(grid), coordinates)
            features.append(selected)
            offset += count
        return features, position

    def query(self, code, coordinates, return_features=False):
        local, position = self.retrieve(code, coordinates)
        inputs = torch.cat((*local, position[None].expand(len(code), -1, -1)), -1)
        encoded = self.amplitudes(inputs.flatten(0, 1))
        if self.is_classical:
            value = encoded.tanh() if return_features else encoded
        else:
            if self.is_amplitude:
                state = encoded @ self.quantum.circuit_matrix()
            else:
                offsets = torch.einsum("pc,lqc->plq", coordinates, self.position_weights)
                angles = torch.pi * encoded.tanh().reshape(len(code), len(coordinates), 5, 5) + offsets[None]
                state = angle_state(angles.flatten(0, 1), self.quantum)
            measured = state.square() @ self.observables.to(state.dtype)
            value = measured if return_features else parity_decode(measured[:, :5], 3)
        return value.reshape(len(code), len(coordinates), -1)

    def forward(self, code, coordinates):
        if not self.is_grid:
            return self.query(code, coordinates)
        features = self.query(code, self.grid_coordinates.to(code.dtype), return_features=True)
        rgb = self.image_decoder(features.transpose(1, 2).reshape(len(code), 15, 8, 8))
        # This output is a native raster. Training/evaluation always use i/32,j/32.
        indices = (coordinates * 32).round().long()
        if not torch.allclose(coordinates * 32, indices.to(coordinates.dtype), atol=1e-5, rtol=0):
            raise ValueError("Convolutional RGB output supports native pixel queries only")
        if indices.min() < 0 or indices.max() >= 32:
            raise ValueError("Pixel index outside the native image")
        return rgb.flatten(2).transpose(1, 2)[:, indices[:, 0] * 32 + indices[:, 1]]


STAGES = {
    "separate": ("patch_amp", "attention_angle", "attention_amp", "attention_amp_wide"),
    "combined": ("patch_angle", "patch_angle_identity", "patch_classical"),
    "grid": ("quantum_grid_conv", "classical_grid_conv"),
}


def build_interface(architecture, seed):
    devices = [0] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        if architecture in {"attention_amp", "attention_amp_wide"}:
            return build_token_decoder(TokenConfig(field_width=128 if architecture == "attention_amp" else WIDE_ATTENTION_WIDTH), seed)
        torch.manual_seed(seed + 100)
        model = InterfaceDecoder(architecture)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
                torch.manual_seed(seed + offset)
                module.reset_parameters()
        if model.quantum is not None:
            torch.manual_seed(seed + 1001)
            with torch.no_grad():
                model.quantum.weights.normal_(0, .1)
                model.quantum.cond.normal_(0, .1)
        return model


def resources(model, architecture):
    grid = architecture.endswith("grid_conv")
    classical = architecture in {"patch_classical", "classical_grid_conv"}
    angle = "angle" in architecture or architecture == "quantum_grid_conv"
    processing = not classical and architecture != "patch_angle_identity"
    return {"shared_parameters": sum(p.numel() for p in model.parameters()),
        "image_code_entries": 336, "qubits": 0 if classical else 5,
        "native_grid_output": grid, "quantum_queries_per_image_training_update": 0 if classical else 64 if grid else 128,
        "quantum_queries_per_full_image": 0 if classical else 64 if grid else 1024,
        "effective_trainable_processing_angles": 125 if processing else 0,
        "controlled_ry_per_query": 100 if processing else 0,
        "processing_ry_per_query": 25 if processing else 0,
        "data_ry_per_query": 25 if angle and not classical else 0,
        "arbitrary_amplitude_preparation": not angle and not classical,
        "readout": "15 Z/ZZ features then shared conv" if grid and not classical else "classical" if classical else "original grouped Z RGB"}
