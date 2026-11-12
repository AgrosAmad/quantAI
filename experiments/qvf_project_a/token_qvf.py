"""Positioned image tokens and scale-conditioned QVF; vanilla QVF stays separate."""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from experiments.qvf_repro.qvf_fast import (FastQuantumLayer, MLPEmbedding,
    _apply_cry, _apply_ry, parity_decode, positional_embedding)
from .conv_qvf import local_features
from .models import CoordinateDecoder


@dataclass
class TokenConfig:
    retrieval: str = "attention"
    encoding: str = "single"
    qubits: int = 5
    latent_dim: int = 336
    grid_sizes: tuple = (2, 4, 8)
    token_width: int = 4
    attention_width: int = 16
    position_strength: float = 2.0
    field_width: int = 128
    field_layers: int = 3
    frequencies: int = 6
    circuit_depth: int = 5
    image_size: int = 32
    decoder: str = "qvf"
    readout: str = "vanilla"
    mode: str = "ae"

    def __post_init__(self):
        self.grid_sizes = tuple(self.grid_sizes)
        if self.retrieval not in {"global", "bilinear", "attention"}:
            raise ValueError("Unknown token retrieval")
        if self.encoding not in {"single", "staged"}:
            raise ValueError("Unknown quantum encoding")
        if self.qubits not in {5, 7} or self.circuit_depth != 5:
            raise ValueError("The token pilot supports five or seven qubits and five repetitions")
        if self.retrieval == "global" and self.encoding != "single":
            raise ValueError("Global-code controls use the original single upload")
        if len(self.grid_sizes) != 3 or any(s < 2 for s in self.grid_sizes):
            raise ValueError("Use three spatial grids with at least two positions per axis")
        if min(self.latent_dim, self.token_width, self.attention_width, self.field_width,
               self.field_layers) < 1 or self.frequencies < 1 or self.position_strength < 0:
            raise ValueError("Invalid widths, frequencies or position strength")
        if self.retrieval != "global" and self.latent_dim != sum(s * s for s in self.grid_sizes) * self.token_width:
            raise ValueError("Token storage must equal the declared per-image latent budget")
        if (self.image_size, self.decoder, self.readout, self.mode) != (32, "qvf", "vanilla", "ae"):
            raise ValueError("This study uses native RGB and deterministic direct code fitting")


def batched_ry(state, angles, wire):
    """Different RY angles per state, [batch, 2, ..., 2], wire zero is MSB."""
    a, b = state.unbind(wire + 1)
    angle = angles.reshape(-1, *([1] * (a.ndim - 1))) / 2
    c, s = angle.cos(), angle.sin()
    return torch.stack((c * a - s * b, s * a + c * b), dim=wire + 1)


class ScaleQuantumLayer(FastQuantumLayer):
    """Same QVF gate order/parameters, with optional feature uploads between blocks.

    Construct fixed matrices on CPU in double precision also at seven qubits.
    The three blocks together contain exactly the original five repetitions.
    Input-dependent RY gates operate on the existing state, without a reset.
    """

    cpu_matrix_max_qubits = 7

    def block_matrix(self, start, stop):
        n = self.n_qubits
        w, c = self.weights.to("cpu", torch.float64), self.cond.to("cpu", torch.float64)
        state = torch.eye(2 ** n, dtype=torch.float64).reshape(2 ** n, *([2] * n))
        for r in range(start, stop):
            for control in range(n):
                for target in range(n):
                    if control != target:
                        state = _apply_cry(state, c[control, target, r], control + 1, target + 1)
            for wire in range(n):
                state = _apply_ry(state, w[wire, r], wire + 1)
        return state.reshape(2 ** n, 2 ** n).to(self.weights.device, self.weights.dtype)

    def forward(self, amplitudes, uploads=None):
        if uploads is None:
            return super().forward(amplitudes)
        if uploads.shape != (len(amplitudes), 2, self.n_qubits):
            raise ValueError("Expected two feature uploads per state")
        state = amplitudes
        for stage, (start, stop) in enumerate(((0, 2), (2, 4), (4, 5))):
            if stage:
                state = state.reshape(-1, *([2] * self.n_qubits))
                for wire in range(self.n_qubits):
                    state = batched_ry(state, uploads[:, stage - 1, wire], wire)
                state = state.flatten(1)
            state = state @ self.block_matrix(start, stop)
        return state.square() @ self.signs.to(state.dtype)


class TokenQVFDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.retrieval == "global":
            raise ValueError("Use CoordinateDecoder for the global-code control")
        width, pos_width = cfg.token_width, 2 + 4 * cfg.frequencies
        # Both retrieval arms use the identical value projection.
        self.values = nn.ModuleList(nn.Linear(width, width) for _ in cfg.grid_sizes)
        if cfg.retrieval == "attention":
            self.queries = nn.ModuleList(nn.Linear(pos_width, cfg.attention_width) for _ in cfg.grid_sizes)
            self.keys = nn.ModuleList(nn.Linear(width + 2, cfg.attention_width) for _ in cfg.grid_sizes)
        for index, size in enumerate(cfg.grid_sizes):
            y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
            self.register_buffer(f"positions_{index}", torch.stack((y, x), -1).reshape(-1, 2), persistent=False)
        input_width = width * (3 if cfg.encoding == "single" else 1) + pos_width
        self.amplitudes = MLPEmbedding(input_width, cfg.field_width, 2 ** cfg.qubits, cfg.field_layers)
        if cfg.encoding == "staged":
            self.uploads = nn.ModuleList(nn.Linear(width, cfg.qubits) for _ in range(2))
        self.quantum = ScaleQuantumLayer(cfg.qubits, cfg.circuit_depth, False)

    @property
    def circuit(self):
        return self.quantum

    def retrieve(self, code, coordinates):
        if code.ndim != 2 or code.shape[1] != self.cfg.latent_dim:
            raise ValueError("Expected the complete per-image token code")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Expected [pixels, row/column] coordinates")
        features, offset = [], 0
        position = positional_embedding(coordinates, self.cfg.frequencies)
        for index, size in enumerate(self.cfg.grid_sizes):
            count = size * size * self.cfg.token_width
            # Channel-major storage agrees exactly with local_features.
            grid = code[:, offset:offset + count].reshape(-1, self.cfg.token_width, size, size)
            tokens = grid.flatten(2).transpose(1, 2)
            values = self.values[index](tokens)
            if self.cfg.retrieval == "bilinear":
                sampled = local_features(values.transpose(1, 2).reshape_as(grid), coordinates)
            else:
                centres = getattr(self, f"positions_{index}").to(code.dtype)
                keys = self.keys[index](torch.cat((tokens, centres[None].expand(len(code), -1, -1)), -1))
                queries = self.queries[index](position)
                scores = torch.einsum("pd,bnd->bpn", queries, keys) / self.cfg.attention_width ** 0.5
                distance = ((coordinates[:, None] - centres[None]) * (size - 1)).square().sum(-1)
                weights = (scores - self.cfg.position_strength * distance).softmax(-1)
                sampled = weights @ values
            features.append(sampled)
            offset += count
        return features, position

    def forward(self, code, coordinates):
        features, position = self.retrieve(code, coordinates)
        position = position[None].expand(len(code), -1, -1)
        initial = features if self.cfg.encoding == "single" else features[:1]
        inputs = torch.cat((*initial, position), -1).flatten(0, 1)
        amplitudes = self.amplitudes(inputs)
        angles = None
        if self.cfg.encoding == "staged":
            # Smooth bounded angles; no replacement state preparation.
            angles = torch.stack([torch.pi * torch.tanh(layer(feature))
                for layer, feature in zip(self.uploads, features[1:])], dim=2).flatten(0, 1)
        rgb = parity_decode(self.quantum(amplitudes, angles), 3)
        return rgb.reshape(len(code), len(coordinates), 3)


def build_token_decoder(cfg, seed=0):
    """Independent component seeds preserve common initial weights across arms."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 100)
        model = CoordinateDecoder(cfg) if cfg.retrieval == "global" else TokenQVFDecoder(cfg)
        if cfg.retrieval == "global":
            model.circuit.cpu_matrix_max_qubits = 7  # numerical implementation only
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
                torch.manual_seed(seed + offset)
                module.reset_parameters()
                if name.startswith("uploads."):
                    # Start the added data rotations close to identity rather
                    # than making random scale-dependent rotations of order pi.
                    nn.init.normal_(module.weight, std=0.01)
                    nn.init.zeros_(module.bias)
        torch.manual_seed(seed + 1001)
        with torch.no_grad():
            model.circuit.weights.normal_(0, 0.1)
            model.circuit.cond.normal_(0, 0.1)
    return model
