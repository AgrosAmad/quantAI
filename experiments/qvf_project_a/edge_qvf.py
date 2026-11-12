"""V10: all-scale initial amplitudes plus optional later feature rotations.

V9 sources/checkpoints stay unchanged. At zero added angles, this staged field
represents the same function as the attention-only decoder.
"""

import torch
from torch import nn

from experiments.qvf_repro.qvf_fast import parity_decode
from .token_qvf import TokenConfig, TokenQVFDecoder, build_token_decoder


class AllScaleStagedQVF(TokenQVFDecoder):
    def __init__(self):
        super().__init__(TokenConfig())  # all three summaries enter amplitudes
        self.uploads = nn.ModuleList(nn.Linear(self.cfg.token_width, self.cfg.qubits) for _ in range(2))
        for layer in self.uploads:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, code, coordinates):
        features, position = self.retrieve(code, coordinates)
        position = position[None].expand(len(code), -1, -1)
        amplitudes = self.amplitudes(torch.cat((*features, position), -1).flatten(0, 1))
        angles = torch.stack([torch.pi * torch.tanh(layer(feature))
            for layer, feature in zip(self.uploads, features[1:])], dim=2).flatten(0, 1)
        rgb = parity_decode(self.quantum(amplitudes, angles), 3)
        return rgb.reshape(len(code), len(coordinates), 3)


def build_edge_decoder(architecture, seed):
    if architecture not in {"attention", "staged_all"}:
        raise ValueError("Expected attention or staged_all")
    # Existing initializer seeds CUDA as well as CPU. Preserve physical GPU0's
    # generator when already initialized, without touching reserved GPU1.
    devices = [0] if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        baseline = build_token_decoder(TokenConfig(), seed)
        if architecture == "attention":
            return baseline
        model = AllScaleStagedQVF()
        result = model.load_state_dict(baseline.state_dict(), strict=False)
        assert set(result.missing_keys) == {f"uploads.{i}.{p}" for i in range(2) for p in ("weight", "bias")}
        assert not result.unexpected_keys
        return model


def neighbour_pairs(size=32):
    """All rightward edges followed by all downward edges; no row wrapping."""
    if size < 2:
        raise ValueError("At least two pixels per axis are required")
    indices = torch.arange(size * size).reshape(size, size)
    horizontal = torch.stack((indices[:, :-1].flatten(), indices[:, 1:].flatten()), -1)
    vertical = torch.stack((indices[:-1].flatten(), indices[1:].flatten()), -1)
    return horizontal, vertical


def rgb_importance_weights(size=32):
    """Uniform edge endpoints over-sample interior pixels; correct their RGB loss.

    For a square image with equally many horizontal/vertical edges, an endpoint's
    probability is degree / total_degree. The returned inverse-probability weight
    gives uniform full-image pixel MSE in expectation. Do not renormalize each
    sampled batch by the sum of weights.
    """
    h, v = neighbour_pairs(size)
    degree = torch.bincount(torch.cat((h, v)).flatten(), minlength=size * size).float()
    return degree.mean() / degree


def sample_pairs(generator, per_orientation=32, size=32):
    if per_orientation < 1:
        raise ValueError("Sample at least one pair per orientation")
    h, v = neighbour_pairs(size)
    horizontal = h[torch.randint(len(h), (per_orientation,), generator=generator)]
    vertical = v[torch.randint(len(v), (per_orientation,), generator=generator)]
    return torch.cat((horizontal, vertical)).flatten()


def edge_objective(prediction, target, pixel_weights, selected_codes, edge_weight, code_weight):
    """Each consecutive pair of query entries is one actual image edge.

    Equal horizontal/vertical counts make the edge term an unbiased estimate of
    the existing full-image finite-difference metric (no division by spacing).
    """
    if prediction.shape != target.shape or prediction.shape[1] % 2 or pixel_weights.shape != prediction.shape[1:2]:
        raise ValueError("Expected matching RGB tensors and one weight per paired pixel")
    error = prediction - target
    pixel = (error.square().mean(-1) * pixel_weights).mean()
    pairs = error.reshape(len(error), -1, 2, error.shape[-1])
    edge = (pairs[:, :, 1] - pairs[:, :, 0]).square().mean()
    penalty = code_weight * selected_codes.norm(dim=1).mean()
    return pixel + edge_weight * edge + penalty, pixel, edge, penalty


def masked_gradient_metrics(prediction, target, pixels):
    """Score only edges with BOTH endpoints in the supplied pixel set."""
    size = prediction.shape[-1]
    mask = torch.zeros(size * size, dtype=torch.bool)
    mask[torch.as_tensor(pixels).cpu()] = True
    error = (prediction.cpu().double() - target.cpu().double()).flatten(2)
    per_direction, counts = [], []
    for pairs in neighbour_pairs(size):
        pairs = pairs[mask[pairs].all(-1)]
        if not len(pairs):
            raise ValueError("No eligible edges in an orientation")
        per_direction.append((error[:, :, pairs[:, 1]] - error[:, :, pairs[:, 0]]).square().mean((1, 2)))
        counts.append(len(pairs))
    per_image = .5 * (per_direction[0] + per_direction[1])
    return {"gradient_mse": per_image.mean().item(), "per_image_gradient_mse": per_image.tolist(),
            "horizontal_edges": counts[0], "vertical_edges": counts[1], "both_endpoints_in_mask": True}
