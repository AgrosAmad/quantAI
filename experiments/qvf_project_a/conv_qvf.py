"""Local-feature QVF decoder; the original pointwise field remains separate.

A global code generates a spatial feature grid. At each query coordinate,
interpolated local features feed the original neural-amplitude QVF head.
"""

import torch
from torch import nn

from experiments.qvf_repro.qvf_fast import QVFImage, positional_embedding
from .models import ModelConfig


def local_features(features, coordinates):
    """Bilinear sampling, border padding, row/column coordinates in [0, 1].

    Endpoint coordinates align with endpoint feature centres. Explicit separable
    hat weights make the coordinate convention easy to audit. This is a
    correctness prototype, not an optimized sparse sampler.
    """
    if features.ndim != 4 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Expected [B,C,H,W] features and [P,2] coordinates")
    height, width = features.shape[-2:]
    query = coordinates.clamp(0, 1)
    rows = torch.arange(height, device=features.device, dtype=features.dtype)
    columns = torch.arange(width, device=features.device, dtype=features.dtype)
    wy = (1 - (query[:, :1] * (height - 1) - rows).abs()).clamp_min(0)
    wx = (1 - (query[:, 1:] * (width - 1) - columns).abs()).clamp_min(0)
    weights = (wy[:, :, None] * wx[:, None, :]).flatten(1)
    return (features.flatten(2) @ weights.T).transpose(1, 2)


class LocalFeatureQVFDecoder(nn.Module):
    """64-number code -> 16x16x32 features -> unchanged five-qubit QVF head.

    The expanded feature grid is derived entirely from the global code; it is
    not a second image input or an encoder-to-decoder skip connection.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig(mode="ae", field_layers=2)
        if (self.cfg.latent_dim, self.cfg.image_size, self.cfg.field_width,
                self.cfg.field_layers, self.cfg.frequencies, self.cfg.qubits,
                self.cfg.circuit_depth, self.cfg.decoder, self.cfg.readout) != (
                    64, 32, 128, 2, 6, 5, 5, "qvf", "vanilla"):
            raise ValueError("This spatial decoder requires its declared 64-code/five-qubit configuration")
        self.generator = nn.Sequential(
            nn.Linear(64, 4 * 4 * 128), nn.Unflatten(1, (128, 4, 4)), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
        )
        self.field = QVFImage({
            "L_embed": 6, "latent_size": 32, "hid_dim": 128, "out_dim": 32,
            "num_hidden_layers_energy": 2, "skip_connection": False,
            "num_repetations": 5, "use_identity_block": False, "gaussian_std": 0.1,
            "output_channels": 3, "use_parity": True,
        })

    @property
    def circuit(self):
        return self.field.quantum

    def prepare_features(self, code):
        if code.ndim != 2 or code.shape[1] != 64:
            raise ValueError("Expected a 64-number code for each image")
        return self.generator(code)

    def query_features(self, features, coordinates):
        local = local_features(features, coordinates)
        position = positional_embedding(coordinates, 6)
        inputs = torch.cat((local, position[None].expand(len(features), -1, -1)), -1)
        return self.field(inputs.flatten(0, 1)).reshape(len(features), len(coordinates), 3)

    def forward(self, code, coordinates):
        return self.query_features(self.prepare_features(code), coordinates)
