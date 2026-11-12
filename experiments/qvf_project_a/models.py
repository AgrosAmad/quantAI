"""A classical image encoder and a shared QVF coordinate decoder.

New models call the validated vanilla QVF field directly. The older affine RGB
readout remains available only as an explicitly named compatibility option.
Latents are ordinary classical vectors; the encoder and VAE objective are additions.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qvf_repro.qvf_fast import FastQuantumLayer, MLPEmbedding, QVFImage, positional_embedding


@dataclass
class ModelConfig:
    latent_dim: int = 64
    image_size: int = 32
    encoder_width: int = 32
    field_width: int = 128
    field_layers: int = 3
    frequencies: int = 6
    qubits: int = 5
    circuit_depth: int = 5
    decoder: str = "qvf"  # qvf, no_circuit, mlp
    mode: str = "vae"     # ae is a deterministic control
    readout: str = "vanilla"  # original grouped Z readout; legacy_affine reloads older experiments

    def __post_init__(self):
        if self.image_size != 32:
            raise ValueError("This first encoder is for 32x32 images.")
        if self.latent_dim < 1 or self.encoder_width < 1 or self.field_width < 1:
            raise ValueError("Network widths and latent_dim must be positive.")
        if self.field_layers < 1 or self.frequencies < 0:
            raise ValueError("field_layers must be positive and frequencies nonnegative.")
        if not 3 <= self.qubits <= 6 or self.circuit_depth < 1:
            raise ValueError("This exact-simulation pilot supports 3-6 qubits and positive depth.")
        if self.decoder not in {"qvf", "no_circuit", "mlp"} or self.mode not in {"vae", "ae"}:
            raise ValueError("Unknown decoder or autoencoder mode.")
        if self.readout not in {"vanilla", "legacy_affine"}:
            raise ValueError("Unknown QVF readout.")

    @classmethod
    def from_saved(cls, values):
        """Pre-readout-version checkpoints/configs always used the affine map."""
        return cls(**{"readout": "legacy_affine", **values})


def pixel_coordinates(height, width, *, device=None):
    """Row-major (row/height, column/width), as in the released QVF convention."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device),
                                torch.arange(width, device=device), indexing="ij")
    return torch.stack((rows / height, cols / width), dim=-1).reshape(-1, 2)


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.mode
        width = cfg.encoder_width
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(2 * width, 4 * width, 4, 2, 1), nn.SiLU(),
            nn.Flatten(),
        )
        self.mean = nn.Linear(4 * width * 4 * 4, cfg.latent_dim)
        self.log_variance = nn.Linear(4 * width * 4 * 4, cfg.latent_dim) if cfg.mode == "vae" else None
        if self.log_variance is not None:
            nn.init.zeros_(self.log_variance.weight)
            nn.init.constant_(self.log_variance.bias, -4.0)

    def forward(self, images):
        features = self.features(images)
        mean = self.mean(features)
        logvar = self.log_variance(features).clamp(-12.0, 6.0) if self.log_variance is not None else None
        return mean, logvar


def sample_posterior(mean, logvar, *, generator=None, noise=None):
    if logvar is None:
        return mean
    if noise is None:
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + torch.exp(0.5 * logvar) * noise


def kl_per_image(mean, logvar):
    """KL to N(0,I), summed over latent entries, one scalar per image."""
    if logvar is None:
        return mean.new_zeros(mean.shape[0])
    return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)


class CoordinateDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.readout == "vanilla" and cfg.decoder != "mlp":
            # Own the exact audited model, rather than copying its amplitude or
            # RGB formulas. Coordinate expansion is the only adapter here.
            self.field = QVFImage({
                "L_embed": cfg.frequencies, "latent_size": cfg.latent_dim,
                "hid_dim": cfg.field_width, "out_dim": 2 ** cfg.qubits,
                "num_hidden_layers_energy": cfg.field_layers, "skip_connection": False,
                "num_repetations": cfg.circuit_depth if cfg.decoder == "qvf" else 0,
                "use_identity_block": False, "gaussian_std": 0.1,
                "output_channels": 3, "use_parity": True,
            })
            return
        coord_dim = 2 + 4 * cfg.frequencies
        self.network = MLPEmbedding(
            cfg.latent_dim + coord_dim, cfg.field_width,
            3 if cfg.decoder == "mlp" else 2 ** cfg.qubits,
            cfg.field_layers, amplitude_output=False,
        )
        self.quantum = None if cfg.decoder == "mlp" else FastQuantumLayer(
            cfg.qubits, cfg.circuit_depth if cfg.decoder == "qvf" else 0, False,
        )

    @property
    def circuit(self):
        return self.field.quantum if hasattr(self, "field") else self.quantum

    @property
    def amplitude_network(self):
        return self.field.embedder if hasattr(self, "field") else self.network

    def forward(self, code, coordinates):
        """code: [B,D], coordinates: [P,2], result: [B,P,3]."""
        if code.ndim != 2 or code.shape[1] != self.cfg.latent_dim:
            raise ValueError("Expected one latent vector per image.")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Expected a shared list of 2D query coordinates.")
        features = positional_embedding(coordinates, self.cfg.frequencies) if self.cfg.frequencies else coordinates
        batch, points = code.shape[0], coordinates.shape[0]
        inputs = torch.cat((code[:, None].expand(-1, points, -1),
                            features[None].expand(batch, -1, -1)), dim=-1)
        inputs = inputs.reshape(batch * points, -1)
        if hasattr(self, "field"):
            return self.field(inputs).reshape(batch, points, 3)
        scores = self.network(inputs)
        if self.quantum is None:
            rgb = scores.sigmoid()
        else:
            # Equivalent to sqrt(softmax(scores)), with stable derivatives when
            # tiny probabilities would otherwise round to zero before sqrt.
            amplitudes = torch.exp(0.5 * F.log_softmax(scores, dim=-1))
            rgb = 0.5 * (self.quantum(amplitudes)[:, :3] + 1.0)
        return rgb.reshape(batch, points, 3)


class QVFAutoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg)
        self.decoder = CoordinateDecoder(cfg)
        self.register_buffer("coordinates", pixel_coordinates(cfg.image_size, cfg.image_size), persistent=False)

    def forward(self, images, indices=None, *, sample=True, generator=None):
        mean, logvar = self.encoder(images)
        code = sample_posterior(mean, logvar, generator=generator) if sample else mean
        coords = self.coordinates if indices is None else self.coordinates[indices]
        return self.decoder(code, coords), mean, logvar

    def render(self, code, pixel_chunk=1024):
        if pixel_chunk < 1:
            raise ValueError("pixel_chunk must be positive")
        pixels = torch.cat([self.decoder(code, coords) for coords in self.coordinates.split(pixel_chunk)], dim=1)
        return pixels.transpose(1, 2).reshape(-1, 3, self.cfg.image_size, self.cfg.image_size)
