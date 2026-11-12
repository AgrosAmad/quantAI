"""Spatial coordinate readouts from frozen, pre-RGB RAE decoder features."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qvf_repro.qvf_fast import QVFImage, parity_decode, positional_embedding


@dataclass
class ReadoutConfig:
    kind: str = "qvf"
    input_channels: int = 1152
    adapter_channels: int = 128
    grid_size: int = 16
    image_size: int = 32
    field_width: int = 128
    field_layers: int = 3
    frequencies: int = 6

    def validate(self):
        if self.kind not in {"qvf", "classical"}:
            raise ValueError("Unknown readout kind")
        if self.image_size % self.grid_size:
            raise ValueError("Native image size must be an integer multiple of the feature grid")
        if min(asdict(self)[k] for k in asdict(self) if k != "kind") < 1:
            raise ValueError("All architecture dimensions must be positive")


def pixel_layout(image_size, grid_size, frequencies):
    """Row-major pixels, containing patch, global position and within-patch offset.

    Global coordinates retain the original QVF (row/size, column/size) convention.
    Patch offsets use pixel centres and range from -1 to +1 inside each patch.
    For 32px images and 16x16 features, offsets are -0.5 and +0.5 on each axis.
    """
    if image_size % grid_size:
        raise ValueError("Feature grid must divide the native image size")
    row, col = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing="ij")
    pixels = torch.stack((row, col), -1).reshape(-1, 2)
    scale = image_size // grid_size
    patch = pixels // scale
    patch_ids = patch[:, 0] * grid_size + patch[:, 1]
    offset = 2 * ((pixels.float() + 0.5) / scale - patch.float()) - 1
    positions = positional_embedding(pixels.float() / image_size, frequencies)
    return patch_ids, torch.cat((offset, positions), -1)


class RAEFeatureReadout(nn.Module):
    """Shared spatial adapter and MLP; either original QVF or linear RGB readout.

    No image/code lookup, RGB residual, learned post-quantum layer or encoder skip.
    Classical control uses the same 32-output MLP followed by a learned linear RGB
    projection. Its 99 output parameters closely match QVF's 150 stored angles.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ReadoutConfig()
        self.cfg.validate()
        c = self.cfg.adapter_channels
        self.project = nn.Conv2d(self.cfg.input_channels, c, 1)
        self.spatial1 = nn.Conv2d(c, c, 3, padding=1)
        self.spatial2 = nn.Conv2d(c, c, 3, padding=1)
        field = QVFImage(self.field_config())
        self.embedder = field.embedder
        if self.cfg.kind == "qvf":
            self.quantum = field.quantum
        else:
            self.embedder.amplitude_output = False
            self.rgb = nn.Linear(32, 3)
        patch_ids, position = pixel_layout(self.cfg.image_size, self.cfg.grid_size, self.cfg.frequencies)
        self.register_buffer("patch_ids", patch_ids)
        self.register_buffer("position", position)

    def field_config(self):
        return {"L_embed": self.cfg.frequencies, "latent_size": self.cfg.adapter_channels + 2,
                "hid_dim": self.cfg.field_width, "out_dim": 32,
                "num_hidden_layers_energy": self.cfg.field_layers, "skip_connection": False,
                "num_repetations": 5, "use_identity_block": False, "gaussian_std": 0.1,
                "output_channels": 3, "use_parity": True}

    def prepare_features(self, features):
        expected = (self.cfg.input_channels, self.cfg.grid_size, self.cfg.grid_size)
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected}], got {tuple(features.shape)}")
        x = F.silu(self.project(features))
        return F.silu(x + self.spatial2(F.silu(self.spatial1(x))))

    def query_inputs(self, adapted, pixel_ids):
        local = adapted.flatten(2)[:, :, self.patch_ids[pixel_ids]].transpose(1, 2)
        position = self.position[pixel_ids].unsqueeze(0).expand(len(adapted), -1, -1)
        return torch.cat((local, position), -1)

    def query(self, adapted, pixel_ids):
        inputs = self.query_inputs(adapted, pixel_ids)
        encoded = self.embedder(inputs.flatten(0, 1))
        if self.cfg.kind == "qvf":
            rgb = parity_decode(self.quantum(encoded), 3)
        else:
            rgb = self.rgb(encoded)
        return rgb.reshape(len(adapted), len(pixel_ids), 3)

    def forward(self, features, pixel_ids):
        return self.query(self.prepare_features(features), pixel_ids)


@torch.no_grad()
def render_readout(model, features, batch_size=32):
    model.eval()
    points = torch.arange(model.cfg.image_size ** 2, device=features.device)
    images = []
    for batch in features.split(batch_size):
        rgb = model(batch, points)
        images.append(rgb.transpose(1, 2).reshape(len(batch), 3, model.cfg.image_size, model.cfg.image_size).cpu())
    return torch.cat(images)
