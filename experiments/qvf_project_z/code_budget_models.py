"""Z10: original QVF with a wider global image code; frozen Z8 stays intact."""
from dataclasses import dataclass

import torch
from torch import nn

from experiments.qvf_repro.qvf_fast import QVFImage, positional_embedding
from .coverage_models import VanillaConfig, VanillaField, resources as original_resources

NAMES = ('vanilla_qvf256',)
LABELS = {'vanilla_qvf256': 'Vanilla QVF (256-value code)'}


@dataclass(frozen=True)
class CodeBudgetConfig(VanillaConfig):
    architecture: str = 'vanilla_qvf256'
    latent_dim: int = 256


class CodeBudgetField(VanillaField):
    """Inherit the original QVF forward, tracing and circuit cache verbatim."""
    def __init__(self, config):
        nn.Module.__init__(self)
        if config.latent_dim <= 0:
            raise ValueError('A global code must have positive size')
        self.cfg = config
        self.field = QVFImage({
            'L_embed': config.frequencies, 'latent_size': config.latent_dim,
            'hid_dim': config.width, 'out_dim': 32,
            'num_hidden_layers_energy': 3, 'skip_connection': False,
            'num_repetations': 5, 'use_identity_block': False, 'gaussian_std': .1,
            'output_channels': 3, 'use_parity': True,
        })
        self._cache = None

    def inputs(self, codes, coordinates):
        if codes.ndim != 2 or codes.shape[1] != self.cfg.latent_dim:
            raise ValueError(f'Expected {self.cfg.latent_dim} global code entries')
        position = positional_embedding(coordinates, self.cfg.frequencies)
        return torch.cat((codes[:, None].expand(-1, len(coordinates), -1),
                          position[None].expand(len(codes), -1, -1)), -1).flatten(0, 1)


def build_model(name, seed=0, **config):
    if name not in NAMES:
        raise ValueError(name)
    # The inherited panel config also describes spatial references. Only the
    # original QVF settings apply here; the named control fixes its code to256.
    if 'latent_dim' in config and config['latent_dim'] != 256:
        raise ValueError('The declared QVF-256 control fixes its code size')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cfg = CodeBudgetConfig(**{k: config[k] for k in ('image_size', 'frequencies', 'width')})
        return CodeBudgetField(cfg)


def resources(model, name):
    if name not in NAMES:
        raise ValueError(name)
    info = original_resources(model, 'vanilla_qvf')
    return {**info, 'image_code_entries': model.cfg.latent_dim,
            'benchmark_architecture': name,
            'extra_first_layer_weights_vs_64': (model.cfg.latent_dim - 64) * model.cfg.width,
            'code_is_global_vector': True,
            'quantum_circuit_and_readout_unchanged': True}
