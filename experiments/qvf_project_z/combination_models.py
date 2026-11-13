"""Z9: attention within each scale, then three quantum branches and fusion.

Attention is classical and local: each of the existing five spatial input slots
attends to at most 5x5 grid tokens. The seven-qubit circuits, five data uploads,
spatial wire roles and RGB measurement map are unchanged from Z6/Z7.
"""
import torch
from torch import nn

from experiments.qvf_repro.qvf_fast import positional_embedding
from .hires_models import HiresField, HiresConfig, build_model as spatial_build, component_seed
from .routing_models import AttentionFusionField, RoutingField
from .coverage_models import trace as old_trace, readout as old_readout, resources as old_resources

NAMES = ('scale_attention_mean', 'scale_attention_fusion', 'scale_attention_coherent')
ALIASES = dict(zip(NAMES, ('separate_scales', 'classical_attention16', 'coherent_routing')))
LABELS = {
    'scale_attention_mean': 'Per-scale attention + three-head mean',
    'scale_attention_fusion': 'Per-scale attention + output attention',
    'scale_attention_coherent': 'Per-scale attention + coherent routing',
}
WINDOW_RADIUS = 2
ATTENTION_DIM = 16


class AttentiveScaleField(HiresField):
    def __init__(self, seed=0, **config):
        super().__init__(HiresConfig(architecture='separate_scales', **config))
        # Identical encoder/value/circuit initialization to the old three-head arm.
        self.load_state_dict(spatial_build('separate_scales', seed, **config).state_dict())
        pos_width = 2 + 4 * self.cfg.frequencies
        self.queries = nn.ModuleList(nn.Linear(pos_width, ATTENTION_DIM) for _ in range(3))
        self.keys = nn.ModuleList(nn.Linear(6, ATTENTION_DIM) for _ in range(3))
        for kind in ('queries', 'keys'):
            for i, layer in enumerate(getattr(self, kind)):
                torch.manual_seed(component_seed(seed, f'{kind}.{i}'))
                layer.reset_parameters()
        y, x = torch.meshgrid(torch.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1),
                              torch.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1), indexing='ij')
        self.register_buffer('window_offsets', torch.stack((y, x), -1).reshape(-1, 2), persistent=False)

    def scale_features(self, tokens, coordinates, scale):
        size = self.cfg.grid_sizes[scale]
        locations = (coordinates[:, None] + self.offsets[None].to(coordinates.dtype)
                     / (size - 1)).clamp(0, 1).reshape(-1, 2)
        centers = (locations * (size - 1)).floor().long()
        candidates = centers[:, None] + self.window_offsets[None]
        valid = ((candidates >= 0) & (candidates < size)).all(-1)
        indices = candidates.clamp(0, size - 1)
        flat_indices = indices[..., 0] * size + indices[..., 1]
        gathered = tokens.index_select(1, flat_indices.flatten()).reshape(
            len(tokens), len(locations), len(self.window_offsets), 4)
        positions = candidates.to(tokens.dtype) / (size - 1)
        query = self.queries[scale](positional_embedding(locations, self.cfg.frequencies))
        # Algebraically Q K^T / sqrt(16), without materializing a 16-value key
        # for every batch/query/window slot. Tested against dense masked attention.
        projected = query @ self.keys[scale].weight / ATTENTION_DIM**.5
        scores = (gathered * projected[None, :, None, :4]).sum(-1)
        scores = scores + (positions * projected[:, None, 4:]).sum(-1)[None]
        scores = scores + (query @ self.keys[scale].bias)[None, :, None] / ATTENTION_DIM**.5
        distance = ((locations[:, None] - positions) * (size - 1)).square().sum(-1)
        scores = (scores - 2. * distance[None]).masked_fill(~valid[None], -torch.inf)
        weights = scores.softmax(-1)
        pooled = (weights[..., None] * gathered).sum(-2)
        return self.values[scale](pooled).reshape(len(tokens), len(coordinates), 5, 4)

    def spatial_features(self, codes, coordinates):
        features = []
        offset = 0
        for i, size in enumerate(self.cfg.grid_sizes):
            count = 4 * size * size
            tokens = codes[:, offset:offset + count].reshape(-1, 4, size * size).transpose(1, 2)
            features.append(self.scale_features(tokens, coordinates, i))
            offset += count
        return torch.cat(features, -1)


def build_model(name, seed=0, **config):
    if name not in NAMES:
        raise ValueError(name)
    config = dict(config)
    config.pop('architecture', None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = AttentiveScaleField(seed, **config)
        if name == 'scale_attention_mean':
            return base
        model = (AttentionFusionField(seed, **config) if name == 'scale_attention_fusion'
                 else RoutingField('coherent_routing', seed, **config))
        model.base = base
        model.cfg = base.cfg
        return model


def trace(model, name, codes, coordinates):
    return old_trace(model, ALIASES[name], codes, coordinates)


def readout(model, name, probabilities, coordinates, batch):
    return old_readout(model, ALIASES[name], probabilities, coordinates, batch)


def resources(model, name):
    result = old_resources(model, ALIASES[name])
    result.update({'benchmark_architecture': name, 'classical_attention': True,
                   'classical_output_attention': name == 'scale_attention_fusion',
                   'within_scale_attention': 'classical local query/key softmax',
                   'attention_window_max_tokens': (2 * WINDOW_RADIUS + 1)**2,
                   'attention_slots_per_scale': 5, 'attention_scales': 3,
                   'attention_key_dimension': ATTENTION_DIM,
                   'quantum_circuit_unchanged_from': ALIASES[name],
                   'global_attention_over_all_grid_tokens': False})
    return result
