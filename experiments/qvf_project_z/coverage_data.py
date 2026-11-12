"""Pinned CIFAR dog split for the full Z8 low-resolution comparison."""
import json
from pathlib import Path
import hashlib
import numpy as np
from torchvision.datasets import CIFAR10
from experiments.qvf_project_a.train import sha256, write_json

HERE = Path(__file__).resolve().parent


def prepare():
    destination = HERE / 'data/cifar_dogs_z8/prepared_32'
    previous = HERE / 'results/20261110_z5_full/protocol.json'
    p = json.loads(previous.read_text())
    if destination.exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        assert manifest['parent_protocol_sha256'] == sha256(previous)
        for split, digest in manifest['arrays_sha256'].items():
            assert sha256(destination / (split + '.npy')) == digest
        return destination
    training = CIFAR10(p['data_root'], train=True, download=False)
    testing = CIFAR10(p['data_root'], train=False, download=False)
    ids = {'train': p['indices']['train'], 'development': p['indices']['validation']}
    assert len(ids['train']) == 4744 and len(ids['development']) == 256
    assert not set(ids['train']).intersection(ids['development'])
    for group in ids.values():
        assert all(training.targets[i] == 5 for i in group)
    arrays = {key: training.data[value] for key, value in ids.items()}
    def digest(image):
        return hashlib.sha256(image.tobytes()).hexdigest()
    known = {digest(image) for array in arrays.values() for image in array}
    candidates = np.flatnonzero(np.array(testing.targets) == 5)
    chosen = []
    for index in np.random.default_rng(20261112).permutation(candidates):
        key = digest(testing.data[index])
        if key not in known:
            chosen.append(int(index)); known.add(key)
        if len(chosen) == 64:
            break
    assert len(chosen) == 64
    ids['test'] = chosen
    arrays['test'] = testing.data[chosen]
    destination.mkdir(parents=True, exist_ok=False)
    for split, array in arrays.items():
        np.save(destination / (split + '.npy'), array)
    write_json(destination / 'manifest.json', {
        'dataset': 'CIFAR-10', 'class': 'dog', 'class_index': 5, 'native_resolution': 32,
        'train_development_source': 'Official CIFAR-10 training split, unchanged Z5 IDs',
        'test_source': '64 preselected official test-set dog images; no training or selection on these pixels',
        'dataset_indices': ids, 'parent_protocol_sha256': sha256(previous),
        'arrays_sha256': {split: sha256(destination / (split + '.npy')) for split in arrays},
        'test_exact_duplicates_against_train_development_removed': True,
        'train_development_exact_duplicates': len(arrays['train']) + len(arrays['development']) -
            len({digest(image) for split in ('train', 'development') for image in arrays[split]}),
        'limits': ['Training/development memberships preserved from Z5; no new deduplication of those splits.',
                   'Near duplicates not independently excluded.',
                   'Different dataset from the Stanford Dogs panel; not a resolution-only ablation.']})
    return destination


if __name__ == '__main__':
    print(prepare())
