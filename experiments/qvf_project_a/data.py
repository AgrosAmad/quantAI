"""Seeded train/validation splits and posterior caching; the CIFAR test set is untouched."""

import torch
from torchvision.datasets import CIFAR10


def split_indices(targets, class_idx, n_train, n_val, seed):
    if n_train < 2 or n_val < 2:
        raise ValueError("At least two training and two validation images are required.")
    if class_idx not in range(-1, 10):
        raise ValueError("class_idx must be -1 (all classes) or 0-9")
    targets = torch.as_tensor(targets)
    candidates = torch.arange(len(targets)) if class_idx == -1 else torch.where(targets == class_idx)[0]
    if n_train + n_val > len(candidates):
        raise ValueError("Requested split exceeds the available training images.")
    generator = torch.Generator().manual_seed(seed)
    shuffled = candidates[torch.randperm(len(candidates), generator=generator)]
    return {"train": shuffled[:n_train], "validation": shuffled[n_train:n_train + n_val]}


def extend_split_indices(targets, class_idx, reference, n_train, seed):
    """Keep validation IDs and the old training prefix fixed while adding data."""
    targets = torch.as_tensor(targets)
    train = torch.as_tensor(reference["train"], dtype=torch.long)
    validation = torch.as_tensor(reference["validation"], dtype=torch.long)
    combined = torch.cat((train, validation))
    if len(combined.unique()) != len(combined) or len(train) < 2 or len(validation) < 2:
        raise ValueError("Reference split must contain unique, disjoint training and validation IDs.")
    if (combined < 0).any() or (combined >= len(targets)).any():
        raise ValueError("Reference dataset indices are out of range.")
    if class_idx not in range(-1, 10) or (class_idx != -1 and (targets[combined] != class_idx).any()):
        raise ValueError("Reference split does not match the requested class.")
    candidates = torch.arange(len(targets)) if class_idx == -1 else torch.where(targets == class_idx)[0]
    remaining = candidates[~torch.isin(candidates, combined)]
    if not len(train) <= n_train <= len(train) + len(remaining):
        raise ValueError("Expanded training size must include the original training set and exclude validation.")
    generator = torch.Generator().manual_seed(seed)
    extra = remaining[torch.randperm(len(remaining), generator=generator)[:n_train - len(train)]]
    return {"train": torch.cat((train, extra)), "validation": validation}


def load_data(root, class_idx, n_train, n_val, seed, reference_indices=None):
    dataset = CIFAR10(str(root), train=True, download=False)
    if reference_indices is None:
        indices = split_indices(dataset.targets, class_idx, n_train, n_val, seed)
    else:
        if len(reference_indices["validation"]) != n_val:
            raise ValueError("Validation size must match the reference split.")
        indices = extend_split_indices(dataset.targets, class_idx, reference_indices, n_train, seed)
    images = {name: torch.from_numpy(dataset.data[index.numpy()].copy()).permute(0, 3, 1, 2).float() / 255.0
              for name, index in indices.items()}
    return images, {name: index.tolist() for name, index in indices.items()}


@torch.no_grad()
def encode_images(model, images, batch_size, device):
    model.eval()
    means, logvars = [], []
    for batch in images.split(batch_size):
        mean, logvar = model.encoder(batch.to(device))
        means.append(mean.cpu())
        if logvar is not None:
            logvars.append(logvar.cpu())
    return {"mean": torch.cat(means), "logvar": torch.cat(logvars) if logvars else None}


def posterior_normalization(train_posterior):
    """Exact first two moments of the empirical TRAINING posterior mixture.

    Sampling variance = variance of means + mean posterior variance. Validation
    data never enters these statistics. A deterministic encoder has no second term.
    """
    means, logvar = train_posterior["mean"].double(), train_posterior["logvar"]
    variance = means.var(dim=0, unbiased=False)
    if logvar is not None:
        variance = variance + logvar.double().exp().mean(dim=0)
    return {"mean": means.mean(dim=0).float(), "std": variance.clamp_min(1e-12).sqrt().float()}
