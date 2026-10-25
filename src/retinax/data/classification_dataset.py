from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .. import CLASS_NAMES

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def transforms_for(train=False):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def make_loaders(root, batch_size=32, seed=42, workers=0, val_fraction=.15, test_fraction=.15):
    root = Path(root)
    raw = datasets.ImageFolder(root, transform=transforms_for(False))
    if set(raw.classes) != set(CLASS_NAMES):
        raise ValueError(f"Expected folders {CLASS_NAMES}; found {raw.classes}")
    labels = [y for _, y in raw.samples]
    idx = list(range(len(labels)))
    try:
        train_idx, hold_idx = train_test_split(idx, test_size=val_fraction + test_fraction, stratify=labels, random_state=seed)
    except ValueError:
        # Tiny smoke-test datasets may not have enough holdout samples for stratification.
        train_idx, hold_idx = train_test_split(idx, test_size=val_fraction + test_fraction, random_state=seed)
    hold_labels = [labels[i] for i in hold_idx]
    relative_test = test_fraction / (val_fraction + test_fraction)
    try:
        val_idx, test_idx = train_test_split(hold_idx, test_size=relative_test, stratify=hold_labels, random_state=seed)
    except ValueError:
        val_idx, test_idx = train_test_split(hold_idx, test_size=relative_test, random_state=seed)
    train_ds = Subset(datasets.ImageFolder(root, transform=transforms_for(True)), train_idx)
    eval_ds = datasets.ImageFolder(root, transform=transforms_for(False))
    val_ds, test_ds = Subset(eval_ds, val_idx), Subset(eval_ds, test_idx)
    counts = Counter(labels[i] for i in train_idx)
    weights = torch.tensor([1.0 / counts[labels[i]] for i in train_idx], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, sampler=sampler, **kw), DataLoader(val_ds, shuffle=False, **kw), DataLoader(test_ds, shuffle=False, **kw)
