"""
Data preprocessing module for IML Hackathon Challenge 2.

Provides:
  - Stratified 4-way split: train (70%), dev1 (10%), dev2 (10%), test (10%)
  - Multiple augmentation levels: standard, aggressive, extreme
  - Custom augmentation transforms (GaussianNoise, ChannelShuffle, PatchShuffle)
  - Batch-level augmentations (MixUp, CutMix)
  - Validation transforms matching the evaluator pipeline
  - DataLoaders for all splits + augmented validation
"""

import math
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from standardization import (
    validate_images,
    fix_orientation,
    convert_to_rgb,
    preprocess_batch,
)


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paths relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "dataset"


def _first_existing_path(candidates: List[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


TRAIN_DIR = _first_existing_path([
    PROJECT_ROOT / "train_set" / "train",
    DATA_ROOT / "train_set" / "train",
])
AUG_DIR = _first_existing_path([
    DATA_ROOT / "augmentations" / "augmentations",
    PROJECT_ROOT / "augmentations" / "augmentations",
])

SPLIT_SEED = 42

# Split ratios — must sum to 1.0
SPLIT_RATIOS = {
    "train": 0.70,
    "dev1":  0.10,
    "dev2":  0.10,
    "test":  0.10,
}

IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 2


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM AUGMENTATION TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

class GaussianNoise:
    """
    Add random Gaussian noise to a tensor image.

    Simulates sensor noise, compression artifacts, and low-light conditions.
    Applied after ToTensor + Normalize.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ChannelShuffle:
    """
    Randomly permute the RGB channels.

    Forces the model to learn shape-based features rather than relying
    on specific color channel ordering (e.g. "sky is always in blue channel").
    Applied after ToTensor.
    """

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            perm = torch.randperm(3)
            return tensor[perm]
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class PatchShuffle:
    """
    Divide the image into a grid and randomly shuffle the patches.

    Disrupts spatial coherence, forcing the model to rely on local texture
    and object parts rather than global spatial layout.
    Applied after ToTensor.
    """

    def __init__(self, grid_size: int = 4, p: float = 0.2):
        self.grid_size = grid_size
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor

        c, h, w = tensor.shape
        gh = h // self.grid_size
        gw = w // self.grid_size

        # Extract patches
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = tensor[:, i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                patches.append(patch)

        # Shuffle and reassemble
        random.shuffle(patches)
        rows = []
        for i in range(self.grid_size):
            row = torch.cat(patches[i * self.grid_size:(i + 1) * self.grid_size], dim=2)
            rows.append(row)
        return torch.cat(rows, dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(grid={self.grid_size}, p={self.p})"


class RandomSolarize:
    """
    Randomly solarize the image (invert pixels above a threshold).

    Creates unusual color distributions that teach the model to be
    invariant to extreme color transformations.
    Applied on PIL images.
    """

    def __init__(self, threshold: int = 128, p: float = 0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img, self.threshold)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold}, p={self.p})"


class RandomPosterize:
    """
    Randomly reduce the number of bits per color channel.

    Simulates aggressive color quantization. Teaches the model to handle
    images with reduced color depth.
    Applied on PIL images.
    """

    def __init__(self, bits: int = 4, p: float = 0.2):
        self.bits = bits
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.posterize(img, self.bits)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(bits={self.bits}, p={self.p})"


class RandomSharpness:
    """
    Randomly adjust image sharpness (sharpen or blur).

    Simulates varying focus conditions and image quality differences.
    Applied on PIL images.
    """

    def __init__(self, range: Tuple[float, float] = (0.5, 2.0), p: float = 0.3):
        self.range = range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            factor = random.uniform(*self.range)
            return ImageEnhance.Sharpness(img).enhance(factor)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(range={self.range}, p={self.p})"


class RandomInvert:
    """
    Randomly invert the colors of the image.

    An extreme color transformation that forces shape-based recognition.
    Applied on PIL images.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.invert(img)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomEqualize:
    """
    Randomly equalize the histogram of the image.

    Normalizes contrast, simulating different exposure conditions.
    Applied on PIL images.
    """

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.equalize(img)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH-LEVEL AUGMENTATIONS (MixUp / CutMix)
# ══════════════════════════════════════════════════════════════════════════════

def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """
    MixUp: blend pairs of images and their labels.

    Creates virtual training examples by linearly interpolating between
    random pairs. Encourages smoother decision boundaries.

    Args:
        images: Batch tensor [B, C, H, W]
        labels: Label tensor [B] (integer class indices)
        alpha:  Beta distribution parameter (higher = more mixing)

    Returns:
        (mixed_images, labels_a, labels_b, lam)
        Use mixed loss: loss = lam * loss(pred, labels_a) + (1-lam) * loss(pred, labels_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """
    CutMix: cut and paste rectangular patches between training images.

    More aggressive than MixUp — forces the model to identify objects from
    partial views. The label is mixed proportionally to the patch area.

    Args:
        images: Batch tensor [B, C, H, W]
        labels: Label tensor [B] (integer class indices)
        alpha:  Beta distribution parameter

    Returns:
        (mixed_images, labels_a, labels_b, lam)
        Use mixed loss: loss = lam * loss(pred, labels_a) + (1-lam) * loss(pred, labels_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, h, w = images.shape

    # Compute random bounding box
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = random.randint(0, w)
    cy = random.randint(0, h)

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)

    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_cutmix_loss(criterion, pred, labels_a, labels_b, lam):
    """
    Compute the mixed loss for MixUp or CutMix.

    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        pred:      Model predictions [B, num_classes]
        labels_a:  First set of labels
        labels_b:  Second set of labels
        lam:       Mixing coefficient

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORM PIPELINES (Standard / Aggressive / Extreme)
# ══════════════════════════════════════════════════════════════════════════════

def get_val_transform():
    """
    Validation / inference transform — matches evaluate.py exactly.

    Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform(level: str = "aggressive"):
    """
    Training transform with configurable augmentation intensity.

    Args:
        level: One of "standard", "aggressive", or "extreme".

    Returns:
        A transforms.Compose pipeline.
    """
    if level == "standard":
        return _standard_train_transform()
    elif level == "aggressive":
        return _aggressive_train_transform()
    elif level == "extreme":
        return _extreme_train_transform()
    else:
        raise ValueError(f"Unknown augmentation level: {level!r}. "
                         f"Choose from: 'standard', 'aggressive', 'extreme'.")


def _standard_train_transform():
    """
    Standard augmentations — mild color/spatial transforms.
    Good baseline for initial training.
    """
    return transforms.Compose([
        # ── Spatial ───────────────────────────────────────────────────────
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),

        # ── Color ─────────────────────────────────────────────────────────
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.2),

        # ── To Tensor + Normalize ─────────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ── Occlusion ─────────────────────────────────────────────────────
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def _aggressive_train_transform():
    """
    Aggressive augmentations — adds blur, perspective, solarize, posterize,
    sharpness, equalize, stronger erasing, and Gaussian noise.
    Recommended for robustness training.
    """
    return transforms.Compose([
        # ── Spatial ───────────────────────────────────────────────────────
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.8, 1.2),
            shear=(-10, 10),
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),

        # ── Color ─────────────────────────────────────────────────────────
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.15,
        ),
        transforms.RandomGrayscale(p=0.2),
        RandomSolarize(threshold=128, p=0.15),
        RandomPosterize(bits=4, p=0.15),
        RandomSharpness(range=(0.3, 2.5), p=0.3),
        RandomEqualize(p=0.2),

        # ── Blur ──────────────────────────────────────────────────────────
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

        # ── To Tensor + Normalize ─────────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ── Tensor-space augmentations ────────────────────────────────────
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        GaussianNoise(mean=0.0, std=0.03),
    ])


def _extreme_train_transform():
    """
    Extreme augmentations — maximum diversity. Adds channel shuffling,
    patch shuffling, color inversion, and aggressive noise/erasing.
    Use with caution — may slow convergence.
    """
    return transforms.Compose([
        # ── Spatial ───────────────────────────────────────────────────────
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(0.7, 1.3),
            shear=(-15, 15),
        ),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.4),

        # ── Color ─────────────────────────────────────────────────────────
        transforms.ColorJitter(
            brightness=0.6,
            contrast=0.6,
            saturation=0.6,
            hue=0.2,
        ),
        transforms.RandomGrayscale(p=0.25),
        RandomSolarize(threshold=100, p=0.2),
        RandomPosterize(bits=3, p=0.2),
        RandomSharpness(range=(0.2, 3.0), p=0.4),
        RandomEqualize(p=0.3),
        RandomInvert(p=0.1),

        # ── Blur ──────────────────────────────────────────────────────────
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0)),

        # ── To Tensor + Normalize ─────────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ── Tensor-space augmentations ────────────────────────────────────
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25)),
        GaussianNoise(mean=0.0, std=0.05),
        ChannelShuffle(p=0.2),
        PatchShuffle(grid_size=4, p=0.15),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class SplitImageDataset(Dataset):
    """
    A dataset constructed from an explicit list of (image_path, label) pairs.

    Unlike ImageNetSubset from base_model.py which scans a directory,
    this takes a pre-built sample list so we can control the split.

    Applies standardization (EXIF fix + RGB conversion) before transforms.
    """

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)

        # Standardization: EXIF orientation fix + RGB conversion
        image = fix_orientation([image])[0]
        image = convert_to_rgb([image])[0]

        if self.transform:
            image = self.transform(image)

        return image, label


class AugmentationDataset(Dataset):
    """
    Loads the provided augmentation samples from the augmentations directory.

    Handles the nested structure:
      augmentations/augmentations/<aug_type>/<class_name>/<image>.jpg
    """

    def __init__(self, aug_dir: Path, class_to_idx: dict, transform=None):
        self.transform = transform
        self.samples = []

        if not aug_dir.exists():
            raise FileNotFoundError(f"Augmentation directory not found: {aug_dir}")

        for aug_type_dir in sorted(aug_dir.iterdir()):
            if not aug_type_dir.is_dir():
                continue
            for class_dir in sorted(aug_type_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if class_name not in class_to_idx:
                    continue
                label = class_to_idx[class_name]
                for img_path in sorted(class_dir.glob("*.jpg")):
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} augmented images from {aug_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)

        # Standardization: EXIF orientation fix + RGB conversion
        image = fix_orientation([image])[0]
        image = convert_to_rgb([image])[0]

        if self.transform:
            image = self.transform(image)

        return image, label


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def _build_class_to_idx() -> dict:
    """
    Build class_name -> local_index mapping from labels.json.
    """
    import json
    labels_path = DATA_ROOT / "labels.json"
    with open(labels_path, "r") as f:
        idx_to_name = json.load(f)
    # labels.json maps "0" -> "goldfish", "1" -> "bald_eagle", etc.
    return {name: int(idx) for idx, name in idx_to_name.items()}


def _collect_all_samples(train_dir: Path, class_to_idx: dict) -> List[Tuple[Path, int]]:
    """
    Collect all (image_path, label) pairs from the training directory.
    """
    samples = []
    for class_name, label in sorted(class_to_idx.items()):
        class_dir = train_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for img_path in sorted(class_dir.glob("*.jpg")):
            samples.append((img_path, label))
    return samples


def stratified_split(
    samples: List[Tuple[Path, int]],
    ratios: Dict[str, float] = None,
    seed: int = SPLIT_SEED,
) -> Dict[str, List[Tuple[Path, int]]]:
    """
    Perform a stratified multi-way split. Within each class, randomly
    partition images according to the given ratios.

    Args:
        samples: All (path, label) pairs.
        ratios:  Dict mapping split name -> fraction (must sum to 1.0).
        seed:    Random seed for reproducibility.

    Returns:
        Dict mapping split name -> list of (path, label) pairs.
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    # Group by label
    by_class: Dict[int, list] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    split_names = list(ratios.keys())

    result = {name: [] for name in split_names}

    for label in sorted(by_class.keys()):
        class_samples = by_class[label]
        rng.shuffle(class_samples)

        n = len(class_samples)
        cursor = 0

        for i, name in enumerate(split_names):
            if i == len(split_names) - 1:
                # Last split gets the remainder
                result[name].extend(class_samples[cursor:])
            else:
                count = round(n * ratios[name])
                result[name].extend(class_samples[cursor:cursor + count])
                cursor += count

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def get_data_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    ratios: Dict[str, float] = None,
    seed: int = SPLIT_SEED,
    augmentation_level: str = "aggressive",
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Build and return five DataLoaders:
      1. train_loader  — 70% training set with augmentations (700/class)
      2. dev1_loader   — 10% dev set 1 for tuning (100/class)
      3. dev2_loader   — 10% dev set 2 for tuning (100/class)
      4. test_loader   — 10% held-out test set (100/class)
      5. aug_loader    — provided augmented images for robustness testing

    Args:
        batch_size:         Batch size for all loaders.
        num_workers:        Number of data-loading workers.
        ratios:             Split ratios (default: 70/10/10/10).
        seed:               Random seed for the split.
        augmentation_level: One of "standard", "aggressive", "extreme".

    Returns:
        (train_loader, dev1_loader, dev2_loader, test_loader, aug_loader)
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    class_to_idx = _build_class_to_idx()

    # ── Collect and split ─────────────────────────────────────────────────
    all_samples = _collect_all_samples(TRAIN_DIR, class_to_idx)
    splits = stratified_split(all_samples, ratios, seed)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"Augmentation level: {augmentation_level}")
    for name, samples in splits.items():
        counts = {}
        for _, label in samples:
            counts[label] = counts.get(label, 0) + 1
        print(
            f"{name:>5}: {len(samples):>5} images  "
            f"(per class: min={min(counts.values())}, max={max(counts.values())})"
        )

    # ── Create datasets ───────────────────────────────────────────────────
    train_dataset = SplitImageDataset(
        splits["train"], transform=get_train_transform(level=augmentation_level)
    )
    dev1_dataset  = SplitImageDataset(splits["dev1"],  transform=get_val_transform())
    dev2_dataset  = SplitImageDataset(splits["dev2"],  transform=get_val_transform())
    test_dataset  = SplitImageDataset(splits["test"],  transform=get_val_transform())
    aug_dataset   = AugmentationDataset(AUG_DIR, class_to_idx, transform=get_val_transform())

    # ── Create DataLoaders ────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dev1_loader = DataLoader(dev1_dataset, **loader_kwargs)
    dev2_loader = DataLoader(dev2_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    aug_loader  = DataLoader(aug_dataset,  **loader_kwargs)

    return train_loader, dev1_loader, dev2_loader, test_loader, aug_loader


def preprocess_and_load(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    ratios: Dict[str, float] = None,
    seed: int = SPLIT_SEED,
    augmentation_level: str = "aggressive",
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Single entry-point that combines standardization + data processing.

    Runs the full pipeline:
      1. Validates the dataset (checks for corrupted files)
      2. Builds stratified splits
      3. Per-image standardization (EXIF fix, RGB conversion) inside Dataset
      4. Applies augmentation transforms (standard/aggressive/extreme)
      5. Returns ready-to-train DataLoaders

    This is the recommended function to call from train.py.

    Args:
        batch_size:         Batch size for all loaders.
        num_workers:        Number of data-loading workers.
        ratios:             Split ratios (default: 70/10/10/10).
        seed:               Random seed for the split.
        augmentation_level: One of "standard", "aggressive", "extreme".

    Returns:
        (train_loader, dev1_loader, dev2_loader, test_loader, aug_loader)
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    class_to_idx = _build_class_to_idx()

    # ── Collect all samples ───────────────────────────────────────────────
    all_samples = _collect_all_samples(TRAIN_DIR, class_to_idx)

    # ── Validate: open every image and check for corruption ───────────────
    print("Running corruption validation on dataset...")
    images_to_check = []
    for path, _ in all_samples:
        try:
            images_to_check.append(Image.open(path))
        except Exception:
            images_to_check.append(None)

    result = validate_images(images_to_check)

    if result.num_corrupted > 0:
        print(result.summary())
        # Keep only valid samples
        all_samples = [all_samples[i] for i in result.valid_indices]
        print(f"Kept {len(all_samples)} valid samples after filtering.")
    else:
        print(f"All {len(all_samples)} images passed validation.")

    # Close opened images to free memory
    del images_to_check

    # ── Split and build loaders (standardization runs per-image in Dataset) ─
    splits = stratified_split(all_samples, ratios, seed)

    print(f"\nAugmentation level: {augmentation_level}")
    for name, samples in splits.items():
        counts = {}
        for _, label in samples:
            counts[label] = counts.get(label, 0) + 1
        print(
            f"{name:>5}: {len(samples):>5} images  "
            f"(per class: min={min(counts.values())}, max={max(counts.values())})"
        )

    train_dataset = SplitImageDataset(
        splits["train"], transform=get_train_transform(level=augmentation_level)
    )
    dev1_dataset  = SplitImageDataset(splits["dev1"],  transform=get_val_transform())
    dev2_dataset  = SplitImageDataset(splits["dev2"],  transform=get_val_transform())
    test_dataset  = SplitImageDataset(splits["test"],  transform=get_val_transform())
    aug_dataset   = AugmentationDataset(AUG_DIR, class_to_idx, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dev1_loader = DataLoader(dev1_dataset, **loader_kwargs)
    dev2_loader = DataLoader(dev2_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    aug_loader  = DataLoader(aug_dataset,  **loader_kwargs)

    return train_loader, dev1_loader, dev2_loader, test_loader, aug_loader


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from standardization import (
        preprocess_batch as std_preprocess_batch,
        tensor_to_images,
        save_debug_grid,
    )

    TRAIN_PATH = Path(__file__).resolve().parent.parent.parent / "dataset" / "train"
    OUTPUT_DIR = Path(__file__).resolve().parent / "debug_output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    IMAGES_PER_CLASS = 2

    print("=" * 60)
    print("  Preprocessing Visual Inspection")
    print("=" * 60)

    # ── Step 1: Collect 2 images from each class ─────────────────────────
    raw_images = []
    image_labels = []

    class_dirs = sorted([d for d in TRAIN_PATH.iterdir() if d.is_dir()])
    print(f"\nFound {len(class_dirs)} classes in {TRAIN_PATH}\n")

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted(class_dir.glob("*.jpg"))[:IMAGES_PER_CLASS]

        if not image_files:
            image_files = sorted(class_dir.glob("*.png"))[:IMAGES_PER_CLASS]
        if not image_files:
            image_files = sorted(class_dir.glob("*.*"))[:IMAGES_PER_CLASS]

        for img_path in image_files:
            img = Image.open(img_path)
            raw_images.append(img)
            image_labels.append(class_name)
            print(f"  Loaded: {class_name}/{img_path.name}  ({img.size}, {img.mode})")

    print(f"\nTotal images loaded: {len(raw_images)}")

    # ── Step 2: Run the full standardization pipeline ────────────────────
    print("\nRunning preprocess_batch()...")
    batch = std_preprocess_batch(raw_images)
    print(f"Output tensor: shape={list(batch.shape)}, dtype={batch.dtype}, "
          f"range=[{batch.min():.3f}, {batch.max():.3f}]")

    # ── Step 3: Apply augmentation and save ───────────────────────────────
    AUG_OUTPUT_DIR = OUTPUT_DIR / "augmented"
    AUG_OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nApplying augmentations (standard level)...")
    aug_transform = get_train_transform(level="standard")
    preprocessed_pil = tensor_to_images(batch)

    aug_tensors = []
    for idx, (img, label) in enumerate(zip(preprocessed_pil, image_labels)):
        aug_tensor = aug_transform(img)          # PIL -> augmented tensor
        aug_tensors.append(aug_tensor)

        # Save each augmented image
        aug_pil = tensor_to_images(aug_tensor)[0]  # single tensor -> PIL
        filename = f"{idx:03d}_{label}_aug.jpg"
        aug_pil.save(AUG_OUTPUT_DIR / filename)
        print(f"  Saved: augmented/{filename}  ({aug_pil.size})")

    aug_batch = torch.stack(aug_tensors)
    aug_grid_path = str(AUG_OUTPUT_DIR / "debug_grid_augmented.jpg")
    save_debug_grid(aug_batch, path=aug_grid_path, max_images=40, cols=4)

    # ── Step 4: Convert back to images and save ──────────────────────────
    print(f"\nSaving preprocessed images to: {OUTPUT_DIR}")
    pil_images = tensor_to_images(batch)

    for idx, (img, label) in enumerate(zip(pil_images, image_labels)):
        filename = f"{idx:03d}_{label}.jpg"
        save_path = OUTPUT_DIR / filename
        img.save(save_path)
        print(f"  Saved: {filename}  ({img.size})")

    # ── Step 5: Save a combined debug grid ───────────────────────────────
    grid_path = str(OUTPUT_DIR / "debug_grid.jpg")
    save_debug_grid(batch, path=grid_path, max_images=40, cols=4)

    print(f"\n{'=' * 60}")
    print(f"  Done! Check the images in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
