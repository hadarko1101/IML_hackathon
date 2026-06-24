import random
import sys
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import HF_INDEX_TO_IDX, HF_INDEX_TO_NAME, TARGET_HF_INDICES
from model import ModelArchitecture


DATA_ROOT = PROJECT_ROOT / "train_set" / "train"
OUTPUT = TEAM_DIR / "weights.joblib"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.2
MAX_IMAGES_PER_CLASS = 50
LOG_EVERY_BATCHES = 10
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImagePathDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def list_class_images(class_dir: Path):
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_train_validation_split(
    root: Path,
    seed: int = SEED,
    max_images_per_class: int | None = MAX_IMAGES_PER_CLASS,
):
    if not root.exists():
        raise FileNotFoundError(f"Training folder not found: {root}")

    rng = random.Random(seed)
    train_samples = []
    validation_samples = []

    for hf_idx in sorted(TARGET_HF_INDICES):
        class_name = HF_INDEX_TO_NAME[hf_idx]
        class_dir = root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        image_paths = list_class_images(class_dir)
        if not image_paths:
            raise RuntimeError(f"No images found in class folder: {class_dir}")

        shuffled_paths = image_paths[:]
        rng.shuffle(shuffled_paths)

        if max_images_per_class is not None:
            shuffled_paths = shuffled_paths[:max_images_per_class]

        train_count = int(len(shuffled_paths) * (1.0 - VALIDATION_FRACTION))
        local_idx = HF_INDEX_TO_IDX[hf_idx]

        train_samples.extend((path, local_idx) for path in shuffled_paths[:train_count])
        validation_samples.extend((path, local_idx) for path in shuffled_paths[train_count:])

        print(
            f"{class_name:<16} train={train_count:<4} "
            f"validation={len(shuffled_paths) - train_count:<4}"
        )

    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)

    return train_samples, validation_samples


def create_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, validation_transform


def run_epoch(model, loader, criterion, optimizer, device, epoch: int):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if batch_idx == 1 or batch_idx % LOG_EVERY_BATCHES == 0 or batch_idx == len(loader):
            print(
                f"  epoch {epoch:02d} train "
                f"batch {batch_idx:03d}/{len(loader):03d} "
                f"loss={total_loss / total:.4f} "
                f"acc={correct / total:.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if batch_idx == 1 or batch_idx % LOG_EVERY_BATCHES == 0 or batch_idx == len(loader):
            print(
                f"  epoch {epoch:02d} valid "
                f"batch {batch_idx:03d}/{len(loader):03d} "
                f"loss={total_loss / total:.4f} "
                f"acc={correct / total:.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


def save_weights(model, output_path: Path) -> None:
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, output_path)
    print(f"Saved best weights to {output_path}")


def main():
    """
    Full training pipeline.

    This script creates weights.joblib from a deterministic 80/20 split of
    train_set/train.
    """
    seed_everything(SEED)

    print(f"Loading images from {DATA_ROOT}")
    if MAX_IMAGES_PER_CLASS is None:
        print("Using all images per class.")
    else:
        print(f"Using at most {MAX_IMAGES_PER_CLASS} images per class for this run.")

    train_samples, validation_samples = build_train_validation_split(DATA_ROOT)
    print(
        f"\nTotal split: train={len(train_samples)} "
        f"validation={len(validation_samples)}"
    )

    train_transform, validation_transform = create_transforms()
    train_dataset = ImagePathDataset(train_samples, transform=train_transform)
    validation_dataset = ImagePathDataset(
        validation_samples,
        transform=validation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = ModelArchitecture(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_validation_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nStarting epoch {epoch:02d}/{EPOCHS}", flush=True)
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            epoch,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )

        scheduler.step()

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_weights(model, OUTPUT)
            model.to(device)

    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")


if __name__ == "__main__":
    main()
