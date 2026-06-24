import random
import sys
from pathlib import Path
from typing import Optional

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
from data_processing import get_data_loaders


DATA_ROOT = PROJECT_ROOT / "dataset" / "train_set" / "train"
OUTPUT = TEAM_DIR / "weights.joblib"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
STEPS_PER_EPOCH = 50
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

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def run_epoch(model, loader, data_iter, criterion, optimizer, device, epoch: int, steps: int):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx in range(1, steps + 1):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)

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

        if batch_idx == 1 or batch_idx % LOG_EVERY_BATCHES == 0 or batch_idx == steps:
            print(
                f"  epoch {epoch:02d} train "
                f"batch {batch_idx:03d}/{steps:03d} "
                f"loss={total_loss / total:.4f} "
                f"acc={correct / total:.4f}",
                flush=True,
            )

    return total_loss / total, correct / total, data_iter


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

    print("Loading data using data_processing.py...")
    train_loader, dev1_loader, dev2_loader, test_loader, aug_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation_level="aggressive"
    )
    
    validation_loader = dev1_loader

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
    train_iter = iter(train_loader)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nStarting epoch {epoch:02d}/{EPOCHS}", flush=True)
        train_loss, train_accuracy, train_iter = run_epoch(
            model,
            train_loader,
            train_iter,
            criterion,
            optimizer,
            device,
            epoch,
            STEPS_PER_EPOCH
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
