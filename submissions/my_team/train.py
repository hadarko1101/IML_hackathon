import random
import sys
from pathlib import Path

import joblib
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ModelArchitecture
from data_processing import get_data_loaders, get_train_transform


OUTPUT = TEAM_DIR / "weights.joblib"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
STEPS_PER_EPOCH = None
STANDARD_AUG_EPOCHS = 3
SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LOG_EVERY_BATCHES = 10
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0
DEV1_SCORE_WEIGHT = 0.55
DEV2_SCORE_WEIGHT = 0.25
AUG_SCORE_WEIGHT = 0.20

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, epoch: int):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = len(loader) if STEPS_PER_EPOCH is None else STEPS_PER_EPOCH
    data_iter = iter(loader)

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

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch: int, split_name: str):
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
                f"  epoch {epoch:02d} {split_name:<5} "
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


def set_train_augmentation(train_loader, epoch: int) -> str:
    if epoch <= STANDARD_AUG_EPOCHS:
        level = "standard"
    else:
        level = "aggressive"

    train_loader.dataset.transform = get_train_transform(level=level)
    return level


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
        augmentation_level="standard",
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

    best_score = -1.0
    print(
        "Checkpoint score: "
        f"{DEV1_SCORE_WEIGHT:.2f}*dev1 + "
        f"{DEV2_SCORE_WEIGHT:.2f}*dev2 + "
        f"{AUG_SCORE_WEIGHT:.2f}*aug"
    )
    if STEPS_PER_EPOCH is None:
        print(f"Training uses full epochs: {len(train_loader)} batches per epoch.")
    else:
        print(f"Training uses {STEPS_PER_EPOCH} batches per epoch.")

    for epoch in range(1, EPOCHS + 1):
        augmentation_level = set_train_augmentation(train_loader, epoch)
        print(
            f"\nStarting epoch {epoch:02d}/{EPOCHS} "
            f"with {augmentation_level} augmentation",
            flush=True,
        )
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )
        dev1_loss, dev1_accuracy = evaluate(
            model,
            dev1_loader,
            criterion,
            device,
            epoch,
            "dev1",
        )
        dev2_loss, dev2_accuracy = evaluate(
            model,
            dev2_loader,
            criterion,
            device,
            epoch,
            "dev2",
        )
        aug_loss, aug_accuracy = evaluate(
            model,
            aug_loader,
            criterion,
            device,
            epoch,
            "aug",
        )
        checkpoint_score = (
            DEV1_SCORE_WEIGHT * dev1_accuracy
            + DEV2_SCORE_WEIGHT * dev2_accuracy
            + AUG_SCORE_WEIGHT * aug_accuracy
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"dev1_loss={dev1_loss:.4f} "
            f"dev1_acc={dev1_accuracy:.4f} "
            f"dev2_loss={dev2_loss:.4f} "
            f"dev2_acc={dev2_accuracy:.4f} "
            f"aug_loss={aug_loss:.4f} "
            f"aug_acc={aug_accuracy:.4f} "
            f"score={checkpoint_score:.4f}"
        )

        scheduler.step()

        if checkpoint_score > best_score:
            best_score = checkpoint_score
            save_weights(model, OUTPUT)
            model.to(device)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
        EPOCHS,
        "test",
    )
    print(f"Best checkpoint score: {best_score:.4f}")
    print(f"Held-out test: loss={test_loss:.4f} acc={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
