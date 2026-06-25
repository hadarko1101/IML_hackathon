import argparse
import csv
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

from model import MODEL_REGISTRY, build_model
from data_processing import get_data_loaders, get_train_transform


OUTPUT = TEAM_DIR / "weights.joblib"
METRICS_OUTPUT = TEAM_DIR / "training_metrics.csv"

# IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 25
STEPS_PER_EPOCH = 200
STANDARD_AUG_EPOCHS = 5
SEED = 42
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
LOG_EVERY_BATCHES = 10
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0
DEV1_SCORE_WEIGHT = 0.55
DEV2_SCORE_WEIGHT = 0.25
AUG_SCORE_WEIGHT = 0.20
ROBUST_EVAL_EVERY = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int, use_amp: bool):
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

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

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
def evaluate(model, loader, criterion, device, epoch: int, split_name: str, use_amp: bool):
    if len(loader.dataset) == 0:
        print(f"  epoch {epoch:02d} {split_name:<5} skipped (empty split)")
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="balanced_resnet",
        help="Architecture variant to train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Where to save the best model state_dict.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=METRICS_OUTPUT,
        help="CSV file for per-epoch metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for all loaders.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=STEPS_PER_EPOCH,
        help="Limit train batches per epoch. Omit for full epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers. Use 0 on Windows if multiprocessing is unstable.",
    )
    parser.add_argument(
        "--robust-eval-every",
        type=int,
        default=ROBUST_EVAL_EVERY,
        help="Evaluate dev2/aug every N epochs. dev1 is evaluated every epoch.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision.",
    )
    return parser.parse_args()


def append_metrics(metrics_path: Path, row: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    """
    Full training pipeline.

    This script creates weights.joblib from a deterministic 80/20 split of
    train_set/train.
    """
    args = parse_args()
    seed_everything(SEED)

    print("Loading data using data_processing.py...")
    train_loader, dev1_loader, dev2_loader, test_loader, aug_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_level="standard",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Training on {device}")
    print(f"Mixed precision: {'on' if use_amp else 'off'}")
    print(f"Training model: {args.model}")
    print(f"Best weights output: {args.output}")
    print(f"Metrics output: {args.metrics_output}")

    model = build_model(args.model, num_classes=20).to(device)
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=MIN_LEARNING_RATE,
    )

    best_score = -1.0
    print(
        "Checkpoint score: "
        f"{DEV1_SCORE_WEIGHT:.2f}*dev1 + "
        f"{DEV2_SCORE_WEIGHT:.2f}*dev2 + "
        f"{AUG_SCORE_WEIGHT:.2f}*aug"
    )
    if args.steps_per_epoch is None:
        print(f"Training uses full epochs: {len(train_loader)} batches per epoch.")
    else:
        print(f"Training uses {args.steps_per_epoch} batches per epoch.")
    print(f"dev2/aug evaluation cadence: every {args.robust_eval_every} epoch(s).")

    global STEPS_PER_EPOCH
    STEPS_PER_EPOCH = args.steps_per_epoch
    dev2_loss = None
    dev2_accuracy = None
    aug_loss = None
    aug_accuracy = None

    for epoch in range(1, args.epochs + 1):
        augmentation_level = set_train_augmentation(train_loader, epoch)
        print(
            f"\nStarting epoch {epoch:02d}/{args.epochs} "
            f"with {augmentation_level} augmentation",
            flush=True,
        )
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            use_amp,
        )
        dev1_loss, dev1_accuracy = evaluate(
            model,
            dev1_loader,
            criterion,
            device,
            epoch,
            "dev1",
            use_amp,
        )
        run_robust_eval = (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.robust_eval_every == 0
        )
        if run_robust_eval:
            dev2_loss, dev2_accuracy = evaluate(
                model,
                dev2_loader,
                criterion,
                device,
                epoch,
                "dev2",
                use_amp,
            )
            aug_loss, aug_accuracy = evaluate(
                model,
                aug_loader,
                criterion,
                device,
                epoch,
                "aug",
                use_amp,
            )
        else:
            print(
                f"  epoch {epoch:02d} dev2/aug skipped "
                f"(using previous values for checkpoint score)",
                flush=True,
            )

        assert dev2_loss is not None
        assert dev2_accuracy is not None
        assert aug_loss is not None
        assert aug_accuracy is not None
        checkpoint_score = (
            DEV1_SCORE_WEIGHT * dev1_accuracy
            + DEV2_SCORE_WEIGHT * dev2_accuracy
            + AUG_SCORE_WEIGHT * aug_accuracy
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
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

        append_metrics(
            args.metrics_output,
            {
                "model": args.model,
                "epoch": epoch,
                "augmentation": augmentation_level,
                "lr": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "dev1_loss": dev1_loss,
                "dev1_acc": dev1_accuracy,
                "dev2_loss": dev2_loss,
                "dev2_acc": dev2_accuracy,
                "aug_loss": aug_loss,
                "aug_acc": aug_accuracy,
                "score": checkpoint_score,
                "best_score_so_far": max(best_score, checkpoint_score),
            },
        )

        scheduler.step()

        if checkpoint_score > best_score:
            best_score = checkpoint_score
            save_weights(model, args.output)
            model.to(device)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
        args.epochs,
        "test",
        use_amp,
    )
    print(f"Best checkpoint score: {best_score:.4f}")
    print(f"Held-out test: loss={test_loss:.4f} acc={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
