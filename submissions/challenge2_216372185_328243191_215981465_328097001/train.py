import argparse
import csv
import random
import sys
import time
from pathlib import Path

import joblib
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import DEFAULT_MODEL_NAME, MODEL_REGISTRY, build_model
from data_processing import IMAGE_SIZE, RESIZE_SIZE, get_data_loaders, get_train_transform


OUTPUT = TEAM_DIR / "weights.joblib"
METRICS_OUTPUT = TEAM_DIR / "training_metrics_wide_224.csv"
CHECKPOINT_OUTPUT = TEAM_DIR / "training_checkpoint.joblib"

BATCH_SIZE = 64
EPOCHS = 35
STEPS_PER_EPOCH = 200
STANDARD_AUG_EPOCHS = 7
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
CHECKPOINT_EVERY_EPOCHS = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def save_weights(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = to_cpu(model.state_dict())
    joblib.dump(state_dict, output_path)
    print(f"Saved best weights to {output_path}")


def save_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch: int,
    best_score: float,
    best_weights_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_name": args.model,
        "model_state_dict": to_cpu(model.state_dict()),
        "optimizer_state_dict": to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": to_cpu(scheduler.state_dict()),
        "scaler_state_dict": to_cpu(scaler.state_dict()),
        "best_score": best_score,
        "best_weights_path": str(best_weights_path),
        "args": vars(args),
        "training_config": {
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "standard_aug_epochs": STANDARD_AUG_EPOCHS,
            "image_size": IMAGE_SIZE,
            "resize_size": RESIZE_SIZE,
        },
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": to_cpu(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
        },
    }
    joblib.dump(checkpoint, checkpoint_path)
    print(f"Saved training checkpoint to {checkpoint_path}")


def load_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    checkpoint = joblib.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    rng_state = checkpoint.get("rng_state", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    model.to(device)
    start_epoch = int(checkpoint["epoch"]) + 1
    best_score = float(checkpoint.get("best_score", -1.0))
    print(
        f"Resumed from {checkpoint_path} "
        f"at epoch {checkpoint['epoch']} with best_score={best_score:.4f}"
    )
    return start_epoch, best_score


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
        default=DEFAULT_MODEL_NAME,
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
        "--checkpoint-output",
        type=Path,
        default=CHECKPOINT_OUTPUT,
        help="Where to save a full checkpoint for continuing training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_EPOCHS,
        help="Save a full training checkpoint every N epochs. Use 0 to save only at the end.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a full checkpoint created by --checkpoint-output.",
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
    start_time = time.perf_counter()
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
    print(f"Image preprocessing: Resize({RESIZE_SIZE}) -> crop({IMAGE_SIZE})")
    print(f"Best weights output: {args.output}")
    print(f"Metrics output: {args.metrics_output}")
    print(f"Training checkpoint output: {args.checkpoint_output}")
    print(f"Training checkpoint cadence: every {args.checkpoint_every} epoch(s)")

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
    start_epoch = 1
    if args.resume is not None:
        start_epoch, best_score = load_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already finished epoch {start_epoch - 1}, "
                f"but --epochs is {args.epochs}. Use a larger --epochs value."
            )

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

    for epoch in range(start_epoch, args.epochs + 1):
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

        should_save_checkpoint = (
            args.checkpoint_every > 0
            and (epoch % args.checkpoint_every == 0 or epoch == args.epochs)
        )
        if should_save_checkpoint:
            save_training_checkpoint(
                args.checkpoint_output,
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                epoch,
                best_score,
                args.output,
            )

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
    if args.checkpoint_every == 0:
        save_training_checkpoint(
            args.checkpoint_output,
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            args.epochs,
            best_score,
            args.output,
        )
    elapsed_minutes = (time.perf_counter() - start_time) / 60
    print(f"Total running time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
