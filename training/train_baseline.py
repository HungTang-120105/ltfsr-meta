from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.baseline_model import BaselineClassifier


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline ResNet18 classifier on CIFAR-100-LT.")
    parser.add_argument("--config", type=Path, default=Path("./config/config_baseline.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("./results/baseline"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    return parser


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    return train_transform, test_transform


def make_loader(dataset: ImageFolder, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def maybe_limit_dataset(dataset: ImageFolder, max_samples: int | None) -> ImageFolder:
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    indices = list(range(max_samples))
    subset = torch.utils.data.Subset(dataset, indices)
    return subset  # type: ignore[return-value]


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy_from_logits(logits.detach(), targets)
        total_batches += 1

    return total_loss / max(total_batches, 1), total_accuracy / max(total_batches, 1)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        requested = torch.device(device_name)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("CUDA was requested but is not available; falling back to CPU.")
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    runtime_cfg = config.get("device", None)

    data_dir = args.data_dir or Path(dataset_cfg["data_dir"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else int(training_cfg["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training_cfg["batch_size"])
    num_workers = int(config.get("num_workers", 2))
    pretrained = args.pretrained if args.pretrained is not None else bool(model_cfg.get("pretrained", True))
    device = resolve_device(args.device or runtime_cfg)

    train_transform, test_transform = build_transforms()
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = make_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.max_train_samples is not None:
        train_dataset = maybe_limit_dataset(train_dataset, args.max_train_samples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    if args.max_test_samples is not None:
        test_dataset = maybe_limit_dataset(test_dataset, args.max_test_samples)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = BaselineClassifier(num_classes=int(model_cfg["num_classes"]), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        momentum=float(training_cfg.get("momentum", 0.9)),
        weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_accuracy = 0.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, None, device)
        scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": best_accuracy,
                    "config": config,
                },
                best_path,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f} | best={best_accuracy:.4f}"
        )

    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()