import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


KEYPOINT_ORDER = [
    "Right_Eye_Canthus",
    "Right_Eye_Lacrimal",
    "Nasion",
    "Left_Eye_Lacrimal",
    "Left_Eye_Canthus",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 model on Label Studio keypoint JSON data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train_config.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Path to the Label Studio JSON export.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing the image files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square input image size for ResNet50.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--weights",
        choices=["none", "imagenet"],
        default=None,
        help="Use ImageNet pretrained ResNet50 weights if available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the best checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    return parser.parse_args()


def load_config(config_path: Path):
    default_config = {
        "json_path": "project-2-at-2026-03-27-12-39-5aa274bc.json",
        "image_dir": ".",
        "epochs": 30,
        "batch_size": 8,
        "lr": 1e-4,
        "image_size": 224,
        "val_ratio": 0.2,
        "weights": "none",
        "output": "resnet50_keypoints_best.pth",
        "seed": 42,
    }

    if not config_path.exists():
        return default_config

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    default_config.update(loaded)
    return default_config


def merge_args_with_config(args):
    config = load_config(args.config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    config["json_path"] = Path(config["json_path"])
    config["image_dir"] = Path(config["image_dir"])
    config["output"] = Path(config["output"])
    config["config"] = args.config
    return argparse.Namespace(**config)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(image_dir: Path, item: dict) -> Path | None:
    raw_name = (item.get("file_upload") or item.get("data", {}).get("img", "")).split("/")[-1]
    candidates = [raw_name]
    if "-" in raw_name:
        candidates.append(raw_name.split("-", 1)[1])

    for candidate in candidates:
        path = image_dir / candidate
        if path.exists():
            return path
    return None


def extract_sample(item: dict, image_dir: Path):
    annotations = item.get("annotations") or []
    if not annotations:
        return None

    image_path = resolve_image_path(image_dir, item)
    if image_path is None:
        return None

    keypoints = {}
    for result in annotations[0].get("result", []):
        value = result.get("value", {})
        labels = value.get("keypointlabels", [])
        if not labels:
            continue
        label = labels[0]
        keypoints[label] = [float(value["x"]) / 100.0, float(value["y"]) / 100.0]

    if any(label not in keypoints for label in KEYPOINT_ORDER):
        return None

    target = []
    for label in KEYPOINT_ORDER:
        target.extend(keypoints[label])

    return {
        "image_path": image_path,
        "target": torch.tensor(target, dtype=torch.float32),
    }


class KeypointDataset(Dataset):
    def __init__(self, samples, image_size: int):
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return self.transform(image), sample["target"]


def build_model(use_imagenet: bool):
    weights = models.ResNet50_Weights.DEFAULT if use_imagenet else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, len(KEYPOINT_ORDER) * 2),
        nn.Sigmoid(),
    )
    return model


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(loader.dataset), 1)


def main():
    args = merge_args_with_config(parse_args())
    seed_everything(args.seed)

    data = json.loads(args.json_path.read_text())
    samples = []
    skipped = 0
    for item in data:
        sample = extract_sample(item, args.image_dir)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if len(samples) < 2:
        raise ValueError("Not enough usable samples were found in the JSON file.")

    dataset = KeypointDataset(samples, image_size=args.image_size)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size < 1:
        train_size = len(dataset) - 1
        val_size = 1

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(use_imagenet=args.weights == "imagenet").to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    print(f"Usable samples: {len(samples)}")
    print(f"Skipped samples: {skipped}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"- train_loss: {train_loss:.6f} "
            f"- val_loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "keypoint_order": KEYPOINT_ORDER,
                    "image_size": args.image_size,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
