import argparse
from pathlib import Path

import cv2
import torch
import yaml
from PIL import Image
from torchvision import transforms

from train_resnet50_keypoints import KEYPOINT_ORDER, build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time webcam inference using the trained keypoint model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("webcam_config.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained .pth checkpoint.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--weights",
        choices=["none", "imagenet"],
        default=None,
        help="Backbone initialization mode used to build the model.",
    )
    parser.add_argument(
        "--face-scale",
        type=float,
        default=None,
        help="Expand detected face box by this scale before inference.",
    )
    parser.add_argument(
        "--detector-scale-factor",
        type=float,
        default=None,
        help="OpenCV face detector scaleFactor value.",
    )
    parser.add_argument(
        "--detector-min-neighbors",
        type=int,
        default=None,
        help="OpenCV face detector minNeighbors value.",
    )
    parser.add_argument(
        "--detector-min-size",
        type=int,
        default=None,
        help="Minimum detected face size in pixels.",
    )
    return parser.parse_args()


def load_config(config_path: Path):
    default_config = {
        "checkpoint": "resnet50_keypoints_best.pth",
        "camera_index": 0,
        "weights": "none",
        "face_scale": 1.35,
        "detector_scale_factor": 1.1,
        "detector_min_neighbors": 5,
        "detector_min_size": 80,
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

    config["checkpoint"] = Path(config["checkpoint"])
    config["config"] = args.config
    return argparse.Namespace(**config)


def load_checkpoint(checkpoint_path: Path, weights: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(use_imagenet=weights == "imagenet")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_size = checkpoint.get("image_size", 224)
    keypoint_order = checkpoint.get("keypoint_order", KEYPOINT_ORDER)
    return model, image_size, keypoint_order


def make_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def expand_box(x, y, w, h, scale, frame_width, frame_height):
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = max(w, h) * scale

    x1 = max(0, int(cx - size / 2.0))
    y1 = max(0, int(cy - size / 2.0))
    x2 = min(frame_width, int(cx + size / 2.0))
    y2 = min(frame_height, int(cy + size / 2.0))
    return x1, y1, x2, y2


def predict_keypoints(frame_bgr, box, model, transform, device, keypoint_order):
    x1, y1, x2, y2 = box
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return []

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor).squeeze(0).cpu().tolist()

    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    points = []
    for idx, label in enumerate(keypoint_order):
        px = preds[idx * 2] * crop_w + x1
        py = preds[idx * 2 + 1] * crop_h + y1
        points.append((label, int(px), int(py)))
    return points


def draw_predictions(frame, box, points):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

    for label, px, py in points:
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
        cv2.putText(
            frame,
            label,
            (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    args = merge_args_with_config(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, keypoint_order = load_checkpoint(args.checkpoint, args.weights, device)
    transform = make_transform(image_size)

    face_cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade for face detection.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    print(f"Keypoints: {', '.join(keypoint_order)}")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.detector_scale_factor,
                minNeighbors=args.detector_min_neighbors,
                minSize=(args.detector_min_size, args.detector_min_size),
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                box = expand_box(x, y, w, h, args.face_scale, frame.shape[1], frame.shape[0])
                points = predict_keypoints(frame, box, model, transform, device, keypoint_order)
                draw_predictions(frame, box, points)
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Keypoint Webcam Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
