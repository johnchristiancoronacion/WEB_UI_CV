"""Create, train, and evaluate the Lab 6 synthetic classroom-object detector."""

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET = HERE / 'dataset'
RUNS = HERE / 'runs'
CLASSES = ('blue-card', 'orange-badge')
SPLITS = {'train': 80, 'val': 16, 'test': 16}
CANVAS_SIZE = 256
SAMPLE_CONFIDENCE = 0.25


def background(rng):
    shade = rng.integers(55, 158, (1, 1, 3))
    variation = rng.normal(0, 3.5, (CANVAS_SIZE, CANVAS_SIZE, 1))
    light = np.linspace(-17, 24, CANVAS_SIZE).reshape(1, CANVAS_SIZE, 1)
    image = np.clip(shade + variation + light, 0, 255).astype(np.uint8)
    for _ in range(9):
        start = tuple(rng.integers(0, CANVAS_SIZE, 2).tolist())
        end = tuple(rng.integers(0, CANVAS_SIZE, 2).tolist())
        color = tuple(np.clip(shade[0, 0] + rng.integers(-16, 17), 0, 255).tolist())
        cv2.line(image, start, end, color, 1, cv2.LINE_AA)
    return image


def draw_card(image, x, y, width, height):
    cv2.rectangle(image, (x + 3, y + 4), (x + width + 2, y + height + 3), (55, 49, 42), -1)
    cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (164, 95, 29), -1)
    cv2.rectangle(image, (x + 3, y + 3), (x + width - 4, y + height - 4), (221, 133, 31), 2)
    cv2.rectangle(image, (x + 8, y + 9), (x + width - 9, y + height - 10), (233, 231, 222), -1)
    cv2.circle(image, (x + width // 4, y + height // 2), height // 7, (212, 142, 57), -1, cv2.LINE_AA)
    for offset in (3, 9, 15):
        cv2.line(image, (x + width // 2, y + 11 + offset),
                 (x + width - 13, y + 11 + offset), (196, 134, 81), 2)


def draw_badge(image, x, y, width, height):
    axes = (width // 2, height // 2)
    center = (x + axes[0], y + axes[1])
    cv2.ellipse(image, (center[0] + 3, center[1] + 4), axes, 0, 0, 360, (49, 53, 60), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, 0, 0, 360, (24, 113, 239), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, (axes[0] - 3, axes[1] - 3), 0, 0, 360, (79, 166, 254), 2, cv2.LINE_AA)
    cv2.ellipse(image, center, (axes[0] - 11, axes[1] - 7), 0, 0, 360, (165, 217, 254), -1, cv2.LINE_AA)
    cv2.line(image, (center[0] - 11, center[1]), (center[0] + 11, center[1]), (26, 95, 154), 2, cv2.LINE_AA)


def generate_image(rng, index):
    image = background(rng)
    class_ids = (0, 1) if index % 4 < 2 else (index % 2,)
    left_class = int(rng.integers(0, 2))
    rows = []
    for class_id in class_ids:
        width = int(rng.integers(64, 110) if class_id == 0 else rng.integers(58, 99))
        height = round(width * (0.68 if class_id == 0 else 0.79))
        if len(class_ids) == 1:
            x = int(rng.integers(10, CANVAS_SIZE - width - 10))
        elif class_id == left_class:
            x = int(rng.integers(9, CANVAS_SIZE // 2 - width - 3))
        else:
            x = int(rng.integers(CANVAS_SIZE // 2 + 3, CANVAS_SIZE - width - 9))
        y = int(rng.integers(12, CANVAS_SIZE - height - 12))
        draw = (draw_card, draw_badge)[class_id]
        draw(image, x, y, width, height)
        values = (x + width / 2, y + height / 2, width, height)
        rows.append(f'{class_id} ' + ' '.join(f'{value / CANVAS_SIZE:.7f}' for value in values))
    return image, '\n'.join(rows) + '\n'


def build_dataset(seed):
    yaml = DATASET / 'data.yaml'
    for (split, count), split_seed in zip(SPLITS.items(), np.random.SeedSequence(seed).spawn(len(SPLITS))):
        rng = np.random.default_rng(split_seed)
        images = DATASET / 'images' / split
        labels = DATASET / 'labels' / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image, annotation = generate_image(rng, index)
            name = f'scene-{index:03d}'
            if not cv2.imwrite(str(images / f'{name}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f'Could not write {name}.jpg')
            (labels / f'{name}.txt').write_text(annotation, encoding='utf-8')
    names = json.dumps(CLASSES)
    yaml.write_text(
        f'path: {json.dumps(DATASET.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\n'
        f'nc: {len(CLASSES)}\nnames: {names}\n',
        encoding='utf-8',
    )
    return yaml


def box_metrics(metrics):
    return {
        'precision': round(float(metrics.box.mp), 6),
        'recall': round(float(metrics.box.mr), 6),
        'map50': round(float(metrics.box.map50), 6),
        'map50_95': round(float(metrics.box.map), 6),
    }


def train(args, dataset):
    RUNS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(RUNS / 'config'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(2)
    default_weights = ROOT / 'models' / 'yolo11n.pt'
    pretrained = args.weights or (default_weights if default_weights.is_file() else 'yolo11n.pt')
    model = YOLO(pretrained)
    model.train(
        data=str(dataset), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, seed=args.seed, deterministic=True,
        project=str(RUNS), name='classroom-objects', plots=False,
        optimizer='AdamW', lr0=0.001, lrf=0.2, nbs=args.batch,
        warmup_epochs=0, mosaic=0, mixup=0,
        hsv_h=0.0, hsv_s=0.15, hsv_v=0.15, translate=0.08, scale=0.15,
        amp=False, patience=0,
    )
    best_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    trained = YOLO(best_path)
    common = dict(data=str(dataset), imgsz=args.imgsz, batch=args.batch, device=args.device,
                  workers=0, plots=False, project=str(RUNS))
    val = trained.val(split='val', name='validation', **common)
    test = trained.val(split='test', name='held-out-test', **common)
    sample = ROOT / 'assets' / 'cards.jpg'
    sample_prediction = trained.predict(str(sample), imgsz=args.imgsz, device=args.device,
                                        conf=SAMPLE_CONFIDENCE, iou=0.45, verbose=False)[0]
    sample_objects = [
        {'class_name': CLASSES[int(class_id)], 'confidence': round(float(confidence), 6)}
        for class_id, confidence in zip(sample_prediction.boxes.cls.tolist(), sample_prediction.boxes.conf.tolist())
    ]
    export = Path(trained.export(format='onnx', imgsz=args.imgsz, device='cpu', opset=12,
                                 simplify=False, dynamic=False, nms=False))
    model_path = ROOT / 'models' / 'lab6-custom.onnx'
    network = cv2.dnn.readNetFromONNX(str(export))
    network.setInput(np.zeros((1, 3, args.imgsz, args.imgsz), np.float32))
    if network.forward().shape[:2] != (1, 4 + len(CLASSES)):
        raise ValueError('The ONNX model must emit raw box coordinates and one score per class.')
    results = {
        'dataset': 'Synthetic classroom objects',
        'classes': list(CLASSES),
        'splits': SPLITS,
        'seed': args.seed,
        'epochs': args.epochs,
        'image_size': args.imgsz,
        'validation': box_metrics(val),
        'test': box_metrics(test),
        'sample': {'image': 'assets/cards.jpg', 'split': 'test', 'confidence_threshold': SAMPLE_CONFIDENCE,
                   'detections': sample_objects},
    }
    with TemporaryDirectory(dir=ROOT) as temporary:
        staged_model = Path(temporary) / model_path.name
        staged_results = Path(temporary) / 'results.json'
        shutil.copyfile(export, staged_model)
        staged_results.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        staged_model.replace(model_path)
        staged_results.replace(HERE / 'results.json')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, help='Path to a YOLO11n checkpoint. Defaults to the official weights.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=192)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=106)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dataset-only', action='store_true')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz < 32 or args.imgsz % 32:
        parser.error('Epochs and batch must be positive, and imgsz must be a positive multiple of 32.')
    if args.seed < 0:
        parser.error('Seed must be zero or positive.')
    if args.weights is not None and not args.weights.is_file():
        parser.error(f'Could not find the checkpoint {args.weights}.')
    dataset = build_dataset(args.seed)
    print(f'Dataset specification: {dataset}')
    shutil.copyfile(DATASET / 'images' / 'test' / 'scene-000.jpg', ROOT / 'assets' / 'cards.jpg')
    if not args.dataset_only:
        train(args, dataset)


if __name__ == '__main__':
    main()
