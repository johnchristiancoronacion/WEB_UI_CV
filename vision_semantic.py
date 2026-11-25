"""Pixel-wise Cityscapes segmentation with a pretrained SegFormer-B0 model."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "segformer-cityscapes.onnx"
MODEL_URL = (
    "https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-512-1024/"
    "resolve/498e4add74041c22a9908388f600d836d57f06ea/onnx/model.onnx"
)

CLASSES = (
    "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic light", "Traffic sign", "Vegetation",
    "Terrain", "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle",
)
PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)[:, ::-1]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _session():
    if not MODEL_PATH.is_file():
        raise ValueError("SegFormer-B0 weights are missing. Run python app.py --download-segmentation, then try again.")
    try:
        import onnxruntime as runtime
    except ImportError as error:
        raise ValueError("Install the packages in requirements.txt to run semantic segmentation.") from error
    options = runtime.SessionOptions()
    options.intra_op_num_threads = 2
    return runtime.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])


def _class_map(scores, width, height):
    labels = np.zeros((height, width), np.uint8)
    best = cv2.resize(scores[0], (width, height), interpolation=cv2.INTER_LINEAR)
    for index, score in enumerate(scores[1:], start=1):
        candidate = cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR)
        improved = candidate > best
        labels[improved] = index
        best[improved] = candidate[improved]
    return labels


def _legend(visible, counts, total):
    width = 650
    chart = np.full((56 + 35 * ((len(visible) + 1) // 2), width, 3), (248, 245, 239), np.uint8)
    ink = (77, 52, 35)
    cv2.putText(chart, "Predicted class coverage", (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.64, ink, 1, cv2.LINE_AA)
    for row, index in enumerate(visible):
        x = 18 + row % 2 * (width // 2)
        y = 56 + row // 2 * 35
        color = tuple(int(channel) for channel in PALETTE[index])
        cv2.rectangle(chart, (x, y - 13), (x + 19, y + 6), color, cv2.FILLED)
        label = f"{CLASSES[index]}  {counts[index] / total:.1%}"
        cv2.putText(chart, label, (x + 30, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.54, ink, 1, cv2.LINE_AA)
    return chart


def segment(image):
    """Return a color-coded class mask, overlay, and detected class proportions."""
    session = _session()
    resized = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    tensor = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])
    scores = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    if scores.ndim != 4 or scores.shape[0:2] != (1, len(CLASSES)):
        raise ValueError("The semantic model must provide 19 Cityscapes class scores per pixel.")
    height, width = image.shape[:2]
    scale = min(1, 1024 / max(width, height))
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    labels = _class_map(scores[0], *display_size)
    mask = PALETTE[labels]
    display = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(display, 0.47, mask, 0.53, 0)
    counts = np.bincount(labels.ravel(), minlength=len(CLASSES))
    visible = sorted(np.flatnonzero(counts), key=lambda index: -counts[index])
    legend = _legend(visible, counts, labels.size)
    return [("Semantic class mask", mask), ("Semantic class overlay", overlay), ("Predicted classes", legend)], {
        "Method": "SegFormer-B0 · Cityscapes",
        "Classes predicted": len(visible),
        "Largest predicted class": f"{CLASSES[visible[0]]} ({counts[visible[0]] / labels.size:.1%})",
        "Model input": "1024 × 512 px",
    }
