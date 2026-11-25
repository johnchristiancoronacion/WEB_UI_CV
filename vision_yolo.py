from pathlib import Path
from threading import Lock

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'yolo11n.onnx'
MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx'
CUSTOM_PATH = MODEL_PATH.with_name('lab6-custom.onnx')
CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)
MODELS = {
    'pretrained': (MODEL_PATH, 640, CLASSES, 'YOLO11n · COCO'),
    'custom': (CUSTOM_PATH, 192, ('blue-card', 'orange-badge'), 'YOLO11n · Custom cards'),
}
COLORS = [(223, 151, 96), (91, 224, 159), (224, 132, 204), (122, 194, 245), (222, 184, 129)]
_networks = {}
_network_lock = Lock()


def model_configuration(options):
    model = options.get('model', 'pretrained')
    if model not in MODELS:
        raise ValueError('Choose the pretrained or custom YOLO11 model.')
    return MODELS[model]


def session_for(model_path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ValueError('Install the project requirements to use YOLO11 inference.') from error

    with _network_lock:
        if model_path not in _networks:
            if not model_path.is_file():
                setup = 'python -m training.train' if model_path == CUSTOM_PATH else 'python app.py --download-model'
                raise ValueError(f'YOLO11n weights are missing. Run {setup}, then try again.')
            configuration = ort.SessionOptions()
            configuration.intra_op_num_threads = 3
            _networks[model_path] = ort.InferenceSession(str(model_path), configuration, providers=['CPUExecutionProvider'])
        return _networks[model_path]


def detect(image, options):
    model_path, input_size, model_classes, _ = model_configuration(options)
    confidence = float(options.get('confidence', 25)) / 100
    overlap = float(options.get('iou', 45)) / 100
    if not 0.01 <= confidence <= 0.99 or not 0.01 <= overlap <= 0.99:
        raise ValueError('Confidence and NMS overlap must be between 1% and 99%.')

    session = session_for(model_path)
    image_input = session.get_inputs()[0]
    if isinstance(image_input.shape[-1], int) and image_input.shape[-2] == image_input.shape[-1]:
        input_size = image_input.shape[-1]
    height, width = image.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = np.full((input_size, input_size, 3), 114, np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = cv2.resize(image, (resized_width, resized_height))

    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (input_size, input_size), swapRB=True)
    predictions = session.run(None, {image_input.name: blob})[0]
    if predictions.ndim != 3 or predictions.shape[:2] != (1, 4 + len(model_classes)):
        raise ValueError('The YOLO11 weights must use the raw detection output format.')
    predictions = predictions[0].T
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    scores = np.max(predictions[:, 4:], axis=1)
    candidates = np.flatnonzero(scores > confidence)
    boxes = []
    classes = []
    confidences = []
    for index in candidates:
        x, y, box_width, box_height = predictions[index, :4]
        left = max(0, min(width - 1, round((x - box_width / 2 - pad_x) * width / resized_width)))
        top = max(0, min(height - 1, round((y - box_height / 2 - pad_y) * height / resized_height)))
        right = max(0, min(width - 1, round((x + box_width / 2 - pad_x) * width / resized_width)))
        bottom = max(0, min(height - 1, round((y + box_height / 2 - pad_y) * height / resized_height)))
        if right <= left or bottom <= top:
            continue
        boxes.append([left, top, right - left, bottom - top])
        classes.append(int(class_ids[index]))
        confidences.append(float(scores[index]))

    selected = cv2.dnn.NMSBoxesBatched(boxes, confidences, classes, confidence, overlap) if boxes else []
    thickness = max(1, round(max(width, height) / 420))
    font_scale = max(0.38, min(1.3, max(width, height) / 1250))
    output = image.copy()
    detections = []
    for index in np.asarray(selected, dtype=int).ravel()[:100]:
        left, top, box_width, box_height = boxes[index]
        class_id = classes[index]
        name = model_classes[class_id]
        score = confidences[index]
        color = COLORS[class_id % len(COLORS)]
        label = f'{name} {score:.0%}'
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_top = max(0, top - label_height - baseline - 5)
        cv2.rectangle(output, (left, top), (left + box_width, top + box_height), color, thickness)
        cv2.rectangle(output, (left, label_top), (left + label_width + 8, label_top + label_height + baseline + 5), color, -1)
        cv2.putText(output, label, (left + 4, label_top + label_height + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (12, 12, 12), thickness, cv2.LINE_AA)
        detections.append({'class_name': name, 'confidence': round(score, 4), 'box': boxes[index]})
    return output, detections
