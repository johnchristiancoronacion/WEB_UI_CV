import argparse
import base64
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

import vision_advanced
import vision_basic
import vision_semantic
import vision_yolo


ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / 'assets' / 'street.jpg'
CUSTOM_SAMPLE = ROOT / 'assets' / 'cards.jpg'
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}
app = Flask(__name__, static_folder='web', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024


@app.get('/')
def index():
    return app.send_static_file('index.html')


@app.get('/sample.jpg')
def sample():
    return send_from_directory(SAMPLE.parent, SAMPLE.name)


@app.get('/custom-sample.jpg')
def custom_sample():
    return send_from_directory(CUSTOM_SAMPLE.parent, CUSTOM_SAMPLE.name)


@app.get('/api/lab-06/training')
def training_results():
    return send_from_directory(ROOT / 'training', 'results.json')


@app.get('/report')
@app.get('/report/<path:filename>')
def report(filename='index.html'):
    return send_from_directory(ROOT / 'report', filename)


def encode(image):
    success, data = cv2.imencode('.png', image)
    if not success:
        raise ValueError('The output image could not be encoded.')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def decode_image(file, default_path=SAMPLE):
    if file is None:
        image = cv2.imread(str(default_path))
        if image is None:
            raise ValueError('The built-in sample image is missing.')
        return image

    try:
        with Image.open(file.stream) as header:
            if header.width * header.height > 8_000_000:
                raise ValueError('Choose an image with 8 megapixels or fewer.')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('Choose a readable image such as JPEG, PNG, BMP or WebP.') from error

    file.stream.seek(0)
    image = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('The image could not be decoded.')
    return image


def processed_video(file, options):
    with TemporaryDirectory(prefix='csst106-') as temp:
        source = Path(temp) / ('source' + Path(file.filename).suffix.lower())
        target = Path(temp) / 'detections.webm'
        file.save(source)
        capture = cv2.VideoCapture(str(source))
        writer = None
        frames = 0
        detections = 0
        classes = {}
        first_output = None
        try:
            if not capture.isOpened():
                raise ValueError('OpenCV could not read this video. Try an MP4 or WebM file.')
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = 24.0
            stride = max(1, math.ceil(fps / 4))
            for index in range(math.ceil(fps * 12)):
                if not capture.grab():
                    break
                if index % stride:
                    continue
                success, frame = capture.retrieve()
                if not success:
                    break
                height, width = frame.shape[:2]
                scale = min(1.0, 854 / max(width, height))
                size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
                if frame.shape[1::-1] != size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                output, found = vision_yolo.detect(frame, options)
                if writer is None:
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'VP80'), fps / stride, size)
                    if not writer.isOpened():
                        raise ValueError('This OpenCV installation could not encode WebM video.')
                    first_output = output
                writer.write(output)
                frames += 1
                detections += len(found)
                for item in found:
                    name = item['class_name']
                    classes[name] = classes.get(name, 0) + 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if frames == 0:
            raise ValueError('No readable video frames were found.')
        return {
            'outputs': [{'label': 'First annotated frame', 'source': encode(first_output)}],
            'video': 'data:video/webm;base64,' + base64.b64encode(target.read_bytes()).decode('ascii'),
            'metrics': {
                'Processed frames': frames,
                'Sampling rate': f'{fps / stride:.1f} fps',
                'Detections across frames': detections,
                'Classes': ', '.join(f'{name} × {count}' for name, count in sorted(classes.items())) or 'None',
                'Model': vision_yolo.model_configuration(options)[3],
            },
        }


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error='The upload is too large. The request limit is 25 MB.'), 413


@app.post('/api/labs/<int:lab>')
def run_lab(lab):
    if lab not in range(1, 8):
        return jsonify(error='Choose Lab Activity 1 through 7.'), 404
    try:
        file = request.files.get('media')
        options = request.form.to_dict()
        if file and (Path(file.filename).suffix.lower() in VIDEO_SUFFIXES or file.mimetype.startswith('video/')):
            if lab != 6:
                raise ValueError('Video input is available in Lab Activity 06.')
            return jsonify(processed_video(file, options))

        is_custom = lab == 6 and options.get('model') == 'custom'
        uses_dataset = lab == 5 and options.get('method') == 'eigenfaces'
        image = decode_image(file, CUSTOM_SAMPLE if is_custom else SAMPLE)
        comparison = request.files.get('comparison')
        if lab == 4 and comparison:
            options['match_image'] = decode_image(comparison)
        if lab == 3 and file is None:
            image = cv2.convertScaleAbs(image, alpha=0.48, beta=63)

        if lab in (1, 2, 3):
            outputs, metrics = vision_basic.process(lab, image, options)
        elif lab == 6:
            annotated, detections = vision_yolo.detect(image, options)
            classes = sorted({item['class_name'] for item in detections})
            outputs = [('YOLO11 detections', annotated)]
            metrics = {'Objects detected': len(detections), 'Classes': ', '.join(classes) or 'None', 'Model': vision_yolo.model_configuration(options)[3]}
        else:
            outputs, metrics = vision_advanced.process(lab, image, options)

        height, width = image.shape[:2]
        response = {
            'outputs': [{'label': label, 'source': encode(output)} for label, output in outputs],
            'metrics': metrics,
        }
        if not uses_dataset and options.get('live') != '1':
            label = file.filename if file else 'Held-out card sample' if is_custom else 'Street scene sample'
            response['input'] = {'label': label, 'source': encode(image), 'width': width, 'height': height}
        if lab == 6:
            response['detections'] = detections
        return jsonify(response)
    except (ValueError, TypeError, OverflowError, cv2.error) as error:
        return jsonify(error=str(error)), 422


def download_model(target, url):
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temp:
        download = Path(temp) / target.name
        urlretrieve(url, download)
        download.replace(target)
    print(f'Model saved to {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSST106 computer vision laboratory')
    parser.add_argument('--port', type=int, default=8000)
    setup = parser.add_mutually_exclusive_group()
    setup.add_argument('--download-model', action='store_true', help='download the pretrained YOLO11n model')
    setup.add_argument('--download-lfw', action='store_true', help='cache the LFW face dataset for Lab 5')
    setup.add_argument('--download-segmentation', action='store_true', help='download SegFormer Cityscapes for Lab 7')
    args = parser.parse_args()
    if args.download_model:
        download_model(vision_yolo.MODEL_PATH, vision_yolo.MODEL_URL)
    elif args.download_lfw:
        vision_advanced.download_lfw()
    elif args.download_segmentation:
        download_model(vision_semantic.MODEL_PATH, vision_semantic.MODEL_URL)
    else:
        app.run(host='127.0.0.1', port=args.port)
