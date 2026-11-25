"""Computer vision operations for the SURF, PCA, and segmentation labs."""

import math
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


def _number(options, key, default, minimum, maximum):
    try:
        value = float(options.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(maximum, max(minimum, value))


def _enabled(options, key):
    return str(options.get(key, False)).lower() in {"true", "on", "1"}


def _fit(image, limit):
    height, width = image.shape[:2]
    if max(height, width) <= limit:
        return image
    scale = limit / max(height, width)
    return cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


# Box approximations of SURF's second-order Gaussian filters at the base size.
# Each rectangle records its left, top, right, bottom, and signed weight.
_SURF_XX = ((0, 2, 3, 7, 1), (3, 2, 6, 7, -2), (6, 2, 9, 7, 1))
_SURF_YY = tuple((top, left, bottom, right, weight) for left, top, right, bottom, weight in _SURF_XX)
_SURF_XY = ((1, 1, 4, 4, 1), (5, 1, 8, 4, -1), (1, 5, 4, 8, -1), (5, 5, 8, 8, 1))


def _rect_sum(integral, left, top, right, bottom):
    return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]


def _box_response(integral, x, y, size, pattern):
    response = np.zeros((len(y), len(x)), np.float32)
    for left, top, right, bottom, weight in pattern:
        left, top, right, bottom = (round(edge * size / 9) for edge in (left, top, right, bottom))
        sums = _rect_sum(integral, x + left, y[:, None] + top, x + right, y[:, None] + bottom)
        response += sums.astype(np.float32) * (weight / ((right - left) * (bottom - top) * 255))
    return response


def _hessian_layer(integral, size, step):
    height, width = (axis - 1 for axis in integral.shape)
    shape = ((height + step - 1) // step, (width + step - 1) // step)
    layer = np.full(shape, -np.inf, np.float32)
    half = size // 2
    first = (half + step - 1) // step
    y_stop = (height - size + half) // step + 1
    x_stop = (width - size + half) // step + 1
    if y_stop <= first or x_stop <= first:
        return layer

    y = np.arange(first, y_stop) * step - half
    x = np.arange(first, x_stop) * step - half
    xx = _box_response(integral, x, y, size, _SURF_XX)
    yy = _box_response(integral, x, y, size, _SURF_YY)
    xy = _box_response(integral, x, y, size, _SURF_XY)
    layer[first:y_stop, first:x_stop] = xx * yy - 0.81 * xy * xy
    return layer


def _surf_candidates(integral, threshold):
    kernel = np.ones((3, 3), np.uint8)
    neighbors = kernel.copy()
    neighbors[1, 1] = 0
    candidates = []
    for octave in range(4):
        step = 2**octave
        sizes = [3 + 6 * (layer + 1) * step for layer in range(4)]
        if sizes[2] > min(integral.shape) - 1:
            break
        layers = [_hessian_layer(integral, size, step) for size in sizes]
        for index in (1, 2):
            previous, current, following = layers[index - 1:index + 2]
            if not np.isfinite(following).any():
                continue
            peaks = (current >= threshold) & (current > cv2.dilate(current, neighbors))
            peaks &= current > cv2.dilate(previous, kernel)
            peaks &= current > cv2.dilate(following, kernel)
            peaks &= np.isfinite(following)
            ys, xs = np.nonzero(peaks)
            candidates.extend(
                (float(current[y, x]), x * step, y * step, sizes[index])
                for y, x in zip(ys, xs)
            )
    return sorted(candidates, reverse=True)


_ORIENTATION_Y, _ORIENTATION_X = np.mgrid[-6:7, -6:7]
_ORIENTATION_DISTANCE = _ORIENTATION_X**2 + _ORIENTATION_Y**2
_ORIENTATION_CIRCLE = _ORIENTATION_DISTANCE <= 36
_ORIENTATION_ANGLES = np.deg2rad(np.arange(0, 360, 5))[:, None]
_DESCRIPTOR_Y, _DESCRIPTOR_X = np.mgrid[-9.5:10.5, -9.5:10.5]
_DESCRIPTOR_WEIGHTS = np.exp(-(_DESCRIPTOR_X**2 + _DESCRIPTOR_Y**2) / (2 * 3.3**2))


def _surf_orientation(integral, x, y, size):
    scale = size * 1.2 / 9
    half = max(1, round(2 * scale))
    wave_x = np.rint(x + _ORIENTATION_X * scale).astype(int)
    wave_y = np.rint(y + _ORIENTATION_Y * scale).astype(int)
    height, width = (axis - 1 for axis in integral.shape)
    valid = _ORIENTATION_CIRCLE & (wave_x >= half) & (wave_x <= width - half)
    valid &= (wave_y >= half) & (wave_y <= height - half)
    if not valid.any():
        return -1.0
    wave_x, wave_y = wave_x[valid], wave_y[valid]
    weights = np.exp(-_ORIENTATION_DISTANCE[valid] / (2 * 2.5**2))
    left = _rect_sum(integral, wave_x - half, wave_y - half, wave_x, wave_y + half)
    right = _rect_sum(integral, wave_x, wave_y - half, wave_x + half, wave_y + half)
    top = _rect_sum(integral, wave_x - half, wave_y - half, wave_x + half, wave_y)
    bottom = _rect_sum(integral, wave_x - half, wave_y, wave_x + half, wave_y + half)
    gradient_x = (right - left) * weights
    gradient_y = (bottom - top) * weights
    angles = np.arctan2(gradient_y, gradient_x)
    differences = (angles - _ORIENTATION_ANGLES + np.pi) % (2 * np.pi) - np.pi
    sectors = np.abs(differences) <= np.pi / 6
    sums_x, sums_y = sectors @ gradient_x, sectors @ gradient_y
    strongest = np.argmax(sums_x**2 + sums_y**2)
    return math.degrees(math.atan2(sums_y[strongest], sums_x[strongest])) % 360


def _surf_descriptor(integral, x, y, size, angle):
    scale = size * 1.2 / 9
    half = max(1, round(scale))
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    wave_x = np.rint(x + (_DESCRIPTOR_X * cosine - _DESCRIPTOR_Y * sine) * scale).astype(int)
    wave_y = np.rint(y + (_DESCRIPTOR_X * sine + _DESCRIPTOR_Y * cosine) * scale).astype(int)
    height, width = (axis - 1 for axis in integral.shape)
    valid = (wave_x >= half) & (wave_x <= width - half) & (wave_y >= half) & (wave_y <= height - half)
    wave_x, wave_y = wave_x[valid], wave_y[valid]
    left = _rect_sum(integral, wave_x - half, wave_y - half, wave_x, wave_y + half)
    right = _rect_sum(integral, wave_x, wave_y - half, wave_x + half, wave_y + half)
    top = _rect_sum(integral, wave_x - half, wave_y - half, wave_x + half, wave_y)
    bottom = _rect_sum(integral, wave_x - half, wave_y, wave_x + half, wave_y + half)
    grad_x, grad_y = right - left, bottom - top
    dx = np.zeros((20, 20), np.float32)
    dy = np.zeros_like(dx)
    dx[valid] = (cosine * grad_x + sine * grad_y) * _DESCRIPTOR_WEIGHTS[valid]
    dy[valid] = (-sine * grad_x + cosine * grad_y) * _DESCRIPTOR_WEIGHTS[valid]
    cells = np.stack((dx, dy, np.abs(dx), np.abs(dy)), axis=-1)
    vector = cells.reshape(4, 5, 4, 5, 4).sum(axis=(1, 3)).ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _surf_features(image, threshold, limit, upright):
    working = _fit(image, 960)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    integral = cv2.integral(gray)
    candidates = _surf_candidates(integral, threshold / 100)
    scale_x = image.shape[1] / working.shape[1]
    scale_y = image.shape[0] / working.shape[0]
    keypoints = []
    descriptors = []
    for response, x, y, size in candidates[:limit]:
        angle = -1.0 if upright else _surf_orientation(integral, x, y, size)
        keypoints.append(cv2.KeyPoint(x * scale_x, y * scale_y, size * scale_x, angle, response))
        descriptors.append(_surf_descriptor(integral, x, y, size, max(0, angle)))
    return keypoints, np.asarray(descriptors, dtype=np.float32).reshape(-1, 64), len(candidates)


def _comparison_image(image):
    working = _fit(image, 960)
    height, width = working.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 12, 0.9)
    return cv2.warpAffine(working, transform, (width, height), borderMode=cv2.BORDER_REFLECT)


def _builtin_features(image, detector, limit):
    working = _fit(image, 960)
    if min(working.shape[:2]) < 2:
        return [], None, 0
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    points, descriptors = detector.detectAndCompute(gray, None)
    scale_x = image.shape[1] / working.shape[1]
    scale_y = image.shape[0] / working.shape[0]
    mapped = [
        cv2.KeyPoint(point.pt[0] * scale_x, point.pt[1] * scale_y, point.size * scale_x,
                     point.angle, point.response, point.octave, point.class_id)
        for point in points[:limit]
    ]
    return mapped, descriptors[:limit] if descriptors is not None else None, len(points)


def _best_matches(first, second, norm):
    if first is None or second is None or len(first) == 0 or len(second) < 2:
        return []
    match_pairs = cv2.BFMatcher(norm).knnMatch(first, second, k=2)
    filtered = [pair[0] for pair in match_pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    return sorted(filtered, key=lambda match: match.distance)


def _surf(image, options):
    threshold = _number(options, "threshold", 1.0, 0.1, 20)
    limit = int(_number(options, "max_features", 150, 10, 500))
    upright = _enabled(options, "upright")
    source = _fit(image, 960)
    supplied = options.get("match_image")
    comparison = _fit(supplied, 960) if supplied is not None else _comparison_image(source)
    sift = cv2.SIFT_create(nfeatures=limit)
    orb = cv2.ORB_create(nfeatures=limit)

    def surf_features(target):
        return _surf_features(target, threshold, limit, upright)

    overlays, match_views = [], []
    metrics = {
        "Hessian threshold": f"{threshold:g}%",
        "Orientation": "Upright" if upright else "Haar-wavelet",
        "Comparison image": "Second upload" if supplied is not None else "Sample rotated 12° and scaled to 90%",
    }
    detectors = (
        ("SURF", surf_features, cv2.NORM_L2, (66, 240, 180)),
        ("SIFT", lambda target: _builtin_features(target, sift, limit), cv2.NORM_L2, (235, 181, 99)),
        ("ORB", lambda target: _builtin_features(target, orb, limit), cv2.NORM_HAMMING, (178, 118, 240)),
    )
    for name, extract, norm, color in detectors:
        started = time.perf_counter()
        keypoints, descriptors, count = extract(source)
        other_keypoints, other_descriptors, _ = extract(comparison)
        elapsed = 1000 * (time.perf_counter() - started)
        matches = _best_matches(descriptors, other_descriptors, norm)
        overlay = cv2.drawKeypoints(source, keypoints, None, color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        match_view = cv2.drawMatches(
            source, keypoints, comparison, other_keypoints, matches[:35], None, matchColor=color,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        overlays.append((f"{name} keypoints", overlay))
        match_views.append((f"{name} matches", match_view))
        metrics[f"{name} keypoints / matches"] = f"{len(keypoints)} displayed ({count} detected) / {len(matches)}"
        metrics[f"{name} extraction time"] = f"{elapsed:.1f} ms for both images"
    return overlays + match_views, metrics


def _pca(image, options):
    requested = int(_number(options, "components", 20, 1, 256))
    gray = _fit(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 320)
    data = gray.astype(np.float32)
    mean = data.mean(axis=0)
    centered = data - mean
    singular, left, right = cv2.SVDecomp(centered)
    maximum = min(gray.shape)
    retained = min(requested, maximum)
    reconstructed = (left[:, :retained] * singular[:retained].ravel()) @ right[:retained] + mean
    output = np.rint(reconstructed).clip(0, 255).astype(np.uint8)
    squared = singular.ravel().astype(np.float64) ** 2
    energy = squared.sum()
    variance = squared[:retained].sum() / energy if energy else 1.0
    difference = cv2.absdiff(gray, output)
    mse = float(np.mean(difference.astype(np.float32) ** 2))
    psnr = f"{10 * math.log10(255**2 / mse):.2f} dB" if mse else "Perfect reconstruction"
    return [("Grayscale input", gray), ("PCA reconstruction", output), ("Absolute error", difference)], {
        "Components retained": f"{retained} / {maximum}",
        "Variance retained": f"{variance:.2%}",
        "Mean squared error": round(mse, 2),
        "PSNR": psnr,
        "Analysis dimensions": f"{gray.shape[1]} × {gray.shape[0]} px",
    }


_LFW_DATA = Path(__file__).resolve().parent / "models" / "scikit-data"
_LFW_ARCHIVE = _LFW_DATA.parent / "lfw-70.npz"
_EIGENFACES_LOCK = Lock()


def _fetch_lfw(download):
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError as error:
        raise ValueError("Install the packages in requirements.txt to run the Eigenfaces experiment.") from error
    return fetch_lfw_people(
        data_home=str(_LFW_DATA), min_faces_per_person=70, resize=0.4, download_if_missing=download,
    )


def download_lfw():
    """Prepare the exercise's official LFW dataset in the local model cache."""
    if not _LFW_ARCHIVE.is_file():
        faces = _fetch_lfw(download=True)
        np.savez_compressed(_LFW_ARCHIVE, images=faces.images, target=faces.target, target_names=faces.target_names)
    return _LFW_ARCHIVE


@lru_cache(maxsize=1)
def _lfw_arrays():
    if _LFW_ARCHIVE.is_file():
        with np.load(_LFW_ARCHIVE, allow_pickle=False) as faces:
            return faces["images"], faces["target"], faces["target_names"]
    try:
        faces = _fetch_lfw(download=False)
    except OSError as error:
        raise ValueError("LFW is missing. Run python app.py --download-lfw, then try again.") from error
    return faces.images, faces.target, faces.target_names


def _portrait_gallery(portraits, titles, normalized=False):
    columns = min(4, len(portraits))
    rows = math.ceil(len(portraits) / columns)
    cell_width, cell_height = 162, 190
    gallery = np.full((rows * cell_height, columns * cell_width, 3), (248, 245, 239), np.uint8)
    for index, (portrait, labels) in enumerate(zip(portraits, titles)):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        if normalized:
            amount = max(1e-9, float(np.abs(portrait).max()))
            gray = np.uint8(np.clip((portrait / amount + 1) * 127.5, 0, 255))
        else:
            gray = np.uint8(np.clip(portrait, 0, 1) * 255)
        thumb = cv2.resize(gray, (120, 142), interpolation=cv2.INTER_CUBIC)
        gallery[y + 8:y + 150, x + 21:x + 141] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        for line, label in enumerate(labels):
            cv2.putText(gallery, label, (x + 8, y + 166 + line * 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.37, (77, 52, 35), 1, cv2.LINE_AA)
    return gallery


def _variance_plot(cumulative, retained, cutoff):
    plot = np.full((370, 720, 3), (248, 245, 239), np.uint8)
    left, right, top, bottom = 68, 680, 40, 302
    ink = (76, 48, 30)
    grid = (226, 220, 210)

    def position(count, fraction):
        return (round(left + count / len(cumulative) * (right - left)),
                round(bottom - fraction * (bottom - top)))

    for fraction in (0, 0.25, 0.5, 0.75, 0.9, 1):
        _, y = position(0, fraction)
        cv2.line(plot, (left, y), (right, y), grid, 1, cv2.LINE_AA)
        cv2.putText(plot, f"{fraction:.0%}", (13, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.47, ink, 1, cv2.LINE_AA)
    for count in (0, len(cumulative) // 4, len(cumulative) // 2, 3 * len(cumulative) // 4, len(cumulative)):
        x, _ = position(count, 0)
        cv2.putText(plot, str(count), (x - 9, bottom + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, ink, 1, cv2.LINE_AA)
    points = np.array([position(i, amount) for i, amount in enumerate(cumulative, start=1)], np.int32)
    cv2.polylines(plot, [points], False, (148, 116, 34), 3, cv2.LINE_AA)
    for count, color in ((retained, (213, 109, 77)), (cutoff, (121, 60, 160))):
        x, y = position(count, cumulative[count - 1])
        cv2.line(plot, (x, y), (x, bottom), color, 1, cv2.LINE_AA)
        cv2.circle(plot, (x, y), 4, color, -1, cv2.LINE_AA)
    cv2.putText(plot, f"Selected: {retained}", (178, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (213, 109, 77), 1, cv2.LINE_AA)
    cv2.putText(plot, f"90% variance: {cutoff}", (342, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (121, 60, 160), 1, cv2.LINE_AA)
    cv2.putText(plot, "Number of principal components", (208, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.54, ink, 1, cv2.LINE_AA)
    return plot


def _classification_table(report, names):
    headings = ("Identity", "Precision", "Recall", "F1 score", "Support")
    entries = [*names, "macro avg", "weighted avg"]
    x_positions = (24, 267, 370, 462, 556)
    chart = np.full((58 + 34 * len(entries), 670, 3), (248, 245, 239), np.uint8)
    for x, heading in zip(x_positions, headings):
        cv2.putText(chart, heading, (x, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (84, 55, 30), 1, cv2.LINE_AA)
    for row, name in enumerate(entries):
        y = 57 + row * 34
        scores = report[name]
        cv2.line(chart, (20, y - 21), (651, y - 21), (226, 220, 211), 1)
        values = (name, f"{scores['precision']:.3f}", f"{scores['recall']:.3f}",
                  f"{scores['f1-score']:.3f}", str(int(scores["support"])))
        for x, value in zip(x_positions, values):
            cv2.putText(chart, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (84, 55, 30), 1, cv2.LINE_AA)
    return chart


@lru_cache(maxsize=4)
def _eigenfaces(components):
    try:
        from scipy.linalg import eigvalsh
        from sklearn.decomposition import PCA
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise ValueError("Install the packages in requirements.txt to run the Eigenfaces experiment.") from error
    images, target, names = _lfw_arrays()
    height, width = images.shape[1:]
    flattened = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    train, test, train_labels, test_labels = train_test_split(flattened, target, test_size=0.2, random_state=42)
    retained = min(components, train.shape[0], train.shape[1])
    pca = PCA(n_components=retained, svd_solver="randomized", whiten=True, random_state=42)
    with threadpool_limits(limits=2, user_api="blas"):
        train_projection = pca.fit_transform(train)
        test_projection = pca.transform(test)
        classifier = SVC(kernel="rbf", class_weight="balanced").fit(train_projection, train_labels)
        predictions = classifier.predict(test_projection)
        centered = train - train.mean(axis=0)
        gram = centered @ centered.T
        power = np.maximum(0, eigvalsh(gram, driver="evd", overwrite_a=True, check_finite=False)[::-1])
    report = classification_report(test_labels, predictions, labels=np.arange(len(names)), target_names=names,
                                   output_dict=True, zero_division=0)
    fields = pca.components_.reshape(retained, height, width)
    leading = _portrait_gallery(fields[:12], [(f"Eigenface {i + 1}",) for i in range(min(12, retained))], True)
    outputs = [("Leading eigenfaces", leading)]
    if retained > 12:
        trailing = _portrait_gallery(fields[-3:], [(f"Eigenface {i + 1}",) for i in range(retained - 3, retained)], True)
        outputs.append(("Trailing eigenfaces", trailing))
    cumulative = power.cumsum(dtype=np.float64) / power.sum(dtype=np.float64)
    cutoff = int(np.searchsorted(cumulative, 0.9) + 1)
    outputs.extend((("Cumulative explained variance", _variance_plot(cumulative, retained, cutoff)),
                    ("Classification report", _classification_table(report, names))))
    misses = np.flatnonzero(predictions != test_labels)
    examples = misses[:8]
    if len(examples):
        titles = [(f"True: {names[test_labels[index]]}", f"Pred: {names[predictions[index]]}") for index in examples]
        errors = _portrait_gallery(test[examples].reshape(-1, height, width), titles)
        outputs.append(("Misclassified test faces", errors))
    return outputs, {
        "Dataset": "Labeled Faces in the Wild",
        "Samples / identities": f"{len(images)} / {len(names)}",
        "Features per face": f"{height * width:,} ({width} × {height} px)",
        "Training / test faces": f"{len(train)} / {len(test)}",
        "Whitened components": retained,
        "Variance retained": f"{pca.explained_variance_ratio_.sum():.2%}",
        "Components for 90% variance": cutoff,
        "Test accuracy": f"{np.mean(predictions == test_labels):.2%}",
        "Macro F1": round(report["macro avg"]["f1-score"], 3),
        "Misclassified faces": len(misses),
    }


def _pca_lab(image, options):
    if options.get("method", "eigenfaces") == "reconstruction":
        return _pca(image, options)
    retained = int(_number(options, "components", 150, 1, 200))
    with _EIGENFACES_LOCK:
        return _eigenfaces(retained)


_LABEL_COLORS = np.array([
    [79, 212, 100], [240, 159, 47], [75, 112, 240], [226, 145, 247],
    [70, 220, 247], [219, 211, 54], [76, 165, 247], [180, 120, 136],
], dtype=np.uint8)


def _color_labels(pixels, centers):
    distances = np.sum((pixels[:, None] - centers) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def _cluster_colors(image, options):
    count = int(_number(options, "clusters", 3, 2, 8))
    height, width = image.shape[:2]
    pixels = image.reshape(-1, 3)
    sample = pixels
    if len(sample) > 20_000:
        indices = np.random.default_rng(106).choice(len(sample), size=20_000, replace=False)
        sample = sample[indices]
    sample = sample.astype(np.float32)
    distinct = np.unique(sample, axis=0)
    count = min(count, len(distinct))
    seeds = [distinct[np.argmin(distinct.sum(axis=1))]]
    for _ in range(1, count):
        distance = np.min(np.sum((distinct[:, None] - seeds) ** 2, axis=2), axis=1)
        seeds.append(distinct[np.argmax(distance)])
    centers = np.asarray(seeds)
    if count > 1:
        labels = _color_labels(sample, centers).astype(np.int32)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 40, 0.25)
        _, _, centers = cv2.kmeans(sample, count, labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS)

    all_labels = np.concatenate([
        _color_labels(pixels[start:start + 50_000], centers)
        for start in range(0, len(pixels), 50_000)
    ]).reshape(height, width)
    quantized = np.rint(centers).clip(0, 255).astype(np.uint8)[all_labels]
    return [("Cluster colors", quantized), ("Region labels", _LABEL_COLORS[all_labels])], {
        "Method": "K-means color clustering",
        "Color clusters": len(np.unique(all_labels)),
    }


def _foreground(image, mask):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result


def _watershed(image, inverse):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    background = cv2.dilate(opened, kernel, iterations=2)
    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    seeds = np.uint8(distance > 0.45 * distance.max())
    count, markers = cv2.connectedComponents(seeds)
    markers += 1
    markers[(background != 0) & (seeds == 0)] = 0
    overlay = image.copy()
    if min(gray.shape) >= 3 and count > 1:
        cv2.watershed(image, markers)
        boundary = markers == -1
        boundary[[0, -1], :] = False
        boundary[:, [0, -1]] = False
        overlay[boundary] = (80, 100, 255)
    return [("Foreground seeds", seeds * 255), ("Watershed boundaries", overlay)], {
        "Method": "Marker-based watershed",
        "Seed regions": count - 1,
    }


def _region_growing(image, options):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x = int(_number(options, "seed_x", width // 2, 0, width - 1))
    y = int(_number(options, "seed_y", height // 2, 0, height - 1))
    tolerance = int(_number(options, "tolerance", 18, 0, 255))
    bordered = np.zeros((height + 2, width + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    pixels, _, mask, _ = cv2.floodFill(gray, bordered, (x, y), 0, tolerance, tolerance, flags)
    mask = mask[1:-1, 1:-1]
    overlay = image.copy()
    cv2.drawMarker(overlay, (x, y), (63, 240, 175), cv2.MARKER_CROSS, 17, 2, cv2.LINE_AA)
    return [("Seed location", overlay), ("Grown region mask", mask), ("Foreground cutout", _foreground(image, mask))], {
        "Method": "Seeded region growing",
        "Seed coordinates (x, y)": f"({x}, {y})",
        "Seed intensity": int(gray[y, x]),
        "Intensity tolerance": tolerance,
        "Foreground coverage": f"{pixels / gray.size:.2%}",
    }


def _edge_segments(image, options):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = _number(options, "low", 50, 0, 255)
    high = _number(options, "high", 150, 1, 255)
    if low > high:
        raise ValueError("The low Canny threshold must be at or below the high threshold.")
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(smooth, low, high, L2gradient=True)
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size_cutoff = max(16, gray.size * 0.0005)
    regions = [contour for contour in contours if cv2.contourArea(contour) >= size_cutoff]
    mask = np.zeros_like(gray)
    overlay = image.copy()
    cv2.drawContours(mask, regions, -1, 255, cv2.FILLED)
    cv2.drawContours(overlay, regions, -1, (60, 234, 171), 2, cv2.LINE_AA)
    return [("Canny edges", edges), ("Enclosed regions", mask), ("Contour overlay", overlay)], {
        "Method": "Edge-based contour segmentation",
        "Canny thresholds": f"{low:g} / {high:g}",
        "Enclosed regions": len(regions),
        "Foreground coverage": f"{np.count_nonzero(mask) / mask.size:.2%}",
    }


def _segmentation(image, options):
    method = options.get("method", "otsu")
    inverse = _enabled(options, "invert")
    if method == "kmeans":
        return _cluster_colors(image, options)
    if method == "watershed":
        return _watershed(image, inverse)
    if method == "region":
        return _region_growing(image, options)
    if method == "edges":
        return _edge_segments(image, options)
    if method == "semantic":
        from vision_semantic import segment
        return segment(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    if method == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 31, 5)
        metrics = {"Method": "Adaptive Gaussian threshold", "Neighborhood": "31 × 31 px"}
    else:
        threshold, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
        metrics = {"Method": "Otsu threshold", "Threshold": int(threshold)}
    metrics["Foreground coverage"] = f"{np.count_nonzero(mask) / mask.size:.2%}"
    return [("Segmentation mask", mask), ("Foreground cutout", _foreground(image, mask))], metrics


def process(lab, image, options):
    """Return labeled result images and display metrics for one advanced lab."""
    return {4: _surf, 5: _pca_lab, 7: _segmentation}[lab](image, options)
