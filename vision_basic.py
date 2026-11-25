import math
import re

import cv2
import numpy as np


def _number(options, key, default, minimum, maximum):
    label = key.replace("_", " ").capitalize()
    try:
        value = float(options.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number.") from error
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _optional_integer(options, key, default, minimum, maximum):
    if options.get(key) in (None, ""):
        return default
    value = _number(options, key, default, minimum, maximum)
    if value != int(value):
        raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a whole number.")
    return int(value)


def _rotated_image(image, angle):
    right_angles = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        -90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        -180: cv2.ROTATE_180,
    }
    if angle in right_angles:
        return cv2.rotate(image, right_angles[angle])

    height, width = image.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    transform = cv2.getRotationMatrix2D(center, angle, 1)
    cosine = abs(transform[0, 0])
    sine = abs(transform[0, 1])
    rotated_width = max(1, math.ceil(width * cosine + height * sine))
    rotated_height = max(1, math.ceil(width * sine + height * cosine))
    transform[0, 2] += (rotated_width - width) / 2
    transform[1, 2] += (rotated_height - height) / 2
    return cv2.warpAffine(image, transform, (rotated_width, rotated_height))


def _pixel_comparison(before, after, x, y):
    radius, zoom = 6, 16
    left, right = max(0, x - radius), min(before.shape[1], x + radius + 1)
    top, bottom = max(0, y - radius), min(before.shape[0], y + radius + 1)
    panel_width, panel_height = (right - left) * zoom, (bottom - top) * zoom
    frame_width = max(120, panel_width) + 20
    chart = np.full((panel_height + 46, frame_width * 2, 3), (249, 247, 244), dtype=np.uint8)
    for index, (label, image) in enumerate((("Before", before), ("After", after))):
        start_x = index * frame_width + (frame_width - panel_width) // 2
        start_y = 32
        patch = image[top:bottom, left:right]
        panel = cv2.resize(patch, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)
        chart[start_y:start_y + panel_height, start_x:start_x + panel_width] = panel
        corner = (start_x + (x - left) * zoom, start_y + (y - top) * zoom)
        end = (corner[0] + zoom - 1, corner[1] + zoom - 1)
        cv2.rectangle(chart, corner, end, (25, 25, 25), 2)
        cv2.rectangle(chart, corner, end, (255, 255, 255), 1)
        cv2.putText(chart, label, (index * frame_width + 10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (59, 44, 24), 1, cv2.LINE_AA)
    return chart


def _images(image, options):
    scale = _number(options, "scale", 65, 10, 200) / 100
    angle = _number(options, "rotation", 90, -180, 180)
    crop = _number(options, "crop", 65, 10, 100) / 100
    flip = options.get("flip", "horizontal")
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if flip not in flip_codes:
        raise ValueError("Choose horizontal, vertical, or both flip axes.")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(image)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    rotated = _rotated_image(image, angle)
    rotated_height, rotated_width = rotated.shape[:2]

    crop_width = _optional_integer(options, "crop_width", max(1, round(width * crop)), 1, width)
    crop_height = _optional_integer(options, "crop_height", max(1, round(height * crop)), 1, height)
    left = _optional_integer(options, "crop_x", (width - crop_width) // 2, 0, width - 1)
    top = _optional_integer(options, "crop_y", (height - crop_height) // 2, 0, height - 1)
    if left + crop_width > width or top + crop_height > height:
        raise ValueError("The crop region must fit inside the image.")
    cropped = image[top:top + crop_height, left:left + crop_width]

    x = _optional_integer(options, "pixel_x", width // 2, 0, width - 1)
    y = _optional_integer(options, "pixel_y", height // 2, 0, height - 1)
    color = options.get("pixel_color") or "#ff00ff"
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("Enter the new pixel color as a six-digit RGB hex color.")
    new_pixel = tuple(int(color[index:index + 2], 16) for index in (5, 3, 1))
    old_pixel = tuple(int(channel) for channel in image[y, x])
    edited = image.copy()
    edited[y, x] = new_pixel

    outputs = [
        ("Grayscale", gray),
        ("Edited pixel enlarged", _pixel_comparison(image, edited, x, y)),
        ("Pixel-edited image", edited),
        ("Red channel", red),
        ("Green channel", green),
        ("Blue channel", blue),
        ("Resized image", resized),
        ("Rotated image", rotated),
        ("Flipped image", cv2.flip(image, flip_codes[flip])),
        ("Cropped region of interest", cropped),
    ]
    metrics = {
        "Original size": f"{width} × {height} px",
        "Array shape (height, width, BGR channels)": f"({height}, {width}, 3)",
        "Pixel coordinates (x, y)": f"({x}, {y})",
        "Original pixel (B, G, R)": str(old_pixel),
        "Edited pixel (B, G, R)": str(new_pixel),
        "Resized dimensions": f"{resized_width} × {resized_height} px",
        "Rotated canvas": f"{rotated_width} × {rotated_height} px",
        "Crop origin (x, y)": f"({left}, {top})",
        "Crop dimensions": f"{crop_width} × {crop_height} px",
        "Grayscale mean intensity": round(float(gray.mean()), 2),
    }
    return outputs, metrics


def _display_response(response):
    magnitude = np.abs(response)
    peak = float(magnitude.max())
    if peak == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)
    return np.rint(magnitude * (255.0 / peak)).astype(np.uint8)


def _comparison_panels(items):
    panel_width, panel_height, gap = 290, 300, 14
    chart = np.full((panel_height + 47, gap + len(items) * (panel_width + gap), 3),
                    (249, 247, 244), dtype=np.uint8)
    for index, (label, image) in enumerate(items):
        height, width = image.shape[:2]
        scale = min(panel_width / width, panel_height / height)
        fitted_width, fitted_height = max(1, round(width * scale)), max(1, round(height * scale))
        left = gap + index * (panel_width + gap)
        cv2.putText(chart, label, (left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.63,
                    (59, 44, 24), 1, cv2.LINE_AA)
        preview = cv2.resize(image, (fitted_width, fitted_height),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        top = 35 + (panel_height - fitted_height) // 2
        left += (panel_width - fitted_width) // 2
        chart[top:top + fitted_height, left:left + fitted_width] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    return chart


def _edges(image, options):
    low = _number(options, "low", 100, 0, 255)
    high = _number(options, "high", 200, 1, 255)
    blur = _number(options, "blur", 3, 1, 15)
    if low > high:
        raise ValueError("The low Canny threshold must be at or below the high threshold.")
    if blur != int(blur) or int(blur) % 2 == 0:
        raise ValueError("Choose an odd Gaussian kernel size between 1 and 15.")
    kernel_size = int(blur)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel_display = _display_response(sobel)

    prewitt_x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(smoothed, cv2.CV_32F, prewitt_x_kernel)
    prewitt_y = cv2.filter2D(smoothed, cv2.CV_32F, prewitt_x_kernel.T)
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)
    canny = cv2.Canny(smoothed, low, high)

    outputs = [
        ("Original, Sobel, and Canny comparison", _comparison_panels([
            ("Original grayscale", gray), ("Sobel magnitude", sobel_display), ("Canny edges", canny),
        ])),
        ("Gaussian-smoothed grayscale", smoothed),
        ("Sobel X", _display_response(sobel_x)),
        ("Sobel Y", _display_response(sobel_y)),
        ("Sobel magnitude", sobel_display),
        ("Prewitt magnitude", _display_response(prewitt)),
        ("Laplacian", _display_response(laplacian)),
        ("Canny edges", canny),
    ]
    count = cv2.countNonZero(canny)
    metrics = {
        "Gaussian kernel": f"{kernel_size} × {kernel_size}",
        "Canny thresholds": f"{low:g} / {high:g}",
        "Canny edge pixels": count,
        "Canny edge density": f"{100 * count / canny.size:.2f}%",
        "Peak Sobel magnitude": round(float(sobel.max()), 2),
    }
    return outputs, metrics


def _histogram_plot(channels):
    chart = np.full((350, 760, 3), (249, 247, 244), dtype=np.uint8)
    ink = (59, 44, 24)
    grid = (224, 219, 210)
    font = cv2.FONT_HERSHEY_SIMPLEX
    left, right, top, bottom = 76, 730, 66, 292
    histograms = [cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel() for _, channel, _ in channels]
    peak = max(float(hist.max()) for hist in histograms)
    for step in range(5):
        fraction = step / 4
        y = round(bottom - fraction * (bottom - top))
        cv2.line(chart, (left, y), (right, y), grid, 1, cv2.LINE_AA)
        label = f"{round(peak * fraction):,}"
        width = cv2.getTextSize(label, font, 0.44, 1)[0][0]
        cv2.putText(chart, label, (left - width - 8, y + 4), font, 0.44, ink, 1, cv2.LINE_AA)
    for level in (0, 64, 128, 192, 255):
        x = round(left + level * (right - left) / 255)
        cv2.line(chart, (x, top), (x, bottom + 5), grid, 1, cv2.LINE_AA)
        cv2.putText(chart, str(level), (x - 10, bottom + 22), font, 0.44, ink, 1, cv2.LINE_AA)
    cv2.line(chart, (left, top), (left, bottom), ink, 1, cv2.LINE_AA)
    cv2.line(chart, (left, bottom), (right, bottom), ink, 1, cv2.LINE_AA)
    cv2.putText(chart, "Pixel count", (13, 51), font, 0.46, ink, 1, cv2.LINE_AA)
    cv2.putText(chart, "Pixel intensity", (348, 337), font, 0.46, ink, 1, cv2.LINE_AA)

    x_positions = np.rint(np.linspace(left, right, 256)).astype(np.int32)
    x_legend = left
    for histogram, (label, _, color) in zip(histograms, channels):
        y_positions = np.rint(bottom - histogram / peak * (bottom - top)).astype(np.int32)
        points = np.column_stack((x_positions, y_positions))
        cv2.polylines(chart, [points], False, color, 2, cv2.LINE_AA)
        cv2.line(chart, (x_legend, 28), (x_legend + 19, 28), color, 3, cv2.LINE_AA)
        cv2.putText(chart, label, (x_legend + 25, 33), font, 0.47, ink, 1, cv2.LINE_AA)
        x_legend += cv2.getTextSize(label, font, 0.47, 1)[0][0] + 51
    return chart


def _histograms(image, options):
    clip = _number(options, "clip", 2, 0.5, 10)
    tiles = _number(options, "tiles", 8, 2, 16)
    if tiles != int(tiles):
        raise ValueError("Tiles must be a whole number between 2 and 16.")
    tile_count = int(tiles)

    blue, green, red = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_count, tile_count)).apply(gray)
    low, high, _, _ = cv2.minMaxLoc(gray)
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX) if low < high else gray.copy()
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    color_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    rgb = _histogram_plot([
        ("Red", red, (57, 62, 217)),
        ("Green", green, (92, 142, 25)),
        ("Blue", blue, (229, 107, 16)),
    ])
    luminance = _histogram_plot([
        ("Original grayscale", gray, (56, 54, 50)),
        ("Global equalization", equalized, (198, 115, 21)),
        ("CLAHE", clahe, (125, 70, 191)),
        ("Linear stretch", stretched, (87, 133, 41)),
    ])
    outputs = [
        ("Contrast comparison", _comparison_panels([
            ("Original grayscale", gray), ("Global equalization", equalized), ("CLAHE", clahe),
        ])),
        ("RGB histogram", rgb),
        ("Original grayscale", gray),
        ("Global histogram equalization", equalized),
        ("CLAHE", clahe),
        ("Linear contrast stretch", stretched),
        ("Color equalization (YUV luminance)", color_equalized),
        ("Grayscale histogram comparison", luminance),
    ]
    metrics = {
        "Original intensity standard deviation": round(float(gray.std()), 2),
        "Equalized intensity standard deviation": round(float(equalized.std()), 2),
        "CLAHE intensity standard deviation": round(float(clahe.std()), 2),
        "Stretched intensity standard deviation": round(float(stretched.std()), 2),
        "CLAHE clip limit": clip,
        "CLAHE tile grid": f"{tile_count} × {tile_count}",
    }
    return outputs, metrics


def process(lab: int, image: np.ndarray, options: dict) -> tuple[list[tuple[str, np.ndarray]], dict]:
    """Run an introductory image, edge, or histogram lab on a BGR image."""
    if lab == 1:
        return _images(image, options)
    if lab == 2:
        return _edges(image, options)
    if lab == 3:
        return _histograms(image, options)
    raise ValueError(f"Unsupported introductory lab {lab}.")
