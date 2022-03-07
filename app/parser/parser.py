import os
import cv2
import easyocr
from multiprocessing.pool import ThreadPool
import numpy as np

MODELS_PATH = os.path.join(os.path.dirname(__file__), './models')
MODEL_NAME = 'custom_trade'

RED = np.array([20, 0, 255])
GREEN = np.array([20, 255, 0])
WHITE = np.array([255, 255, 255])

IGNORED_CHARS = ': -'
LINE_HEIGHT = 65
LINE_THR = 2


# Image utils
def map_range(value, in_min, in_max, out_min, out_max):
    return np.clip(
        out_min + (((value - in_min) / (in_max - in_min)) * (out_max - out_min)),
        out_min,
        out_max,
    )


def color_mask(image: np.ndarray, color: np.ndarray):
    diff = np.max(np.absolute(image - color), axis=-1)
    return map_range(diff, 150, 64, 0, 255).astype(np.uint8)


def crop_image(image: np.ndarray, bbox: tuple, margin: int = 0):
    [x0, y0, x1, y1] = map(int, bbox)
    x0 = max(0, x0 - margin)
    x1 = min(image.shape[0], x1 + 1 + margin)
    y0 = max(0, y0 - margin)
    y1 = min(image.shape[1], y1 + 1 + margin)
    return image[x0:x1, y0:y1]


def trim_image(image, mask=None, margin=0):
    if mask is None:
        mask = image > 10
    nz = np.nonzero(mask)
    return crop_image(
        image, (nz[0].min(), nz[1].min(), nz[0].max() + 1, nz[1].max() + 1), margin
    )


def ensure_line_size(image: np.ndarray):
    h, w = image.shape[:2]
    if abs(h - LINE_HEIGHT) > LINE_THR:
        return cv2.resize(image, (int(LINE_HEIGHT * w / h), LINE_HEIGHT))
    return image


# OCR
def init_reader():
    print('Init reader')
    return easyocr.Reader(
        ['en'],
        gpu=False,
        model_storage_directory=MODELS_PATH,
        user_network_directory=MODELS_PATH,
        download_enabled=False,
        recog_network='custom_trade'
    )


reader = init_reader()


def predict_text(image: np.ndarray):
    predicted = reader.readtext(
        image,
        workers=0,
        slope_ths=0.5,
        ycenter_ths=1.0,
        height_ths=1.0,
        width_ths=1.0,
        x_ths=5.0,
        y_ths=5.0,
        output_format='dict'
    )

    text = predicted[0]['text']
    for char in IGNORED_CHARS:
        text.replace(char, '')

    return text


def parse_text(text: str):
    symbol = text.split(' ')[0].replace('$', '')
    body = text.split(' ', maxsplit=1)[1].replace(' ', '').replace('$', '')
    parts = body.split(',')

    out = [symbol]
    call = None
    put = None
    for part in parts:
        if 'C' in part:
            values = part.replace('C', '').split('>')
            if len(values) == 2:
                call = [float(values[0]), float(values[1])]
        if 'P' in part:
            values = part.replace('P', '').split('<')
            if len(values) == 2:
                put = [float(values[0]), float(values[1])]

    out.extend(call if call else ['', ''])
    out.extend(put if put else ['', ''])
    return out


workers = os.environ.get('PARSER_WORKERS', 4)
pool = ThreadPool(processes=workers)
print(f'Using {workers} workers.')


def parse_image(image: np.ndarray):
    # Split lines
    image_rg = color_mask(image, GREEN) + color_mask(image, RED)
    line_groups = np.split(image, np.nonzero(np.sum(image_rg, axis=1) < 20)[0] + 1)

    # Filter empty lines
    line_groups = [g for g in line_groups if g.shape[0] > 40]
    print(f'Detected {len(line_groups)} lines.')

    filtered_lines = []
    for line in line_groups:
        line_white = color_mask(line, WHITE)
        line_green = color_mask(line, GREEN)
        line_red = color_mask(line, RED)
        line_gray = line_white + line_green + line_red

        # Trim and resize line
        line_gray = trim_image(line_gray)
        line_gray = ensure_line_size(line_gray)

        # Remove initial symbol
        line_gray = line_gray[:, 80:]
        line_gray = trim_image(line_gray)
        filtered_lines.append(line_gray)

    text_lines = pool.map(predict_text, filtered_lines)
    responses = [
        parse_text(line)
        for line in text_lines
        if line.startswith('$')
    ]

    for line in responses:
        print(' '.join(map(str, line)))
    return responses
