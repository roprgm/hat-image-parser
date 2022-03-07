from flask import Flask, render_template, request, Response
from parser import parse_image
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/parse', methods=['POST'])
def parse():
    if request.method == 'POST':
        f = request.files['file']
        image = cv2.imdecode(np.fromstring(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        jpg_image = cv2.imencode('.jpg', image)[1]
        parsed = parse_image(image)
        # f.save(secure_filename(f.filename))
        return render_template('results.html', results=parsed)


