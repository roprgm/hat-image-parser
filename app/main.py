from flask import Flask, render_template, request
import cv2
import numpy as np
from .parser import parse_image

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/parse', methods=['POST'])
def parse():
    if request.method == 'POST':
        f = request.files['file']
        image = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        parsed = parse_image(image)
        return render_template('results.html', results=parsed)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
