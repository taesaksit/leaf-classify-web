import os.path
from flask import Flask, render_template, request
from main import predict512, predict100
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file_image']
    rgb_folder = './static/uploads/rgb/'
    detect_folder = './static/uploads/detect/'

    image_path = os.path.join(rgb_folder, image_file.filename)
    image_path_detect = os.path.join(detect_folder, image_file.filename)

    image_file.save(image_path)

    models = [(predict512, '_512'), (predict100, '_100')]
    results = []

    for model, suffix in models:
        pred, image_detect = model(image_path)
        if image_detect is not None:
            image_detect = Image.fromarray(image_detect)
            image_detect.save(image_path_detect + suffix + '.png')
            results.append((pred, image_path_detect + suffix + '.png',f'Feature:{suffix}'))
        else:
            results.append((pred, image_path, f'Feature:{suffix}'))

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
