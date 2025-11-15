from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import os
app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'signature_model.h5')
@app.route('/', methods=['GET'])
def index():
    return 'Signature Analyzer AI: sube una imagen POST /predict'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}), 400
    f = request.files['file']
    img = Image.open(f).convert('L').resize((128,128))
    arr = np.array(img)/255.0
    arr = arr.reshape(1,128,128,1)
    # Placeholder: without model loaded, return demo score
    score = 0.42
    return jsonify({'score': score, 'message': 'demo prediction; connect a real model in model/signature_model.h5'})
if __name__ == '__main__':
    app.run(debug=True, port=8000)
