from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from model.utils import load_image_gray
import numpy as np

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/signature_model.h5'

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Try to load model lazily
MODEL = None
try:
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH)
        app.logger.info("Modelo cargado desde: %s", MODEL_PATH)
    else:
        app.logger.warning("No se encontró modelo. Use model/train.py para entrenar.")
except Exception as e:
    app.logger.warning("No se pudo cargar TensorFlow (modo demo).", exc_info=e)
    MODEL = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result={"error": "No file uploaded"})
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result={"error": "Empty filename"})
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # preprocess
    arr = load_image_gray(path, size=(128,128))
    arr = np.expand_dims(arr, axis=0)

    if MODEL is None:
        return render_template('index.html', result={"demo": True, "score": 0.42, "note": "Modo demo: entrena el modelo para resultados reales."})

    prob = float(MODEL.predict(arr)[0,0])
    return render_template('index.html', result={"demo": False, "score": prob})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error":"no file"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    arr = load_image_gray(path, size=(128,128))
    arr = np.expand_dims(arr, axis=0)
    if MODEL is None:
        return jsonify({"demo": True, "score": 0.42, "message":"Train the model for real predictions."})
    prob = float(MODEL.predict(arr)[0,0])
    return jsonify({"score": prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
