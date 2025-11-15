#!/usr/bin/env python3
"""
Predict CLI:
python model/predict.py --model model/signature_model.h5 --file examples/test1.png
"""
import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from model.utils import load_image_gray

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='model/signature_model.h5')
    p.add_argument('--file', required=True)
    args = p.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit("File not found: " + args.file)

    if not os.path.exists(args.model):
        print("Modelo no encontrado:", args.model)
        print("Ejecutando modo demo (score fijo). Entrena el modelo con model/train.py")
        print("score: 0.42")
        return

    model = load_model(args.model)
    arr = load_image_gray(args.file, size=(128,128))
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr)[0,0])
    print(f"Probabilidad de ALTERACIÓN (1 = alterada): {prob:.4f}")

if __name__ == '__main__':
    main()
