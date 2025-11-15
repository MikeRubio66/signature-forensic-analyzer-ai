#!/usr/bin/env python3
"""
Entrenamiento de ejemplo para Signature Analyzer AI.

Uso:
python model/train.py --data-dir data --epochs 10 --save-path model/signature_model.h5
"""
import argparse
import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model.utils import prepare_dataset_from_dirs

def build_model(input_shape=(128,128,1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data', help='Directorio con subdirs pos/ neg/')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--save-path', default='model/signature_model.h5')
    args = p.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    X, y = prepare_dataset_from_dirs(args.data_dir, pos_dir='pos', neg_dir='neg', size=(128,128))
    if len(X) == 0:
        raise SystemExit("No training images found. Añade data/pos y data/neg con imágenes PNG/JPG.")

    # shuffle / split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(input_shape=X_train.shape[1:])
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ck = ModelCheckpoint(args.save_path, save_best_only=True, monitor='val_loss', mode='min')
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch, callbacks=[ck, es])

    print("Entrenamiento finalizado. Modelo guardado en:", args.save_path)

if __name__ == '__main__':
    main()
