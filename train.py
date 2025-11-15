# train.py - pipeline de ejemplo (placeholder)
import os
from tensorflow.keras import layers, models
def build_model():
    model = models.Sequential([
        layers.Input(shape=(128,128,1)),
        layers.Conv2D(16,3,activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(32,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = build_model()
    print('Modelo creado. Agrega tu dataset en data/ y añade el pipeline de ImageDataGenerator para entrenar.')
    model.save('signature_model.h5')
