import os
import numpy as np
from PIL import Image

def load_image_gray(path, size=(128,128)):
    img = Image.open(path).convert('L').resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape((*size, 1))
    return arr

def prepare_dataset_from_dirs(base_dir, pos_dir='pos', neg_dir='neg', size=(128,128)):
    """
    Espera:
    base_dir/pos/*.png (firmas 'válidas')
    base_dir/neg/*.png (firmas 'alteradas' o forgeries)
    Retorna X (numpy), y (numpy)
    """
    X = []
    y = []
    for fname in os.listdir(os.path.join(base_dir, pos_dir)):
        path = os.path.join(base_dir, pos_dir, fname)
        try:
            X.append(load_image_gray(path, size=size))
            y.append(0)  # 0 = genuina
        except:
            continue
    for fname in os.listdir(os.path.join(base_dir, neg_dir)):
        path = os.path.join(base_dir, neg_dir, fname)
        try:
            X.append(load_image_gray(path, size=size))
            y.append(1)  # 1 = alterada
        except:
            continue
    X = np.array(X)
    y = np.array(y)
    return X, y
