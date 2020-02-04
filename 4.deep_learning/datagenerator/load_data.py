import os
from pathlib import Path

from keras.utils import to_categorical
import numpy as np
from PIL import Image

def load_all_data(train_data_loc):
    x = []
    y = []

    all_labels = os.listdir(train_data_loc)
    for label_idx, label in enumerate(all_labels):
        for img_name in os.listdir(os.path.join(train_data_loc, label)):
            img = Image.open(os.path.join(train_data_loc, label, img_name))
            x.append(np.array(img))
            y.append(label_idx)

    x = np.array(x, dtype=np.uint8)
    y = to_categorical(y)

    return x, y