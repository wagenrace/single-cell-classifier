import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .load_data import load_all_data

# data generator
current_loc = Path(__file__).parents[1]
train_data_loc = os.path.join(current_loc, "0.process-data", "data", "train")


def create_dataGenerator(pre_load_data: bool=False, isTraining: bool = True, validationFac: float = 0.1):
    subset = "training" if isTraining else "validation"
    parameters1 = {
        "featurewise_center": False,
        "samplewise_center": False,
        "featurewise_std_normalization": False,
        "samplewise_std_normalization": False,
        "zca_whitening": False,
        "zca_epsilon": 1e-06,
        "rotation_range": 360,  # 360
        "width_shift_range": 0.0,
        "height_shift_range": 0.0,
        "brightness_range": None,
        "shear_range": 0.0,
        "zoom_range": 0.0,
        "channel_shift_range": 0.0,
        "fill_mode": "nearest",
        "cval": 0.0,
        "horizontal_flip": True,
        "vertical_flip": True,
        "rescale": 1.0 / 255,
        "preprocessing_function": None,
        "data_format": "channels_last",
        "validation_split": validationFac,
        "dtype": "float32",
    }
    
    data_generator =  ImageDataGenerator(**parameters1)

    if pre_load_data:
        x, y = load_all_data(train_data_loc)
        parameters2 = {
            "batch_size": 32,
            "shuffle": True,
            "sample_weight": None,
            "seed": None,
            "save_to_dir": None,
            "save_prefix": "",
            "save_format": "png",
            "subset": subset,
        }
        return data_generator.flow(x, y, **parameters2)
    else:
        parameters2 = {
            "target_size": (201, 201),
            "color_mode": "rgb",
            "classes": None,
            "class_mode": "categorical",
            "batch_size": 32,
            "shuffle": True,
            "seed": None,
            "save_to_dir": None,
            "save_prefix": "",
            "save_format": "png",
            "follow_links": False,
            "subset": subset,
            "interpolation": "nearest",
        }   
        return data_generator.flow_from_directory(
            train_data_loc, **parameters2
        )

