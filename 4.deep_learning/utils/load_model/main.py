from keras.models import model_from_json
import json
import os

def load_model(name, location="saved_models"):
    location = os.path.join(location, name)
    with open(os.path.join(location, 'model.json'), 'r') as f:
        json_string = json.load(f)
    model = model_from_json(json_string)
    model.load_weights(os.path.join(location, 'weights.h5'))

    return model