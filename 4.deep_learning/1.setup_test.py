#%%
from pathlib import Path
import os
from utils import save_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

from datagenerator import create_dataGenerator

num_classes = 17

# model
base_model = ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=num_classes,
)

base_model.trainable = False

head_model = Sequential(
    [GlobalAveragePooling2D(), Dense(num_classes), Activation("softmax"),]
)

model = Sequential([base_model, head_model])

# data generator
dataGenerator = create_dataGenerator(False)

# Compile model
optimizer = Adadelta(learning_rate=1.0, rho=0.95)
loss = categorical_crossentropy

model.compile(
    optimizer,
    loss=loss,
    metrics=None,
    loss_weights=None,
    sample_weight_mode=None,
    weighted_metrics=None,
    target_tensors=None,
)

# train model
model.fit_generator(
    dataGenerator,
    steps_per_epoch=None,
    epochs=3,
    verbose=1,
    callbacks=None,
    validation_data=None,
    validation_steps=None,
    validation_freq=1,
    class_weight=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    shuffle=True,
    initial_epoch=0,
)

save_model(model, "ResNet50")
