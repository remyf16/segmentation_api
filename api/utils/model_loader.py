import os
import numpy as np
from tensorflow.keras.models import load_model as keras_load

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(base_dir, '..','..', 'models', 'unet_mini_best.h5'))
    model = keras_load(model_path, compile=False)
    return model

def predict_mask(model, image_array):
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    predicted_mask = np.argmax(prediction[0], axis=-1)
    return predicted_mask