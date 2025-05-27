import numpy as np
from PIL import Image

TARGET_SIZE = (512, 256)

def preprocess_image(image: Image.Image):
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0  # normalisation
    return image_array

def postprocess_mask(mask_array):
    # Pour visualiser en niveau de gris simple
    mask_img = Image.fromarray((mask_array * (255 // mask_array.max())).astype(np.uint8))
    return mask_img
