import numpy as np
from PIL import Image

TARGET_SIZE = (512, 256)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # ouverture + conversion
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0
    return image_array

def postprocess_mask(mask_array):
    # Si le masque a 4 dimensions : (1, H, W, C), on squeeze
    if len(mask_array.shape) == 4:
        mask_array = mask_array[0]

    # Si le masque est multicanal : (H, W, C), on applique argmax
    if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
        mask_array = np.argmax(mask_array, axis=-1)

    # Normalisation pour affichage (niveau de gris)
    if mask_array.max() > 0:
        mask_array = (mask_array * (255 // mask_array.max())).astype(np.uint8)
    else:
        mask_array = mask_array.astype(np.uint8)

    return Image.fromarray(mask_array)
