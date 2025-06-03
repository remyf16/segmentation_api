import numpy as np
from PIL import Image

TARGET_SIZE = (512, 256)  # (width, height) pour .resize()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
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

    # Palette de couleurs : (classe_id → couleur RGB)
    palette = np.array([
        [0, 0, 0],        # Classe 0 : noir
        [255, 0, 0],      # Classe 1 : rouge
        [0, 255, 0],      # Classe 2 : vert
        [0, 0, 255],      # Classe 3 : bleu
        [255, 255, 0],    # Classe 4 : jaune
        [255, 0, 255],    # Classe 5 : magenta
        [0, 255, 255],    # Classe 6 : cyan
        [128, 128, 128]   # Classe 7 : gris
    ], dtype=np.uint8)

    # Convertir le masque (H, W) → (H, W, 3) via la palette
    mask_rgb = palette[mask_array]

    return Image.fromarray(mask_rgb)
