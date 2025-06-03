import numpy as np
from PIL import Image

TARGET_SIZE = (512, 256)  # (width, height) pour .resize()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image) / 255.0
    return image_array

def postprocess_mask(mask_array):
    # Gestion des dimensions
    if len(mask_array.shape) == 4:
        mask_array = mask_array[0]
    if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
        mask_array = np.argmax(mask_array, axis=-1)

    # Palette RGB
    palette = np.array([
        [0, 0, 0],        # 0 : Fond
        [255, 0, 0],      # 1 : Classe 1
        [255, 255, 0],    # 2 : Classe 2
        [0, 255, 0],      # 3 : Classe 3
        [0, 0, 255],      # 4 : ...
        [255, 0, 255],
        [0, 255, 255],
        [128, 128, 128]
    ], dtype=np.uint8)

    # Masque colorisé
    mask_rgb = palette[mask_array]
    mask_img = Image.fromarray(mask_rgb)

    # Stats de classes
    unique, counts = np.unique(mask_array, return_counts=True)
    total = mask_array.size
    stats = [{"class_id": int(cls), "percent": round(100 * count / total, 1)} for cls, count in zip(unique, counts)]

    return mask_img, stats
    
def overlay_mask(original_path, mask_img, alpha=0.5):
    original = Image.open(original_path).convert("RGB").resize(mask_img.size)
    return Image.blend(original, mask_img.convert("RGB"), alpha)
