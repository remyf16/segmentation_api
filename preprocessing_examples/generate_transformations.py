import os
from PIL import Image, ImageFilter
import numpy as np
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Paramètres
IMG_WIDTH_UNET = 512
IMG_HEIGHT_UNET = 256
IMG_SIZE_DL = 512

# Dossiers
catalog_path = "catalog"
output_path = "preprocessing_examples"
os.makedirs(output_path, exist_ok=True)

# Sélection d’une image exemple
example_file = next((f for f in os.listdir(catalog_path) if f.endswith(".png") and "_mask" not in f), None)
original = Image.open(os.path.join(catalog_path, example_file)).convert("RGB")

# Resize pour U-Net
unet_resized = original.resize((IMG_WIDTH_UNET, IMG_HEIGHT_UNET))

# Floutage
unet_blurred = unet_resized.filter(ImageFilter.GaussianBlur(radius=2))

# Resize pour DeepLabV3+
deeplab_resized = original.resize((IMG_SIZE_DL, IMG_SIZE_DL))

# Prétraitement MobileNetV3
dl_array = np.array(deeplab_resized).astype(np.float32)
dl_array = preprocess_input(dl_array)
dl_display = ((dl_array - dl_array.min()) / (dl_array.max() - dl_array.min()) * 255).astype(np.uint8)
dl_processed = Image.fromarray(dl_display)

# Sauvegarde des versions
original.save(os.path.join(output_path, "1_original.png"))
unet_resized.save(os.path.join(output_path, "2_unet_resized.png"))
unet_blurred.save(os.path.join(output_path, "3_unet_blurred.png"))
deeplab_resized.save(os.path.join(output_path, "4_deeplab_resized.png"))
dl_processed.save(os.path.join(output_path, "5_deeplab_preprocessed.png"))
