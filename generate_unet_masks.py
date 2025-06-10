import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from utils.image_processing import preprocess_image, postprocess_mask

CATALOG_PATH = "catalog"
UNET_MODEL_PATH = "models/unet_mini_best.h5"  # chemin relatif depuis ~/P8/segmentation_project/

model = load_model(UNET_MODEL_PATH, compile=False)

for filename in os.listdir(CATALOG_PATH):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")) and "_mask" not in filename:
        filepath = os.path.join(CATALOG_PATH, filename)
        basename = os.path.splitext(filename)[0]

        print(f"[U-Net] {filename}")
        image_array = preprocess_image(filepath)
        prediction = model.predict(image_array[np.newaxis, ...])[0]
        mask, _ = postprocess_mask(prediction)
        mask.save(os.path.join(CATALOG_PATH, f"{basename}_mask_unet.png"))
