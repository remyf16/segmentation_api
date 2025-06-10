import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model

# Paramètres du modèle
MODEL_PATH = "../models/deeplabv3plus_mobilenetv3_v2.h5"
CATALOG_PATH = "catalog"
OUTPUT_SUFFIX = "_mask_deeplab.png"
IMG_WIDTH, IMG_HEIGHT = 512, 512

# Fonction de prétraitement (autonome)
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # 🟢 Correction ici
    image_array = np.array(image) / 255.0
    return image_array.astype(np.float32)

# Fonction de post-traitement pour transformer le masque en image couleur
def postprocess_mask(mask_array):
    mask = np.argmax(mask_array, axis=-1).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    return mask_img, mask  # on peut retourner les deux si besoin

# Chargement du modèle
model = load_model(MODEL_PATH, compile=False)

# Création du dossier catalog s'il n'existe pas
os.makedirs(CATALOG_PATH, exist_ok=True)

# Liste des fichiers image à traiter
image_files = [f for f in os.listdir(CATALOG_PATH)
               if f.endswith(".png") and not f.endswith(OUTPUT_SUFFIX)
               and "_mask" not in f]

# Génération des masques
for filename in tqdm(image_files, desc="Génération des masques DeepLabV3+"):
    input_path = os.path.join(CATALOG_PATH, filename)
    output_path = os.path.join(CATALOG_PATH, filename.replace(".png", OUTPUT_SUFFIX))

    try:
        image_array = preprocess_image(input_path)
        prediction = model.predict(image_array[np.newaxis, ...])[0]
        mask_img, _ = postprocess_mask(prediction)
        mask_img.save(output_path)
    except Exception as e:
        print(f"Erreur avec {filename} : {e}")
