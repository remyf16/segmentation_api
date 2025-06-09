import os
import numpy as np
from PIL import Image

# Classes utilisées (adaptées à ton modèle)
CLASS_NAMES = {
    0: "Arrière-plan",
    1: "Véhicules",
    2: "Piétons",
    3: "Bâtiments",
    4: "Route",
    5: "Signalisation",
    6: "Végétation",
    7: "Autres"
}

def compute_class_stats(catalog_path):
    class_counts = {label: 0 for label in CLASS_NAMES.values()}

    for file in os.listdir(catalog_path):
        if file.endswith("_mask.png"):
            path = os.path.join(catalog_path, file)
            mask = np.array(Image.open(path))
            for class_id, label in CLASS_NAMES.items():
                class_counts[label] += np.sum(mask == class_id)

    return class_counts
