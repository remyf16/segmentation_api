from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_loader import load_model
from utils.image_processing import preprocess_image, postprocess_mask, overlay_mask
from utils.class_stats import compute_class_stats

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialisation de l'application
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Création des dossiers nécessaires
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("catalog", exist_ok=True)

# Dossiers statiques
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/catalog", StaticFiles(directory="catalog"), name="catalog")

# Chargement du modèle
model = load_model()

# Configuration des templates Jinja2
templates = Jinja2Templates(directory="templates")

# ---------------------
# Page principale : formulaire
# ---------------------
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    catalog_images = [f for f in os.listdir("catalog") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "catalog_images": catalog_images
    })

# ---------------------
# Route POST : prédiction
# ---------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(None),
    catalog_image: str = Form("")
):
    file_id = str(uuid.uuid4())
    input_path = f"uploads/{file_id}.png"
    output_path = f"outputs/{file_id}_mask.png"

    # Cas 1 : Upload manuel
    if file and file.filename != "":
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Cas 2 : Sélection depuis le catalogue
    elif catalog_image:
        shutil.copy(f"catalog/{catalog_image}", input_path)

    else:
        return HTMLResponse("Aucune image fournie.", status_code=400)

    # Prédiction
    image_array = preprocess_image(input_path)
    prediction = model.predict(image_array[np.newaxis, ...])[0]
    mask, stats = postprocess_mask(prediction)

    overlay = overlay_mask(input_path, mask)
    overlay_path = f"outputs/{file_id}_overlay.png"
    overlay.save(overlay_path)
    mask.save(output_path)

    # Traduction des classes
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
    for item in stats:
        item["label"] = CLASS_NAMES.get(item["class_id"], f"Classe {item['class_id']}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original": f"/uploads/{file_id}.png",
        "output": f"/outputs/{file_id}_mask.png",
        "overlay": f"/outputs/{file_id}_overlay.png",
        "stats": stats
    })

# ---------------------
# Page d'exploration du dataset
# ---------------------
@app.get("/explore", response_class=HTMLResponse)
async def explore_page(request: Request):
    sample_images = sorted([
        f for f in os.listdir("catalog") 
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_mask" not in f
    ])[:6]

    class_counts = compute_class_stats("catalog")

    fig, ax = plt.subplots()
    class_labels = list(class_counts.keys())
    class_values = list(class_counts.values())
    ax.bar(class_labels, class_values, color='teal')
    ax.set_title("Distribution des classes")
    ax.set_ylabel("Nombre de pixels")
    ax.set_xlabel("Classe")
    plt.xticks(rotation=45)

    buffer = BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)

    # Génération des paths pour les comparaisons
    comparisons = []
    for img in sample_images:
        basename = os.path.splitext(img)[0]
        comparisons.append({
            "original": f"/catalog/{img}",
            "unet": f"/catalog/{basename}_mask_unet.png",
            "deeplab": f"/catalog/{basename}_mask.png"  # ou _mask_dl.png selon nommage
        })

    return templates.TemplateResponse("explore.html", {
        "request": request,
        "samples": sample_images,
        "plot_data": img_base64,
        "class_counts": class_counts,
        "comparisons": comparisons
    })

# ---------------------
# Page /about
# ---------------------
@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})