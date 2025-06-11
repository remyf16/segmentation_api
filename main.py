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

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Dict, Tuple


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

# Fonction locale pour calculer la répartition des classes
from plotly.offline import plot
import plotly.graph_objs as go

def compute_class_stats(catalog_path: str) -> Tuple[Dict[str, int], str]:
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

    counts = {label: 0 for label in CLASS_NAMES.values()}

    for filename in os.listdir(catalog_path):
        if filename.endswith("_mask_unet.png"):
            mask_path = os.path.join(catalog_path, filename)
            mask = np.array(Image.open(mask_path))

            for class_id, label in CLASS_NAMES.items():
                counts[label] += int(np.sum(mask == class_id))

    # Génération du graphique Plotly
    labels = list(counts.keys())
    values = list(counts.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color='mediumpurple'),
        text=values,
        textposition='auto'
    ))

    fig.update_layout(
        title="Distribution des classes (U-Net)",
        xaxis_title="Pixels",
        yaxis_title="Classes",
        template="plotly_white"
    )

    html_plot = plot(fig, output_type='div', include_plotlyjs='cdn')

    return counts, html_plot
    
@app.get("/explore", response_class=HTMLResponse)
async def explore_dataset(request: Request):
    catalog_path = "catalog"
    image_files = [f for f in os.listdir(catalog_path) if f.endswith(".png") and "_mask" not in f and "_overlay" not in f]

    comparisons = []
    for img_file in image_files:
        base = img_file.rsplit(".", 1)[0]
        original = f"/catalog/{img_file}"
        unet = f"/catalog/{base}_mask_unet.png"
        deeplab = f"/catalog/{base}_mask_deeplab.png"
        overlay_unet = f"/catalog/{base}_overlay_unet.png"
        overlay_deeplab = f"/catalog/{base}_overlay_deeplab.png"

        if all(os.path.exists(os.path.join(catalog_path, path)) for path in [
            f"{base}_mask_unet.png", f"{base}_mask_deeplab.png",
            f"{base}_overlay_unet.png", f"{base}_overlay_deeplab.png"
        ]):
            comparisons.append({
                "original": original,
                "unet": unet,
                "deeplab": deeplab,
                "overlay_unet": overlay_unet,
                "overlay_deeplab": overlay_deeplab
            })

    class_counts, plot_data = compute_class_stats(catalog_path)

    return templates.TemplateResponse("explore.html", {
        "request": request,
        "comparisons": comparisons,
        "class_counts": class_counts,
        "plot_data": plot_data
    })

# ---------------------
# Page /about
# ---------------------
@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})