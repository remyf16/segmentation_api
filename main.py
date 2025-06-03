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

# Initialisation
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Montages des répertoires statiques
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/catalog", StaticFiles(directory="catalog"), name="catalog")

# Création des répertoires si besoin
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Chargement du modèle
model = load_model()

# Configuration templates
templates = Jinja2Templates(directory="templates")

# Route GET : page formulaire
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    catalog_images = [f for f in os.listdir("catalog") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "catalog_images": catalog_images
    })

# Route POST : prédiction
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

    # Prétraitement et prédiction
    image_array = preprocess_image(input_path)
    prediction = model.predict(image_array[np.newaxis, ...])[0]
    mask, stats = postprocess_mask(prediction)

    overlay = overlay_mask(input_path, mask)
    overlay_path = f"outputs/{file_id}_overlay.png"
    overlay.save(overlay_path)
    mask.save(output_path)

    # Ajout des noms de classes
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
