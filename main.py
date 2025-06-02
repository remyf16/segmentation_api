from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_loader import load_model
from utils.image_processing import preprocess_image, postprocess_mask

import numpy as np
from PIL import Image

# Initialisation
app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Config dossiers
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Modèle
model = load_model()

# Templates HTML
templates = Jinja2Templates(directory="templates")

# Route principale : formulaire HTML
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Route POST : upload et prédiction
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Enregistrer l'image uploadée
    file_id = str(uuid.uuid4())
    input_path = f"uploads/{file_id}.png"
    output_path = f"outputs/{file_id}_mask.png"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prétraitement + prédiction
    image_array = preprocess_image(input_path)
    prediction = model.predict(image_array[np.newaxis, ...])[0]
    mask = postprocess_mask(prediction)

    # Sauvegarde masque
    Image.fromarray(mask).save(output_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original": input_path,
        "output": output_path
    })

# Pour servir les images uploadées et les résultats
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
