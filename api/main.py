from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from utils.image_processing import preprocess_image, postprocess_mask
from utils.model_loader import load_model, predict_mask
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI(
    title="API de Segmentation d’Images",
    root_path="/proxy/8000"
)

# Charger le modèle au démarrage
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "Format non supporté"}, status_code=400)

    try:
        # Lecture et traitement de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = preprocess_image(image)

        # Prédiction
        mask = predict_mask(model, image_array)
        mask_image = postprocess_mask(mask)

        # Conversion de l’image en base64
        buf = io.BytesIO()
        mask_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        mask_b64 = base64.b64encode(byte_im).decode("utf-8")

        return JSONResponse(content={"mask_base64": mask_b64}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"Erreur lors de la prédiction : {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
