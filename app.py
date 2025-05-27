import streamlit as st
import requests
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np
import time

API_URL = "http://localhost:8000/predict/"

# Palette fictive pour exemple (à adapter à tes classes réelles)
CLASS_LABELS = {
    0: "Fond / Route",
    36: "Trottoir / Herbe",
    144: "Bâtiments",
    252: "Objets / Panneaux"
}
CLASS_COLORS = {
    0: "#2E2E2E",       # gris foncé
    36: "#92D050",      # vert clair
    144: "#4BACC6",     # bleu
    252: "#FFC000"      # orange
}

st.set_page_config(page_title="Segmentation d'image", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Segmentation d’image - UnetMini</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Charge une image (.png ou .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Image originale", use_container_width=True)

    if st.button("📤 Envoyer à l'API pour segmentation"):
        with st.spinner("⏳ Prédiction en cours..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                start_time = time.time()
                response = requests.post(API_URL, files=files)
                duration = time.time() - start_time

                response.raise_for_status()
                result = response.json()

                if "mask_base64" in result:
                    mask_data = base64.b64decode(result["mask_base64"])
                    mask_image = Image.open(BytesIO(mask_data)).convert("L")

                    st.success(f"✅ Masque reçu en {duration:.2f} secondes")
                    st.image(mask_image, caption="🗺️ Masque segmenté", use_container_width=True)

                    # Overlay
                    st.subheader("🔍 Superposition masque + image")
                    mask_colored = ImageEnhance.Color(mask_image.convert("RGB")).enhance(2.0)
                    overlay = Image.blend(image.resize(mask_colored.size), mask_colored, alpha=0.4)
                    st.image(overlay, caption="👁️ Overlay image + masque", use_container_width=True)

                    # Download
                    st.download_button(
                        label="💾 Télécharger le masque",
                        data=mask_data,
                        file_name="masque_segmenté.png",
                        mime="image/png"
                    )

                    # Analyse des classes
                    mask_np = np.array(mask_image)
                    total_pixels = mask_np.size
                    class_ids, counts = np.unique(mask_np, return_counts=True)

                    st.markdown("---")
                    st.markdown("### 📊 Distribution des classes")

                    for cid, count in zip(class_ids, counts):
                        percentage = 100 * count / total_pixels
                        label = CLASS_LABELS.get(cid, f"Classe {cid}")
                        color = CLASS_COLORS.get(cid, "#AAAAAA")
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                            f"<div style='width:20px;height:20px;background:{color};margin-right:10px;border-radius:4px;'></div>"
                            f"<strong>{label}</strong>: {percentage:.2f} %"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.error("❌ Réponse inattendue : pas de 'mask_base64'")
            except Exception as e:
                st.error(f"🚨 Erreur lors de la requête : {e}")
else:
    st.info("🖼️ Charge une image pour commencer.")
