#!/bin/bash
# Lancer l'API FastAPI en arrière-plan
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Lancer l'app Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

