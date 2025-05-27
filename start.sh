#!/bin/bash

# Lancer l'API FastAPI (port 8000 en interne)
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Lancer Streamlit sur le port attendu par Azure (port 80)
exec streamlit run app.py --server.port 80 --server.address 0.0.0.0
