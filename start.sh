#!/bin/bash
export PORT=${PORT:-8000}
uvicorn main:app --host 0.0.0.0 --port 8000
