FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# PyTorch version processeur (beaucoup plus léger que la version CUDA)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code et modèles
COPY *.py ./
COPY models/detection.pt models/classification.pt models/

# Les images/annotations sont montées au lancement (volume), pas copiées dans l'image
ENV IMAGES_DIR=/data/images XML_DIR=/data/annotations SORTIE_DIR=/app/sortie_annotee

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]