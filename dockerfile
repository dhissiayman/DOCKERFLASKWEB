FROM python:3.10-slim

WORKDIR /app

# Dépendances système utiles pour OpenCV et Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie tout le projet
COPY . .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Lancer Flask
CMD ["python", "app.py"]
