from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from datetime import datetime

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Info modèles
models_info = {
    "YOLOv8": {"weights": "static/models/yolov8.pt", "image": "images/yolov8.png"},
    "YOLOv9": {"weights": "static/models/yolov9.pt", "image": "images/yolov9.png"},
    "YOLOv10": {"weights": "static/models/yolov10.pt", "image": "images/yolov10.png"},
    
}

DATA_PATH = "data.yaml"  # dataset

def load_metrics_from_pt(model_path, data_path):
    """Évalue un modèle et retourne mAP, précision, rappel"""
    model = YOLO(model_path)
    results = model.val(data=data_path, split="test",save=False, plots=False)
    return {
        "mAP": round(results.box.map50, 3),
        "precision": round(results.box.mp, 3),
        "recall": round(results.box.mr, 3)
    }

@app.route("/")
def home():
    metrics_data = {}
    for model_name, info in models_info.items():
        metrics_data[model_name] = load_metrics_from_pt(info["weights"], DATA_PATH)
        metrics_data[model_name]["image"] = info["image"]
    return render_template("index.html", metrics=metrics_data)

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        model_choice = request.form.get("model")
        file = request.files.get("image")

        if not model_choice or not file or file.filename == "":
            return render_template("test.html", models=list(models_info.keys()), error="Veuillez choisir un modèle et une image.")

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        model = YOLO(models_info[model_choice]["weights"])
        results = model(save_path)

        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        results[0].save(filename=result_path)

        return render_template("test.html",
                               models=list(models_info.keys()),
                               uploaded_image=filename,
                               result_image=result_filename,
                               model=model_choice)

    return render_template("test.html", models=list(models_info.keys()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

