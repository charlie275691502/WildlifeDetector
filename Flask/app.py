from flask import Flask, request, jsonify
from PIL import Image
import torch
from yolov5 import YOLOv5  # if you're using yolov5 module (not ultralytics)

import pathlib
from pathlib import Path

# Monkey-patch PosixPath on Windows
if hasattr(pathlib, 'PosixPath'):
    pathlib.PosixPath = Path

app = Flask(__name__)

# Load model (once)
model = YOLOv5("best.pt", device="cpu")  # or 'cuda'

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)

    results = model.predict(image)

    # json_output = results.pandas().xyxy[0].to_dict(orient="records")  # Convert to a list of dictionaries
    # filtered = [{"name": item["name"], "confidence": item["confidence"]} for item in json_output]
    
    # return jsonify(filtered), 400

    return jsonify(results), 400

@app.route("/", methods=["GET"])
def home():
    return "YOLO model is up and running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
