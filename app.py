from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Load YOLO model

@app.route("/detect", methods=["POST"])
def detect_objects():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    results = model(image)
    
    return jsonify({"detections": results.pandas().xyxy[0].to_dict()})

if __name__ == "__main__":
    app.run(debug=True)
