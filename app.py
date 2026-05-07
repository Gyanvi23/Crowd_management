from flask import Flask, render_template, request, jsonify
from matplotlib.pyplot import box
from ultralytics import YOLO
import cv2
import numpy as np
import base64


app = Flask(__name__)

model = YOLO("yolov8m.pt")
print("YOLO RUNNING")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    print("API CALLED")
    data = request.json['image']

    encoded = data.split(',')[1]

    nparr = np.frombuffer(
        base64.b64decode(encoded),
        np.uint8
    )

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame)

    detections = []

    count = 0

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])
            

            if cls == 0 and conf > 0.3:
                print("PERSON DETECTED")

                count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "x1":x1,
                    "y1":y1,
                    "x2":x2,
                    "y2":y2
                })

    return jsonify({
        "count":count,
        "detections":detections
    })

if __name__ == "__main__":
    app.run(debug=False)