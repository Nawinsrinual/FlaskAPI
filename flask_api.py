from flask import Flask, render_template_string, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import urllib.request
import os
import threading

# ======================
# ดาวน์โหลดโมเดลจาก GitHub ถ้ายังไม่มี
# ======================
model_url = "https://raw.githubusercontent.com/Nawinsrinual/FlaskAPI/main/thai_food_model.keras"
model_path = "thai_food_model.keras"

if not os.path.exists(model_path):
    print("Downloading model from GitHub...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete!")

# ======================
# โหลดโมเดล
# ======================
model = load_model(model_path)

class_names = [
    "Khaoklukkapi", "fishcake", "greencurry", "kailoogkeay", "khaomokkai",
    "khoamunkai", "kungopwunsen", "kungpao", "moosatae", "padThai",
    "padkrapao", "padpukbung", "palo", "somtum", "tomjued",
    "tomjuedmara", "tomkhakai", "tomyumkung"
]

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        img = cv2.resize(frame, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # ปิด progress bar ของ TensorFlow
        preds = model.predict(img, verbose=0)
        label = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    html = """
    <html>
    <head><title>Thai Food Realtime</title></head>
    <body style='text-align:center;background:#f9f9f9;'>
    <h1>🍜 ระบบทำนายอาหารไทยแบบเรียลไทม์</h1>
    <img src="/video" width="70%">
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================
# รัน Flask ใน Thread แยกจาก Jupyter / console
# ======================
def run_app():
    app.run(host="0.0.0.0", port=5000)

threading.Thread(target=run_app).start()
