import cv2
import numpy as np
import mss
import urllib.request
import os
from ultralytics import YOLO

# =========================
# AUTO DOWNLOAD MODEL FACE
# =========================
MODEL_PATH = "yolov8n-face.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO face model...")
    url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download selesai!")

# load model
model = YOLO(MODEL_PATH)

# config (OPTIMIZED)
DISPLAY_SCALE = 1
YOLO_INPUT_SIZE = 80  # 🔥 lebih kecil = lebih cepat

with mss.mss() as sct:
    monitors = sct.monitors
    monitor_number = 2
    monitor = monitors[monitor_number]

    while True:
        # =========================
        # CAPTURE SCREEN
        # =========================
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]

        # =========================
        # RESIZE UNTUK YOLO (SUPER IMPORTANT)
        # =========================
        scale_yolo = YOLO_INPUT_SIZE / max(h, w)
        small_w = int(w * scale_yolo)
        small_h = int(h * scale_yolo)

        small_frame = cv2.resize(frame, (small_w, small_h))

        # =========================
        # DETEKSI
        # =========================
        results = model(small_frame, verbose=False)[0]

        # =========================
        # DRAW BOUNDING BOX
        # =========================
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            # balik ke ukuran asli
            x1 = int(x1 / scale_yolo)
            y1 = int(y1 / scale_yolo)
            x2 = int(x2 / scale_yolo)
            y2 = int(y2 / scale_yolo)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # =========================
        # DISPLAY
        # =========================
        resized = cv2.resize(
            frame,
            (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE))
        )

        cv2.imshow("YOLO Face Detection (Optimized)", resized)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()