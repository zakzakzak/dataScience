import cv2
import numpy as np
import mss
from ultralytics import YOLO

# load YOLO model (nano = ringan)
model = YOLO("yolov8n.pt")

# config
DISPLAY_SCALE = 0.75
PANEL_RATIO = 0.2
YOLO_INPUT_SIZE = 640  # resize biar ringan

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
        # RESIZE UNTUK YOLO (BIAR CEPAT)
        # =========================
        scale_yolo = YOLO_INPUT_SIZE / max(h, w)
        small_frame = cv2.resize(frame, (int(w * scale_yolo), int(h * scale_yolo)))

        # =========================
        # DETEKSI YOLO
        # =========================
        results = model(small_frame, verbose=False)[0]

        # =========================
        # LAYOUT (80% : 20%)
        # =========================
        panel_width = int(w * PANEL_RATIO)
        main_width = w - panel_width

        scale_ratio = main_width / w
        new_height = int(h * scale_ratio)

        frame_main = cv2.resize(frame, (main_width, new_height))

        face_thumbnails = []

        # =========================
        # AMBIL HASIL DETEKSI
        # =========================
        for box in results.boxes:
            cls_id = int(box.cls[0])

            # hanya ambil PERSON (class 0 COCO)
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            # convert ke ukuran asli
            x1 = int(x1 / scale_yolo)
            y1 = int(y1 / scale_yolo)
            x2 = int(x2 / scale_yolo)
            y2 = int(y2 / scale_yolo)

            # scaling ke frame_main
            x1_s = int(x1 * scale_ratio)
            y1_s = int(y1 * scale_ratio)
            x2_s = int(x2 * scale_ratio)
            y2_s = int(y2 * scale_ratio)

            # bounding box
            cv2.rectangle(frame_main, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)

            # crop (anggap ini "wajah" sementara)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            face_thumbnails.append(crop)

        # =========================
        # PANEL KANAN (VERTICAL)
        # =========================
        panel_height = frame_main.shape[0]
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        thumb_size = panel_width
        y_offset = 0

        for face in face_thumbnails:
            if y_offset + thumb_size > panel_height:
                break

            face_resized = cv2.resize(face, (panel_width, thumb_size))
            panel[y_offset:y_offset+thumb_size, :] = face_resized
            y_offset += thumb_size

        # =========================
        # GABUNG
        # =========================
        combined = np.hstack((frame_main, panel))

        # =========================
        # DISPLAY SCALE
        # =========================
        h2, w2 = combined.shape[:2]
        combined_resized = cv2.resize(
            combined,
            (int(w2 * DISPLAY_SCALE), int(h2 * DISPLAY_SCALE))
        )

        cv2.imshow("YOLO Detection Dashboard", combined_resized)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()