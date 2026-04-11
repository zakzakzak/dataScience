import cv2
import numpy as np
import mss

# load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# config
DISPLAY_SCALE = 1
PANEL_RATIO = 0.2  # 20% panel kanan

with mss.mss() as sct:
    monitors = sct.monitors
    monitor_number = 2
    monitor = monitors[monitor_number]

    while True:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # =========================
        # HITUNG LAYOUT (KEEP RATIO)
        # =========================
        h, w = frame.shape[:2]
        panel_width = int(w * PANEL_RATIO)
        main_width = w - panel_width

        # scale proporsional (NO DISTORT)
        scale_ratio = main_width / w
        new_height = int(h * scale_ratio)

        frame_main = cv2.resize(frame, (main_width, new_height))

        # =========================
        # DETEKSI + BBOX + CROP
        # =========================
        face_thumbnails = []

        for (x, y, w_box, h_box) in faces:
            # scaling bbox supaya sesuai frame_main
            x_scaled = int(x * scale_ratio)
            y_scaled = int(y * scale_ratio)
            w_scaled = int(w_box * scale_ratio)
            h_scaled = int(h_box * scale_ratio)

            cv2.rectangle(
                frame_main,
                (x_scaled, y_scaled),
                (x_scaled + w_scaled, y_scaled + h_scaled),
                (0, 255, 0),
                2
            )

            face_crop = frame[y:y+h_box, x:x+w_box]

            if face_crop.size == 0:
                continue

            face_thumbnails.append(face_crop)

        # =========================
        # PANEL KANAN (VERTICAL)
        # =========================
        panel_height = frame_main.shape[0]
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        if len(face_thumbnails) > 0:
            thumb_size = panel_width  # kotak

            y_offset = 0

            for face in face_thumbnails:
                if y_offset + thumb_size > panel_height:
                    break

                face_resized = cv2.resize(face, (panel_width, thumb_size))

                panel[y_offset:y_offset+thumb_size, 0:panel_width] = face_resized
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

        cv2.imshow("Face Detection Dashboard", combined_resized)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()