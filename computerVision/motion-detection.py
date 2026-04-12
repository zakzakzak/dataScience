import cv2
import numpy as np
import mss

# Pilih monitor (1 = utama, 2 = monitor kedua)
MONITOR_INDEX = 2

with mss.mss() as sct:
    monitor = sct.monitors[MONITOR_INDEX]

    prev_frame = None

    while True:
        # Capture screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        # Convert BGRA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Convert ke grayscale (biar ringan)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur dikit biar noise berkurang
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # Hitung perbedaan
        diff = cv2.absdiff(prev_frame, gray)

        # Threshold (biar jelas mana yg berubah)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Optional: dilate biar area lebih kelihatan
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Buat mask berwarna (highlight merah)
        mask = np.zeros_like(frame)
        mask[thresh == 255] = [0, 0, 255]  # merah

        # Combine dengan frame asli
        output = cv2.addWeighted(frame, 1, mask, 0.7, 0)

        # Tampilkan
        cv2.imshow("Motion Detection", output)

        # Update frame sebelumnya
        prev_frame = gray

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()