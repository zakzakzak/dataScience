import cv2
import numpy as np
import mss
from collections import deque

# ======================
# CONFIG
# ======================
MONITOR_INDEX = 2
HISTORY_LENGTH = 100

history = deque(maxlen=HISTORY_LENGTH)

with mss.mss() as sct:
    monitor = sct.monitors[MONITOR_INDEX]
    prev_frame = None

    while True:
        # ======================
        # Capture screen
        # ======================
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # ======================
        # Preprocessing
        # ======================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # ======================
        # Motion detection
        # ======================
        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # ======================
        # Hitung motion level
        # ======================
        motion_level = np.sum(thresh == 255)
        motion_level = min(motion_level, 100000)  # limit biar stabil
        history.append(motion_level)

        # ======================
        # Highlight merah
        # ======================
        mask = np.zeros_like(frame)
        mask[thresh == 255] = [0, 0, 255]

        output = cv2.addWeighted(frame, 1, mask, 0.6, 0)

        # ======================
        # 📊 Grafik Compact
        # ======================
        graph_height = 60
        graph_width = output.shape[1]

        graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

        if len(history) > 1:
            max_val = max(history) if max(history) > 1000 else 1000

            for i in range(1, len(history)):
                x1 = int((i - 1) / len(history) * graph_width)
                x2 = int(i / len(history) * graph_width)

                y1 = int(graph_height - (history[i - 1] / max_val) * graph_height)
                y2 = int(graph_height - (history[i] / max_val) * graph_height)

                cv2.line(graph, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Info text
        if len(history) > 0:
            cv2.putText(graph, f"Motion: {history[-1]}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

        # ======================
        # Combine frame + graph
        # ======================
        combined = np.vstack((output, graph))

        cv2.imshow("Motion Detection + Graph", combined)

        # Update frame sebelumnya
        prev_frame = gray

        # Exit pakai tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()