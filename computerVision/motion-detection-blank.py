import cv2
import numpy as np
import mss
from collections import deque

# ======================
# CONFIG
# ======================
MONITOR_INDEX = 2
HISTORY_LENGTH = 100
SMOOTH_WINDOW = 5
MIN_AREA = 500

history = deque(maxlen=HISTORY_LENGTH)
smooth_history = deque(maxlen=HISTORY_LENGTH)

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

        # Resize biar ringan
        frame = cv2.resize(frame, (960, 540))

        # ======================
        # Preprocessing
        # ======================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # ======================
        # Frame Differencing
        # ======================
        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morphology
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # ======================
        # Filter area (tanpa bounding box)
        # ======================
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_mask = np.zeros_like(frame)
        motion_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < MIN_AREA:
                continue

            motion_area += area

            # langsung fill area merah (tanpa kotak)
            cv2.drawContours(motion_mask, [cnt], -1, (0, 0, 255), -1)

        # ======================
        # Background hitam
        # ======================
        black_bg = np.zeros_like(frame)

        output = cv2.addWeighted(black_bg, 1, motion_mask, 1, 0)

        # ======================
        # Motion signal
        # ======================
        history.append(motion_area)

        if len(history) >= SMOOTH_WINDOW:
            smooth_val = int(np.mean(list(history)[-SMOOTH_WINDOW:]))
        else:
            smooth_val = motion_area

        smooth_history.append(smooth_val)

        # ======================
        # 📊 Compact Graph
        # ======================
        graph_height = 60
        graph_width = output.shape[1]

        graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

        if len(smooth_history) > 1:
            max_val = max(smooth_history) if max(smooth_history) > 1000 else 1000

            for i in range(1, len(smooth_history)):
                x1 = int((i - 1) / len(smooth_history) * graph_width)
                x2 = int(i / len(smooth_history) * graph_width)

                y1 = int(graph_height - (smooth_history[i - 1] / max_val) * graph_height)
                y2 = int(graph_height - (smooth_history[i] / max_val) * graph_height)

                cv2.line(graph, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Text info
        cv2.putText(graph, f"Motion(area): {int(smooth_val)}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

        # ======================
        # Combine UI
        # ======================
        combined = np.vstack((output, graph))

        cv2.imshow("Motion Only View", combined)

        prev_frame = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()