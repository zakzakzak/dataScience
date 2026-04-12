import cv2
import mediapipe as mp
import numpy as np
import mss

# =========================
# INIT
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

sct = mss.mss()
monitor = sct.monitors[2]  # ⚠️ ganti ke [1] kalau cuma 1 monitor

# Landmark (urutan penting)
# [mata kiri, mata kanan, hidung, mulut kiri, mulut kanan, dagu, dahi, kiri wajah, kanan wajah]
KEY_POINTS = [
    33, 263,
    1,
    61, 291,
    199,
    10,
    234, 454
]

# Koneksi triangular (rapi & simetris)
CONNECTIONS = [
    (0, 2), (1, 2), (0, 1),   # atas
    (3, 5), (4, 5), (3, 4),   # bawah
    (2, 3), (2, 4),           # tengah
    (6, 0), (6, 1),           # dahi
    (7, 0), (7, 5),           # kiri
    (8, 1), (8, 5)            # kanan
]

# Window normal (biar tidak stretch)
cv2.namedWindow("Face Triangular", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Triangular", 1200, 800)

# =========================
# LOOP
# =========================
while True:
    # Capture monitor
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 🔥 Resize proporsional (tidak melebar)
    h, w, _ = frame.shape
    scale = 1000 / w
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Convert ke RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w, _ = frame.shape

            points = []
            for idx in KEY_POINTS:
                lm = face.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))

                # titik
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # garis triangular (rapi)
            for a, b in CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan
    cv2.imshow("Face Triangular", frame)

    # Exit ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()