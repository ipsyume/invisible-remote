import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque
import mss

pyautogui.FAILSAFE = False

# -------------------- MediaPipe --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -------------------- PARAMETERS --------------------
RIGHT_YAW_THRESHOLD = 10
LEFT_YAW_THRESHOLD = -10
YAW_DEADZONE = 5
TAB_COOLDOWN = 1.0

PITCH_DOWN_THRESHOLD = -14
PITCH_UP_THRESHOLD = -4
NOD_COOLDOWN = 1.5

ROLL_THRESHOLD = 12
SCROLL_SPEED = 30
SCROLL_COOLDOWN = 0.15

# -------------------- STATE --------------------
last_tab_time = 0
last_nod_time = 0
last_scroll_time = 0

pitch_history = deque(maxlen=10)
scrolling_active = False

# -------------------- SCREEN RECORDING --------------------
sct = mss.mss()
monitor = sct.monitors[1]   # primary screen

screen_w = monitor["width"]
screen_h = monitor["height"]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "invisible_remote_demo.mp4",
    fourcc,
    20,
    (screen_w + 640, max(screen_h, 480))
)

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    action_text = "IDLE"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        idx = [1, 33, 263, 61, 291, 199]
        image_points = np.array([
            (face_landmarks.landmark[i].x * w,
             face_landmarks.landmark[i].y * h)
            for i in idx
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (-30.0, -30.0, -30.0),
            (30.0, -30.0, -30.0),
            (-30.0, 30.0, -30.0),
            (30.0, 30.0, -30.0),
            (0.0, 60.0, -50.0)
        ])

        cam_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]

        current_time = time.time()
        pitch_history.append(pitch)

        # ---- SCROLL ----
        scrolling_active = abs(roll) > ROLL_THRESHOLD
        if scrolling_active and current_time - last_scroll_time > SCROLL_COOLDOWN:
            if roll > ROLL_THRESHOLD:
                pyautogui.scroll(-SCROLL_SPEED)
                action_text = "SCROLL DOWN"
            elif roll < -ROLL_THRESHOLD:
                pyautogui.scroll(SCROLL_SPEED)
                action_text = "SCROLL UP"
            last_scroll_time = current_time

        # ---- CLICK ----
        if (
            min(pitch_history) < PITCH_DOWN_THRESHOLD
            and max(pitch_history) > PITCH_UP_THRESHOLD
            and current_time - last_nod_time > NOD_COOLDOWN
        ):
            pyautogui.click()
            last_nod_time = current_time
            action_text = "CLICK"

        # ---- TAB SWITCH ----
        yaw_effective = yaw if abs(yaw) > YAW_DEADZONE else 0
        if not scrolling_active and current_time - last_tab_time > TAB_COOLDOWN:
            if yaw_effective > RIGHT_YAW_THRESHOLD:
                pyautogui.hotkey("ctrl", "tab")
                last_tab_time = current_time
                action_text = "NEXT TAB"
            elif yaw_effective < LEFT_YAW_THRESHOLD:
                pyautogui.hotkey("ctrl", "shift", "tab")
                last_tab_time = current_time
                action_text = "PREVIOUS TAB"

        # ---- OVERLAY ----
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"ACTION: {action_text}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ---- SCREEN CAPTURE ----
    screen = np.array(sct.grab(monitor))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

    # Resize screen to match height
    screen = cv2.resize(screen, (screen_w, max(screen_h, 480)))

    combined = np.hstack((frame, screen))
    out.write(combined)

    cv2.imshow("Invisible Remote (Recording)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
