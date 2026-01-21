import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = False

# -------------------- MediaPipe Setup --------------------
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

# TAB SWITCH (Yaw)
LEFT_YAW_THRESHOLD = -18
RIGHT_YAW_THRESHOLD = 18
YAW_DEADZONE = 8
TAB_COOLDOWN = 1.3

# NOD CLICK (Pitch)
PITCH_DOWN_THRESHOLD = -14
PITCH_UP_THRESHOLD = -4
NOD_COOLDOWN = 1.5

# SCROLL (Roll)
ROLL_THRESHOLD = 12
SCROLL_SPEED = 30
SCROLL_COOLDOWN = 0.15

# -------------------- STATE --------------------
last_tab_time = 0
last_nod_time = 0
last_scroll_time = 0

pitch_history = deque(maxlen=10)
yaw_history = deque(maxlen=5)

scrolling_active = False

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
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

        success, rot_vec, trans_vec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch, yaw, roll = angles

        current_time = time.time()

        pitch_history.append(pitch)
        yaw_history.append(yaw)

        # -------------------- SCROLL (HIGHEST PRIORITY) --------------------
        if abs(roll) > ROLL_THRESHOLD:
            scrolling_active = True
        else:
            scrolling_active = False

        if scrolling_active and current_time - last_scroll_time > SCROLL_COOLDOWN:
            if roll > ROLL_THRESHOLD:
                pyautogui.scroll(-SCROLL_SPEED)
                last_scroll_time = current_time
                action_text = "SCROLL DOWN"
            elif roll < -ROLL_THRESHOLD:
                pyautogui.scroll(SCROLL_SPEED)
                last_scroll_time = current_time
                action_text = "SCROLL UP"

        # -------------------- NOD CLICK --------------------
        if len(pitch_history) >= 6:
            if (
                min(pitch_history) < PITCH_DOWN_THRESHOLD
                and max(pitch_history) > PITCH_UP_THRESHOLD
                and current_time - last_nod_time > NOD_COOLDOWN
            ):
                pyautogui.click()
                last_nod_time = current_time
                action_text = "CLICK"

        # -------------------- TAB SWITCH (ONLY IF NOT SCROLLING) --------------------
        
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


        # -------------------- DISPLAY --------------------
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"ACTION: {action_text}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Invisible Remote", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
