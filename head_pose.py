import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Key landmark indices
        idx = [1, 33, 263, 61, 291, 199]  
        image_points = np.array([
            (face_landmarks.landmark[i].x * w,
             face_landmarks.landmark[i].y * h)
            for i in idx
        ], dtype="double")

        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (-30.0, -30.0, -30.0), # Left eye
            (30.0, -30.0, -30.0),  # Right eye
            (-30.0, 30.0, -30.0),  # Left mouth
            (30.0, 30.0, -30.0),   # Right mouth
            (0.0, 60.0, -50.0)     # Chin
        ])

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            model_points,
            image_points,
            cam_matrix,
            dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch, yaw, roll = angles

        cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.2f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Step 2 - Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
