
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import time
from collections import deque

# Load trained model
model = tf.keras.models.load_model("sign_language_model.keras")

with open("labels.json", "r", encoding="utf-8") as file:
    labels = json.load(file)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera!")
    raise SystemExit

# Store the latest 100 frames
sequence = deque(maxlen=100)

start_time = time.perf_counter()
last_prediction = "Waiting for sign..."
confidence = 0.0

with HandLandmarker.create_from_options(options) as detector:

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            (time.perf_counter() - start_time) * 1000
        )

        results = detector.detect_for_video(
            mp_image, timestamp
        )

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            landmarks = []

            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])

                cv2.circle(
                    frame, (x, y), 5, (0, 255, 0), -1
                )

                landmarks.extend([
                    point.x,
                    point.y,
                    point.z
                ])

            sequence.append(landmarks)

            # Predict after collecting 100 frames
            if len(sequence) == 100:

                input_data = np.array(
                    sequence,
                    dtype=np.float32
                ).reshape(1, 100, 63)

                prediction = model.predict(
                    input_data, verbose=0
                )[0]

                predicted_index = int(np.argmax(prediction))
                confidence = float(prediction[predicted_index])

                if confidence >= 0.70:
                    last_prediction = labels[predicted_index]
                else:
                    last_prediction = "Not sure"

        else:
            sequence.clear()
            last_prediction = "Show your hand"
            confidence = 0.0

        cv2.putText(
            frame,
            "Prediction: " + last_prediction,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/100",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Kurdish Sign Language Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
