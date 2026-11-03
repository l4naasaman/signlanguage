
import cv2
import mediapipe as mp
import time

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    for point in landmarks:
        x = int(point.x * width)
        y = int(point.y * height)
        points.append((x, y))

    # Draw connections between hand points
    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, (255, 0, 0), 2)

    # Draw the green dots
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)

start_time = time.perf_counter()

with HandLandmarker.create_from_options(options) as detector:

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            print("Could not access camera")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int((time.perf_counter() - start_time) * 1000)

        results = detector.detect_for_video(mp_image, timestamp)

        if results.hand_landmarks:
            for hand in results.hand_landmarks:
                draw_hand(frame, hand)

                cv2.putText(
                    frame,
                    "Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Kurdish Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
