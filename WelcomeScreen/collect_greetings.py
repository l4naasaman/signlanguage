
import cv2
import mediapipe as mp
import csv
import time
import os

category = "Greetings"
filename = "greetings_data.csv"

signs = [
    "hello", "goodbye", "good_morning", "good_night",
    "welcome", "see_you", "how_are_you", "nice_to_meet_you"
]

for i, sign in enumerate(signs, 1):
    print(i, "=", sign)

choice = input("Choose a sign number: ")
if not choice.isdigit() or not 1 <= int(choice) <= len(signs):
    print("Invalid choice!")
    raise SystemExit

selected_sign = signs[int(choice) - 1]

folder = os.path.join("Data", category)
os.makedirs(folder, exist_ok=True)
filename = os.path.join(folder, filename)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

sequence_number = 1
if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        saved = [int(r["sequence"]) for r in reader if r.get("sequence")]
        if saved:
            sequence_number = max(saved) + 1

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Could not open camera!")
    raise SystemExit

recording = False
frame_number = 0

with open(filename, "a", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    if os.path.getsize(filename) == 0:
        headers = ["label", "sequence", "frame"]
        for i in range(21):
            headers += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(headers)

    start_time = time.perf_counter()

    with HandLandmarker.create_from_options(options) as detector:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = int((time.perf_counter() - start_time) * 1000)
            results = detector.detect_for_video(mp_image, timestamp)

            if results.hand_landmarks:
                hand = results.hand_landmarks[0]

                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                if recording:
                    row = [selected_sign, sequence_number, frame_number]
                    for point in hand:
                        row.extend([point.x, point.y, point.z])
                    writer.writerow(row)
                    file.flush()
                    frame_number += 1

            status = f"Recording: {selected_sign}" if recording else f"Selected: {selected_sign}"

            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, "Enter=Start | Backspace=Stop | Esc=Quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Greetings Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and not recording:
                recording = True
                frame_number = 0
                print("Recording:", selected_sign)
            elif key == 8 and recording:
                recording = False
                sequence_number += 1
                print("Recording stopped!")
            elif key == 27:
                break

camera.release()
cv2.destroyAllWindows()
print("Saved in:", filename)
