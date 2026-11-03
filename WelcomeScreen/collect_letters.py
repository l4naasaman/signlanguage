import cv2
import mediapipe as mp
import csv
import time
import os

# Sorani Kurdish letters
kurdish_letters = [
    "ئ", "ا", "ب", "پ", "ت",
    "ج", "چ", "ح", "خ", "د",
    "ر", "ڕ", "ز", "ژ", "س",
    "ش", "ع", "غ", "ف", "ڤ",
    "ق", "ک", "گ", "ل", "ڵ",
    "م", "ن", "ه", "ە", "و",
    "ۆ", "ی", "ێ"
]

# Keyboard mapping
keys = "123456789abcdefghijklmnopqrstuvwx"

letters = {
    ord(key): letter
    for key, letter in zip(keys, kurdish_letters)
}

for key, letter in letters.items():
    print(chr(key), "=", letter)

filename = "letters_data.csv"

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

# Continue sequence numbering
sequence_number = 1

if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, "r", newline="", encoding="utf-8-sig") as old_file:
        reader = csv.DictReader(old_file)

        saved_sequences = [
            int(row["sequence"])
            for row in reader
            if row.get("sequence")
        ]

        if saved_sequences:
            sequence_number = max(saved_sequences) + 1

# Open camera
camera = cv2.VideoCapture(0)

selected_letter = None
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

                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])

                    cv2.circle(
                        frame, (x, y), 5, (0, 255, 0), -1
                    )

                if recording:

                    row = [
                        selected_letter,
                        sequence_number,
                        frame_number
                    ]

                    for point in hand:
                        row.extend([
                            point.x,
                            point.y,
                            point.z
                        ])

                    writer.writerow(row)
                    file.flush()

                    frame_number += 1

            # Display status
            if selected_letter is None:
                status = "Select a letter: keys 1-9 and a-x"

            elif recording:
                status = f"Recording: {selected_letter}"

            else:
                status = f"Selected: {selected_letter}"

            cv2.putText(
                frame,
                status,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Keys 1-9 and A-X select letters",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Enter = Start | Backspace = Stop | Esc = Quit",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow("Kurdish Letter Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Select a letter
            if key in letters and not recording:
                selected_letter = letters[key]
                print("Selected:", selected_letter)

            # Start recording
            elif key == 13 and not recording:  # Enter
                if selected_letter is not None:
                    recording = True
                    frame_number = 0
                    print("Recording:", selected_letter)

            # Stop recording
            elif key == 8 and recording:  # Backspace
                recording = False
                sequence_number += 1
                print("Recording stopped")

            # Quit
            elif key == 27:  # Esc
                break

camera.release()
cv2.destroyAllWindows()

print("Letter collection finished!")