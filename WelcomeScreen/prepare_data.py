
import os
import glob
import json
import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 100
FEATURES = 63

# Find the letter CSV and all category CSV files
csv_files = []

old_files = [
    "letters_data.csv",
    "kurdistan_data.csv",
    "hello_data.csv"
]

for file in old_files:
    if os.path.exists(file):
        csv_files.append(file)

csv_files.extend(
    glob.glob(os.path.join("Data", "**", "*.csv"), recursive=True)
)

csv_files = sorted(set(csv_files))

if not csv_files:
    raise FileNotFoundError("No CSV files found!")

print("CSV files found:")
for path in csv_files:
    print(path)

X = []
y = []

# Read and prepare each CSV
for path in csv_files:
    print("\nReading:", path)

    data = pd.read_csv(path, encoding="utf-8-sig")

    required_columns = (
        ["label", "sequence", "frame"]
        + [
            f"{axis}{i}"
            for i in range(21)
            for axis in ["x", "y", "z"]
        ]
    )

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        print("Skipping file because columns are missing:", missing_columns)
        continue

    # Each sequence is one recording
    for (label, sequence), group in data.groupby(
        ["label", "sequence"]
    ):
        group = group.sort_values("frame")

        values = group[
            [
                f"{axis}{i}"
                for i in range(21)
                for axis in ["x", "y", "z"]
            ]
        ].to_numpy(dtype=np.float32)

        # Skip recordings with missing or invalid values
        if len(values) < 2 or not np.isfinite(values).all():
            continue

        # Convert every recording to 100 frames
        old_positions = np.linspace(0, 1, len(values))
        new_positions = np.linspace(0, 1, SEQUENCE_LENGTH)

        resized = np.zeros(
            (SEQUENCE_LENGTH, FEATURES),
            dtype=np.float32
        )

        for feature in range(FEATURES):
            resized[:, feature] = np.interp(
                new_positions,
                old_positions,
                values[:, feature]
            )

        X.append(resized)
        y.append(str(label))

if not X:
    raise ValueError("No valid recordings were found!")

X = np.array(X, dtype=np.float32)
labels = sorted(set(y))

label_map = {
    label: index
    for index, label in enumerate(labels)
}

y_numbers = np.array(
    [label_map[label] for label in y],
    dtype=np.int32
)

# Save prepared data
np.save("signs_X.npy", X)
np.save("signs_y.npy", y_numbers)

with open("labels.json", "w", encoding="utf-8") as file:
    json.dump(labels, file, ensure_ascii=False, indent=4)

print("\nPreparation complete!")
print("Data shape:", X.shape)
print("Labels shape:", y_numbers.shape)
print("\nRecordings per sign:")

for label in labels:
    print(label, "=", y.count(label))

print("\nTotal recordings:", len(X))
print("Total different signs:", len(labels))
