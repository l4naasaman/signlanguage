
import pandas as pd

# Read the recorded data
data = pd.read_csv("hello_data.csv")

# Show basic information
print("Total frames:", len(data))
print("Total recordings:", data["sequence"].nunique())

# Count frames in each recording
print("\nFrames per recording:")
print(data.groupby("sequence").size())

# Check for missing values
print("\nMissing values:", data.isnull().sum().sum())

# Check the labels
print("\nLabels:", data["label"].unique())
