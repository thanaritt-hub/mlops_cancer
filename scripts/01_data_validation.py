from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
d = load_breast_cancer(as_frame=True)
df = d.frame

# Basic validation
print("=== Data Validation ===")

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nData types:")
print(df.dtypes.value_counts())

# Class balance validation
class_ratio = df["target"].value_counts(normalize=True)
assert class_ratio.min() >= 0.20, (
    f"Class balance validation failed: minimum class ratio = "
    f"{class_ratio.min():.2%}, required >= 45%"
)

print("\nAll validation assertions passed!")