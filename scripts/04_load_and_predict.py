import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer

# MLflow Tracking URI
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/plugt/mlops_cancer/mlflow.db"
)

# Load model from Registry using staging alias
model_uri = "models:/cancer-classifier-prod@staging"
model = mlflow.sklearn.load_model(model_uri)

# Load dataset
d = load_breast_cancer(as_frame=True)

X = d.data
y = d.target

# Class names
class_names = {
    0: "malignant",
    1: "benign"
}

# Get first sample of each class
samples = []

for class_id in [0, 1]:
    index = y[y == class_id].index[0]
    samples.append(index)

# Predict
predictions = model.predict(X.loc[samples])

# Display results
print("=== Prediction Results ===")

for index, prediction in zip(samples, predictions):
    actual = y.loc[index]

    print(
        f"Actual: {class_names[actual]} | "
        f"Predicted: {class_names[prediction]} | "
        f"Correct: {actual == prediction}"
    )