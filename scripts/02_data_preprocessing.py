from pathlib import Path

import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
d = load_breast_cancer(as_frame=True)
df = d.frame

# Split data: 75% train, 25% test
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["target"]
)

# MLflow tracking
mlflow.set_tracking_uri("sqlite:///C:/Users/plugt/mlops_cancer/mlflow.db")

with mlflow.start_run(run_name="preprocessing") as run:

    # Log preprocessing information
    mlflow.log_param("test_size", 0.25)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("stratify", True)

    # Log number of rows
    mlflow.log_metric("training_set_rows", len(train_df))
    mlflow.log_metric("test_set_rows", len(test_df))

    # Save datasets as artifacts
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    train_path = artifact_dir / "train.csv"
    test_path = artifact_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    mlflow.log_artifact(str(train_path), artifact_path="data")
    mlflow.log_artifact(str(test_path), artifact_path="data")

    print("Preprocessing Run ID:", run.info.run_id)
    print("training_set_rows =", len(train_df))
    print("test_set_rows =", len(test_df))