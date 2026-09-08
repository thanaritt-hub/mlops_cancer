import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ============================================================
# Arguments
# ============================================================

preprocessing_run_id = sys.argv[1]
C = float(sys.argv[2])


# ============================================================
# Load data
# ============================================================

d = load_breast_cancer(as_frame=True)
df = d.frame

X = df.drop(columns=["target"])
y = df["target"]


# ============================================================
# Same preprocessing split as previous step
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["target"]
)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///C:/Users/plugt/mlops_cancer/mlflow.db"
)

with mlflow.start_run(run_name="train-evaluate") as run:

    # Log parameters
    mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
    mlflow.log_param("C", C)

    # ========================================================
    # Train model
    # ========================================================

    model = LogisticRegression(
        C=C,
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ========================================================
    # Evaluate
    # ========================================================

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # ========================================================
    # Gate
    # ========================================================

    if accuracy >= 0.95 and roc_auc >= 0.98:

        print("Gate PASSED")

        # Register model
        model_name = "cancer-classifier-prod"

        model_uri = f"runs:/{run.info.run_id}/model"

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name
        )

        # Get registered model version
        client = mlflow.MlflowClient()

        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        latest_version = max(
            versions,
            key=lambda v: int(v.version)
        )

        version = latest_version.version

        # Set alias
        client.set_registered_model_alias(
            model_name,
            "staging",
            version
        )

        print(
            f"Registered {model_name} "
            f"version {version} with alias @staging"
        )

    else:
        print("Gate FAILED")
        print("Model was NOT registered.")