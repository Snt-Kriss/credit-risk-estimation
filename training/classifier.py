import pandas as pd
import numpy as np
import mlflow
import json
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.client import MlflowClient
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path: str) -> DataFrame:
    return pd.read_csv(path)

def train_test_split_data(df: DataFrame, test_size: float):
    df['Risk'] = df['Risk'].replace({'good': 1, 'bad': 0})
    X = df.drop(columns=['Risk'])
    y = df['Risk']
    return train_test_split(X, y, test_size=test_size, random_state=42)

def build_pipeline(categorical_cols, numeric_cols):
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols)
        ]
    )
    # Full pipeline = preprocessing + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(C=1.3, kernel="rbf"))
    ])
    return pipeline

if __name__ == "__main__":
    import logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    logging.info("MlflowClient defined and set tracking uri")

    mlflow.set_experiment("credit-risk-estimator")
    logging.info("Experiment set")

    run_name = "svc-experiment"
    artifact_path = "model"

    df = load_data("./data/german_credit_data.csv")
    logging.info("Data loaded successfully")

    X_train, X_test, y_train, y_test = train_test_split_data(df, test_size=0.2)
    logging.info("Data split")

    categorical_cols = X_train.select_dtypes(include="O").columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude="O").columns.tolist()

    model = build_pipeline(categorical_cols, numeric_cols)

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        logging.info("Pipeline (preprocessor + model) trained")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=np.unique(y_test), index=np.unique(y_test))
        sns.heatmap(df_cm, cmap="Blues", annot=True)
        plt.savefig("confusion_matrix.png")

        metrics = {"recall": recall}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        params = {"C": 1.3, "kernel": "rbf"}
        run_id = mlflow.active_run().info.run_id

        mlflow.sklearn.log_model(sk_model=model, name=artifact_path)
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "recall": recall, "precision": precision})
        mlflow.log_artifact("confusion_matrix.png")

        logging.info("Model and metrics logged to MLflow")

        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(model_uri=model_uri, name="svc-credit-risk-classifier")
        client.transition_model_version_stage(
            name=model_details.name, version=model_details.version, stage="production"
        )
