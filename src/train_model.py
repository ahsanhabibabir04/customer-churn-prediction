
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data_preprocessing import load_and_clean_data
from src.model_pipeline import build_preprocessor


DATA_PATH = "data/customer_churn.csv"
MODEL_DIR = Path("models")


def train_and_evaluate():
    df = load_and_clean_data(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        results.append(metrics)

        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

        print(f"\n{name.upper().replace('_', ' ')}")
        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

    print("\nSaved models and evaluation results in the models/ folder.")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    train_and_evaluate()
