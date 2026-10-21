
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_clean_data


DATA_PATH = "data/customer_churn.csv"
MODEL_PATH = "models/logistic_regression.joblib"
OUTPUT_DIR = Path("models")


def evaluate_model():
    df = load_and_clean_data(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        digits=4,
    )
    auc = roc_auc_score(y_test, probabilities)

    print("CONFUSION MATRIX")
    print(matrix)
    print("\nCLASSIFICATION REPORT")
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as file:
        file.write("CONFUSION MATRIX\n")
        file.write(str(matrix))
        file.write("\n\nCLASSIFICATION REPORT\n")
        file.write(report)
        file.write(f"\nROC-AUC: {auc:.4f}\n")

    print("\nReport saved to models/evaluation_report.txt")


if __name__ == "__main__":
    evaluate_model()
