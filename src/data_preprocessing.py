
import pandas as pd


def load_and_clean_data(file_path):
    """Load the customer churn dataset and clean it."""
    df = pd.read_csv(file_path)

    # Remove the customer identifier; it is not a predictive feature.
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric.
    # Blank values belong to customers with zero tenure.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", "0")
    )

    # Convert the target to numeric labels: No = 0, Yes = 1.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df
