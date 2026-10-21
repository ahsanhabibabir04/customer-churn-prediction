
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.data_preprocessing import load_and_clean_data
from src.model_pipeline import build_preprocessor


DATA_PATH = "data/customer_churn.csv"


class TestCustomerChurnProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_and_clean_data(DATA_PATH)

    def test_dataset_shape(self):
        self.assertEqual(self.df.shape, (7043, 20))

    def test_no_missing_values(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_target_labels_are_binary(self):
        self.assertEqual(set(self.df["Churn"].unique()), {0, 1})

    def test_total_charges_is_numeric(self):
        self.assertTrue(
            np.issubdtype(self.df["TotalCharges"].dtype, np.number)
        )

    def test_preprocessor_output_has_no_nan(self):
        X = self.df.drop(columns=["Churn"])
        transformed = build_preprocessor().fit_transform(X)

        self.assertFalse(np.isnan(transformed).any())

    def test_logistic_regression_model_predicts(self):
        model_path = Path("models/logistic_regression.joblib")
        self.assertTrue(model_path.exists())

        model = joblib.load(model_path)
        X_sample = self.df.drop(columns=["Churn"]).head(5)
        predictions = model.predict(X_sample)

        self.assertEqual(len(predictions), 5)
        self.assertTrue(set(predictions).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
