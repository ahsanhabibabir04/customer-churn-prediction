# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave a service. Built with Python, scikit-learn, pandas, and Streamlit.

## Project Overview

Customer churn is an important business problem for telecommunications companies. This project explores customer data, preprocesses numerical and categorical features, trains classification models, and evaluates their ability to identify customers who may churn.

The project includes an interactive Streamlit web application that generates churn predictions based on customer information.

## Features

- Data cleaning and preprocessing
- Categorical feature encoding and numerical feature scaling
- Logistic Regression and Random Forest classification
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Automated unit tests using Python's `unittest` framework
- Interactive Streamlit application for customer churn prediction
- Model persistence using Joblib
- Reproducible model evaluation using a stratified train-test split

## Dataset

This project uses the IBM Telco Customer Churn sample dataset.

- **Number of records:** 7,043 customers
- **Original columns:** 21
- **Target variable:** `Churn`
- **Target encoding:** `No = 0`, `Yes = 1`

**Dataset Source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The dataset contains customer demographics, subscribed services, account information, payment methods, and churn status.

## Model Performance

Both models were evaluated using a stratified 20% test split with `random_state=42`.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.81% | 77.08% |
| Precision | 50.43% | 55.92% |
| Recall | 78.34% | 64.44% |
| F1-score | 61.36% | 59.88% |
| ROC-AUC | 84.16% | 82.00% |

### Model Selection

Logistic Regression is used in the Streamlit application because it achieved higher churn recall and ROC-AUC on the evaluated test split.

Recall is particularly important in customer churn prediction because it measures the proportion of actual churners correctly identified by the model. Higher recall can help businesses identify more customers who may be at risk of leaving.

Random Forest achieved higher accuracy and precision on this test split. However, selecting a model should depend on the business objective and the relative costs of false positives and false negatives.

**Note:** These metrics are based on a single held-out test split. They do not guarantee the same performance on future customers or different datasets.

## Technologies Used

- **Python 3.11** — Core programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Scikit-learn** — Data preprocessing, model training, and evaluation
- **Joblib** — Model serialization and loading
- **Streamlit** — Interactive web application
- **unittest** — Automated testing
- **Git and GitHub** — Version control and source code hosting

## Requirements

- Python 3.11 recommended
- Git
- pip
- An internet connection for installing dependencies

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ahsanhabibabir04/customer-churn-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd customer-churn-prediction
```

### 3. Create a Virtual Environment

On Windows PowerShell, run:

```powershell
py -3.11 -m venv .venv
```

### 4. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can use the virtual environment's Python executable directly instead of activating it.

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Streamlit Application

Start the web application using:

```bash
python -m streamlit run app.py
```

After the application starts, open the local URL displayed in the terminal.

Enter the required customer information to generate a churn prediction.

## Train the Models

To train the Logistic Regression and Random Forest classifiers, run:

```bash
python -m src.train_model
```

The training process uses the project's preprocessing and model pipeline to prepare the data and train the classifiers.

## Evaluate the Models

Generate the model evaluation report using:

```bash
python -m src.evaluate_model
```

The evaluation process calculates performance metrics and generates the project's evaluation outputs.

## Run Automated Tests

Execute the test suite with:

```bash
python -m unittest discover -s tests -v
```

The tests help verify dataset properties, preprocessing behavior, and model prediction functionality.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   └── customer_churn.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── model_results.csv
│   └── evaluation_report.txt
├── src/
│   ├── data_preprocessing.py
│   ├── model_pipeline.py
│   ├── train_model.py
│   └── evaluate_model.py
└── tests/
    └── test_model.py
```

## Limitations

- Predictions are estimates and cannot guarantee future customer behavior.
- The dataset is a sample and may not represent every telecommunications business.
- Model performance may vary across customer populations and over time.
- A single train-test split may not fully capture model generalization.
- The model should be validated and monitored before being used for real business decisions.
- Business use should account for the costs of false-positive and false-negative predictions.

## Future Improvements

Potential improvements include:

- Hyperparameter tuning and cross-validation
- Comparing additional classification algorithms
- Improving recall while maintaining acceptable precision
- Adding feature importance and model explainability
- Building customer risk segmentation and retention insights
- Monitoring model performance as new customer data becomes available
- Deploying the application to a public hosting platform

## Learning Outcomes

Through this project, I practiced:

- Cleaning and preprocessing real-world tabular data
- Encoding categorical variables and scaling numerical features
- Building machine learning pipelines with scikit-learn
- Training and comparing classification algorithms
- Evaluating models using multiple performance metrics
- Understanding the business importance of churn recall
- Saving and loading trained models with Joblib
- Developing an interactive Streamlit application
- Writing automated tests with `unittest`
- Organizing a machine learning project into reusable modules
- Managing source code and documentation with Git and GitHub

## Author

**M.A. Ahsan Habib Abir**

Aspiring Computer Science and Artificial Intelligence student with an interest in machine learning, data science, and software development.

## Repository

GitHub: [Customer Churn Prediction](https://github.com/ahsanhabibabir04/customer-churn-prediction)

GitHub Profile: [ahsanhabibabir04](https://github.com/ahsanhabibabir04)

---

If you find this project useful, consider giving the repository a star.
