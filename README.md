# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a service.

## Objective

The main objective of this project is to analyze customer data and build a machine learning model to predict customer churn.

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains information such as:

- Customer tenure
- Contract type
- Monthly charges
- Total charges
- Internet services
- Payment method
- Customer churn status

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Models

Two models were used:

1. Logistic Regression
2. Random Forest

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Recall
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── output/
│   ├── churn_distribution.png
│   ├── contract_churn.png
│   ├── tenure_churn.png
│   ├── monthly_charges_churn.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── customer_churn.py
├── requirements.txt
└── README.md

## How to Run

1. Install the required libraries:
   pip install -r requirements.txt

2. Run the project:
   python customer_churn.py