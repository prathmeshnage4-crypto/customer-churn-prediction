import pandas as pd

# Load the dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Check new dataset shape
print("\nDataset shape after cleaning:")
print(df.shape)


# Check customer churn count
print("\nChurn Count:")
print(df["Churn"].value_counts())

# Check churn percentage
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


import matplotlib.pyplot as plt
import seaborn as sns

# Customer Churn Distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("output/churn_distribution.png")
plt.show()

# Contract Type vs Customer Churn
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("output/contract_churn.png")
plt.show()

# Tenure vs Customer Churn
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.savefig("output/tenure_churn.png")
plt.show()

# Monthly Charges vs Customer Churn
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig("output/monthly_charges_churn.png")
plt.show()

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert Churn into numbers
# Yes = 1
# No = 0
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nChurn values after conversion:")
print(df["Churn"].value_counts())

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    drop_first=True
)

print("\nProcessed Dataset:")
print(df.head())

print("\nNew Dataset Shape:")
print(df.shape)

# Create Features (X) and Target (y)

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures (X) shape:")
print(X.shape)

print("\nTarget (y) shape:")
print(y.shape)

# Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000
)

# Train the model
logistic_model.fit(
    X_train,
    y_train
)

# Make predictions
logistic_pred = logistic_model.predict(X_test)

print("\nLogistic Regression completed.")

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
random_forest_model.fit(
    X_train,
    y_train
)

# Make predictions
random_forest_pred = random_forest_model.predict(X_test)

print("\nRandom Forest completed.")

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score
)

# Logistic Regression metrics
logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_recall = recall_score(
    y_test,
    logistic_pred
)

logistic_probability = logistic_model.predict_proba(
    X_test
)[:, 1]

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probability
)

# Random Forest metrics
rf_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

rf_recall = recall_score(
    y_test,
    random_forest_pred
)

rf_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)

# Display results
print("\n========== MODEL RESULTS ==========")

print("\nLogistic Regression")
print("Accuracy :", round(logistic_accuracy, 4))
print("Recall   :", round(logistic_recall, 4))
print("ROC-AUC  :", round(logistic_roc_auc, 4))

print("\nRandom Forest")
print("Accuracy :", round(rf_accuracy, 4))
print("Recall   :", round(rf_recall, 4))
print("ROC-AUC  :", round(rf_roc_auc, 4))

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(
    y_test,
    random_forest_pred
)

# Display confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("output/confusion_matrix.png")
plt.show()

from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probability
)

# Plot ROC curve
plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {rf_roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.tight_layout()
plt.savefig("output/roc_curve.png")
plt.show()

# Model Comparison

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],
    "Recall": [
        logistic_recall,
        rf_recall
    ],
    "ROC-AUC": [
        logistic_roc_auc,
        rf_roc_auc
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(results.round(4))