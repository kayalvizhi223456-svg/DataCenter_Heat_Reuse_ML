import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 70)
print("TIME-BASED VALIDATION — CATBOOST")
print("=" * 70)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
data_path = "data/processed/ml_dataset.csv"
output_dir = "results/evaluation"

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(data_path)

print(f"\nDataset shape: {df.shape}")

# ---------------------------------------------------------
# 3. Time-based split
# ---------------------------------------------------------
train_df = df[df["Year"] <= 2024].copy()
test_df = df[df["Year"] == 2025].copy()

target = "Heat_Reuse_Category"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

print("\n" + "=" * 70)
print("TIME-BASED SPLIT")
print("=" * 70)

print(f"Training years: {train_df['Year'].min()}–{train_df['Year'].max()}")
print(f"Testing year:   {test_df['Year'].min()}")

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

# ---------------------------------------------------------
# 4. Target distribution
# ---------------------------------------------------------
print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTraining target percentage:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)

print("\n2025 test target distribution:")
print(y_test.value_counts())

print("\n2025 test target percentage:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .round(2)
)

# ---------------------------------------------------------
# 5. Identify categorical features
# ---------------------------------------------------------
categorical_features = [
    "Country",
    "City",
    "Facility_Type",
    "Surrounding_Water_Stress_Tier"
]

categorical_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]

# ---------------------------------------------------------
# 6. Train CatBoost
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING CATBOOST ON 2019–2024 DATA")
print("=" * 70)

model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.08,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    class_weights={
        "Low": 1,
        "Medium": 1,
        "High": 10
    },
    random_seed=42,
    verbose=100,
    thread_count=-1
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices
)

print("\nTraining completed.")

# ---------------------------------------------------------
# 7. Predict 2025
# ---------------------------------------------------------
print("\nPredicting 2025 data...")

y_pred = model.predict(X_test)

# CatBoost returns a 2D array
y_pred = y_pred.flatten()

# ---------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

high_precision = precision_score(
    y_test,
    y_pred,
    labels=["High"],
    average="macro",
    zero_division=0
)

high_recall = recall_score(
    y_test,
    y_pred,
    labels=["High"],
    average="macro",
    zero_division=0
)

high_f1 = f1_score(
    y_test,
    y_pred,
    labels=["High"],
    average="macro",
    zero_division=0
)

# ---------------------------------------------------------
# 9. Print results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("2025 TIME-BASED VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")

print("\nHigh Class Performance:")
print(f"High Precision:    {high_precision:.4f}")
print(f"High Recall:       {high_recall:.4f}")
print(f"High F1:           {high_f1:.4f}")

# ---------------------------------------------------------
# 10. Classification report
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT — 2025")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 11. Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low", "Actual Medium", "Actual High"],
    columns=["Predicted Low", "Predicted Medium", "Predicted High"]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX — 2025")
print("=" * 70)

print(cm_df)

# ---------------------------------------------------------
# 12. Save evaluation results
# ---------------------------------------------------------
results = pd.DataFrame({
    "Validation": ["2025_Time_Based"],
    "Training_Period": ["2019-2024"],
    "Test_Period": ["2025"],
    "Accuracy": [accuracy],
    "Macro_Precision": [macro_precision],
    "Macro_Recall": [macro_recall],
    "Macro_F1": [macro_f1],
    "High_Precision": [high_precision],
    "High_Recall": [high_recall],
    "High_F1": [high_f1]
})

results_path = os.path.join(
    output_dir,
    "time_based_validation_results.csv"
)

results.to_csv(results_path, index=False)

# ---------------------------------------------------------
# 13. Save confusion matrix
# ---------------------------------------------------------
cm_path = os.path.join(
    output_dir,
    "time_based_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

# ---------------------------------------------------------
# 14. Save model
# ---------------------------------------------------------
model_path = "models/catboost_time_based_model.cbm"

model.save_model(model_path)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Results:          {results_path}")
print(f"Confusion Matrix: {cm_path}")
print(f"Model:            {model_path}")

print("\n" + "=" * 70)
print("TIME-BASED VALIDATION COMPLETED")
print("=" * 70)
