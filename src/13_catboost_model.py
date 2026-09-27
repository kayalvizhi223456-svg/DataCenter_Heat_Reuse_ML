import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = "data/processed/ml_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("CATBOOST CLASSIFICATION")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

target = "Heat_Reuse_Category"

X = df.drop(columns=[target])
y = df[target]


categorical_features = [
    "Country",
    "City",
    "Facility_Type",
    "Surrounding_Water_Stress_Tier"
]

categorical_indices = [
    X.columns.get_loc(column)
    for column in categorical_features
]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. CATBOOST MODEL
# ============================================================

model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.08,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=42,
    verbose=100,
    thread_count=-1,

    # Explicitly assign weight to each class
    class_weights={
        "Low": 1,
        "Medium": 1,
        "High": 10
    }
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining CatBoost...")

model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

print("\nTraining completed.")


# ============================================================
# 6. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_pred = y_pred.flatten()


# ============================================================
# 7. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CATBOOST PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"],
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"],
        zero_division=0
    )
)


# ============================================================
# 8. EXTRACT METRICS
# ============================================================

macro_precision = report["macro avg"]["precision"]
macro_recall = report["macro avg"]["recall"]
macro_f1 = report["macro avg"]["f1-score"]

high_precision = report["High"]["precision"]
high_recall = report["High"]["recall"]
high_f1 = report["High"]["f1-score"]


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        cm,
        index=[
            "Actual Low",
            "Actual Medium",
            "Actual High"
        ],
        columns=[
            "Predicted Low",
            "Predicted Medium",
            "Predicted High"
        ]
    )
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

os.makedirs(
    "results/models",
    exist_ok=True
)

results = pd.DataFrame({
    "Model": ["CatBoost"],
    "Accuracy": [accuracy],
    "Macro_Precision": [macro_precision],
    "Macro_Recall": [macro_recall],
    "Macro_F1": [macro_f1],
    "High_Precision": [high_precision],
    "High_Recall": [high_recall],
    "High_F1": [high_f1]
})

results.to_csv(
    "results/models/catboost_results.csv",
    index=False
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save_model(
    "models/catboost_heat_reuse_model.cbm"
)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\nResults saved to:")
print("results/models/catboost_results.csv")

print("\nModel saved to:")
print("models/catboost_heat_reuse_model.cbm")

print("\nSaved CatBoost metrics:")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Macro Precision  : {macro_precision:.4f}")
print(f"Macro Recall     : {macro_recall:.4f}")
print(f"Macro F1         : {macro_f1:.4f}")
print(f"High Precision   : {high_precision:.4f}")
print(f"High Recall      : {high_recall:.4f}")
print(f"High F1          : {high_f1:.4f}")


print("\n" + "=" * 70)
print("CATBOOST EXPERIMENT COMPLETED")
print("=" * 70)
