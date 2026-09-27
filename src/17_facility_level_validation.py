import pandas as pd
import os

from catboost import CatBoostClassifier

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 70)
print("FACILITY-LEVEL GENERALIZATION — CATBOOST")
print("=" * 70)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
data_path = "data/processed/heat_reuse_dataset.csv"
output_dir = "results/evaluation"

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(data_path)

print(f"\nDataset shape: {df.shape}")

# ---------------------------------------------------------
# 3. Define features
# ---------------------------------------------------------
target = "Heat_Reuse_Category"

features = [
    "Year",
    "Country",
    "City",
    "Facility_Type",
    "WUE_L_per_kWh",
    "Daily_Electricity_Usage_MWh",
    "Daily_Water_Usage_Gallons",
    "Surrounding_Water_Stress_Tier"
]

group_column = "Facility_ID"

X = df[features].copy()
y = df[target].copy()
groups = df[group_column].copy()

# ---------------------------------------------------------
# 4. Check facility information
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("FACILITY INFORMATION")
print("=" * 70)

print(f"Unique facilities: {groups.nunique():,}")

print(
    f"Average records per facility: "
    f"{len(df) / groups.nunique():.2f}"
)

# ---------------------------------------------------------
# 5. Group-based train/test split
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("GROUP-BASED TRAIN/TEST SPLIT")
print("=" * 70)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

print(f"\nTraining facilities: {train_groups.nunique():,}")
print(f"Testing facilities:  {test_groups.nunique():,}")

# ---------------------------------------------------------
# 6. Verify no facility overlap
# ---------------------------------------------------------
overlap = set(train_groups).intersection(
    set(test_groups)
)

print(f"\nFacility overlap: {len(overlap)}")

if len(overlap) == 0:
    print("SUCCESS: No facility appears in both train and test.")
else:
    print("WARNING: Facility overlap detected!")

# ---------------------------------------------------------
# 7. Target distribution
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING TARGET DISTRIBUTION")
print("=" * 70)

print(y_train.value_counts())

print("\nTraining target percentage:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)

print("\n" + "=" * 70)
print("TEST TARGET DISTRIBUTION")
print("=" * 70)

print(y_test.value_counts())

print("\nTest target percentage:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .round(2)
)

# ---------------------------------------------------------
# 8. Categorical features
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
# 9. Train CatBoost
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING CATBOOST")
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
# 10. Prediction
# ---------------------------------------------------------
print("\nPredicting unseen facilities...")

y_pred = model.predict(X_test)

y_pred = y_pred.flatten()

# ---------------------------------------------------------
# 11. Overall metrics
# ---------------------------------------------------------
accuracy = accuracy_score(
    y_test,
    y_pred
)

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

# ---------------------------------------------------------
# 12. High class metrics
# ---------------------------------------------------------
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
# 13. Print results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("FACILITY-LEVEL VALIDATION RESULTS")
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
# 14. Classification report
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT — UNSEEN FACILITIES")
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
# 15. Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

cm_df = pd.DataFrame(
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

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm_df)

# ---------------------------------------------------------
# 16. Save results
# ---------------------------------------------------------
results = pd.DataFrame({
    "Validation": ["Facility_Level"],
    "Split": ["80% Train Facilities / 20% Unseen Test Facilities"],
    "Training_Samples": [len(X_train)],
    "Testing_Samples": [len(X_test)],
    "Training_Facilities": [train_groups.nunique()],
    "Testing_Facilities": [test_groups.nunique()],
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
    "facility_level_validation_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

# ---------------------------------------------------------
# 17. Save confusion matrix
# ---------------------------------------------------------
cm_path = os.path.join(
    output_dir,
    "facility_level_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

# ---------------------------------------------------------
# 18. Save model
# ---------------------------------------------------------
model_path = (
    "models/catboost_facility_level_model.cbm"
)

model.save_model(model_path)

# ---------------------------------------------------------
# 19. Final output
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Results:          {results_path}")
print(f"Confusion Matrix: {cm_path}")
print(f"Model:            {model_path}")

print("\n" + "=" * 70)
print("FACILITY-LEVEL VALIDATION COMPLETED")
print("=" * 70)
