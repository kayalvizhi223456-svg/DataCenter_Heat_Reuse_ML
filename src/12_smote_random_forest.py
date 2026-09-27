import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = "data/processed/ml_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("SMOTE + RANDOM FOREST")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

target = "Heat_Reuse_Category"

X = df.drop(columns=[target])
y = df[target]


numerical_features = [
    "Year",
    "WUE_L_per_kWh",
    "Daily_Electricity_Usage_MWh",
    "Daily_Water_Usage_Gallons"
]

categorical_features = [
    "Country",
    "City",
    "Facility_Type",
    "Surrounding_Water_Stress_Tier"
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

print("\nOriginal training distribution:")
print(y_train.value_counts())


# ============================================================
# 4. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 5. SMOTE
# ============================================================

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)


# ============================================================
# 6. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. COMPLETE PIPELINE
# ============================================================

pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            smote
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# 8. TRAIN
# ============================================================

print("\nTraining preprocessing + SMOTE + Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 9. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)


# ============================================================
# 10. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("SMOTE RANDOM FOREST PERFORMANCE")
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
# 11. EXTRACT METRICS
# ============================================================

macro_precision = report["macro avg"]["precision"]

macro_recall = report["macro avg"]["recall"]

macro_f1 = report["macro avg"]["f1-score"]

high_precision = report["High"]["precision"]

high_recall = report["High"]["recall"]

high_f1 = report["High"]["f1-score"]


# ============================================================
# 12. CONFUSION MATRIX
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
# 13. SAVE RESULTS
# ============================================================

os.makedirs(
    "results/models",
    exist_ok=True
)


results = pd.DataFrame({
    "Model": ["SMOTE + Random Forest"],
    "Accuracy": [accuracy],
    "Macro_Precision": [macro_precision],
    "Macro_Recall": [macro_recall],
    "Macro_F1": [macro_f1],
    "High_Precision": [high_precision],
    "High_Recall": [high_recall],
    "High_F1": [high_f1]
})


results.to_csv(
    "results/models/smote_random_forest.csv",
    index=False
)


# ============================================================
# 14. DISPLAY SAVED METRICS
# ============================================================

print("\nResults saved to:")
print("results/models/smote_random_forest.csv")

print("\nSaved SMOTE + Random Forest metrics:")

print(f"Accuracy         : {accuracy:.4f}")
print(f"Macro Precision  : {macro_precision:.4f}")
print(f"Macro Recall     : {macro_recall:.4f}")
print(f"Macro F1         : {macro_f1:.4f}")
print(f"High Precision   : {high_precision:.4f}")
print(f"High Recall      : {high_recall:.4f}")
print(f"High F1          : {high_f1:.4f}")


print("\n" + "=" * 70)
print("SMOTE EXPERIMENT COMPLETED")
print("=" * 70)
