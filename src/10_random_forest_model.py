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

# --------------------------------------------------
# 1. Load ML dataset
# --------------------------------------------------

input_file = "data/processed/ml_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("RANDOM FOREST CLASSIFICATION")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

target = "Heat_Reuse_Category"

X = df.drop(columns=[target])
y = df[target]

# --------------------------------------------------
# 3. Define features
# --------------------------------------------------

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

# --------------------------------------------------
# 4. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# --------------------------------------------------
# 5. Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# --------------------------------------------------
# 6. Categorical preprocessing
# --------------------------------------------------

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

# --------------------------------------------------
# 7. Combine preprocessing
# --------------------------------------------------

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

# --------------------------------------------------
# 8. Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# 9. Complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)

# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")

# --------------------------------------------------
# 11. Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)

# --------------------------------------------------
# 12. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")

# --------------------------------------------------
# 13. Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"],
        zero_division=0
    )
)

# --------------------------------------------------
# 14. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Predicted Low", "Predicted Medium", "Predicted High"]
    )
)

# --------------------------------------------------
# 15. Save results
# --------------------------------------------------

os.makedirs(
    "results/models",
    exist_ok=True
)

results = {
    "Model": ["Random Forest"],
    "Accuracy": [accuracy]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/models/model_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/models/model_results.csv")

print("\n" + "=" * 70)
print("RANDOM FOREST MODEL COMPLETED")
print("=" * 70)
