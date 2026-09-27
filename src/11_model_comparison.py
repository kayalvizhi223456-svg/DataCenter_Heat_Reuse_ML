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
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = "data/processed/ml_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("RANDOM FOREST CLASS WEIGHT COMPARISON")
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
# 5. MODELS TO COMPARE
# ============================================================

models = {

    "RF_No_Weights": None,

    "RF_Balanced": "balanced",

    "RF_High_Weight_10": {
        "Low": 1,
        "Medium": 1,
        "High": 10
    }
}

# ============================================================
# 6. RESULTS STORAGE
# ============================================================

results = []

# ============================================================
# 7. TRAIN AND EVALUATE
# ============================================================

for model_name, class_weight in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

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

    print("\nTraining...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"\nAccuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")

    print("\nHigh Class:")
    print(f"Precision      : {high_precision:.4f}")
    print(f"Recall         : {high_recall:.4f}")
    print(f"F1 Score       : {high_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=["Low", "Medium", "High"],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"]
    )

    print("Confusion Matrix:")

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

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "High_Precision": high_precision,
        "High_Recall": high_recall,
        "High_F1": high_f1
    })


# ============================================================
# 8. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string(index=False)
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

os.makedirs(
    "results/models",
    exist_ok=True
)

output_file = (
    "results/models/random_forest_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nComparison saved to:")
print(output_file)

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)
