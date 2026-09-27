import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# 1. Load ML dataset
# --------------------------------------------------

input_file = "data/processed/ml_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("ML PREPROCESSING")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

target = "Heat_Reuse_Category"

X = df.drop(columns=[target])
y = df[target]

print("\nFeatures:")
print(list(X.columns))

print("\nTarget:")
print(target)

# --------------------------------------------------
# 3. Identify numerical features
# --------------------------------------------------

numerical_features = [
    "Year",
    "WUE_L_per_kWh",
    "Daily_Electricity_Usage_MWh",
    "Daily_Water_Usage_Gallons"
]

# --------------------------------------------------
# 4. Identify categorical features
# --------------------------------------------------

categorical_features = [
    "Country",
    "City",
    "Facility_Type",
    "Surrounding_Water_Stress_Tier"
]

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split:")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")

# --------------------------------------------------
# 6. Check target distribution
# --------------------------------------------------

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTraining target percentage:")
print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target percentage:")
print(
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# 7. Numerical preprocessing
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
# 8. Categorical preprocessing
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
# 9. Combine preprocessing
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
# 10. Fit preprocessing on training data only
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

# --------------------------------------------------
# 11. Final message
# --------------------------------------------------

print("\n" + "=" * 70)
print("ML PREPROCESSING COMPLETED")
print("=" * 70)

print("\nImportant:")
print("- Train/test split is stratified.")
print("- Categorical variables are one-hot encoded.")
print("- Unknown categories are handled safely.")
print("- Preprocessing is fitted only on training data.")
