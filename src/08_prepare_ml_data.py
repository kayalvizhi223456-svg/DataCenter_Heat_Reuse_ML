import pandas as pd
import os

# --------------------------------------------------
# 1. Load heat reuse dataset
# --------------------------------------------------

input_file = "data/processed/heat_reuse_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("ML DATASET PREPARATION")
print("=" * 70)

print(f"\nOriginal dataset shape: {df.shape}")

# --------------------------------------------------
# 2. Define target
# --------------------------------------------------

target = "Heat_Reuse_Category"

# --------------------------------------------------
# 3. Define features
# --------------------------------------------------

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

X = df[features].copy()
y = df[target].copy()

# --------------------------------------------------
# 4. Display features
# --------------------------------------------------

print("\nFeatures used for ML:")

for feature in features:
    print(f"- {feature}")

print(f"\nTarget:")
print(f"- {target}")

# --------------------------------------------------
# 5. Verify target distribution
# --------------------------------------------------

print("\nTarget distribution:")

print(y.value_counts())

print("\nTarget percentage:")

print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# 6. Check missing values
# --------------------------------------------------

print("\nMissing values in ML features:")

print(X.isnull().sum())

# --------------------------------------------------
# 7. Check data types
# --------------------------------------------------

print("\nFeature data types:")

print(X.dtypes)

# --------------------------------------------------
# 8. Check leakage columns
# --------------------------------------------------

leakage_columns = [
    "Cooling_System_Type",
    "PUE",
    "Estimated_Capacity_MW",
    "Cooling_Score",
    "PUE_Normalized",
    "Capacity_Normalized",
    "Heat_Reuse_Score"
]

print("\nLeakage columns excluded from ML:")

for column in leakage_columns:
    print(f"- {column}")

# --------------------------------------------------
# 9. Save ML dataset
# --------------------------------------------------

output_directory = "data/processed"

os.makedirs(output_directory, exist_ok=True)

ml_dataset = X.copy()
ml_dataset[target] = y

output_file = (
    "data/processed/ml_dataset.csv"
)

ml_dataset.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# 10. Final information
# --------------------------------------------------

print("\n" + "=" * 70)
print("ML DATASET PREPARATION COMPLETED")
print("=" * 70)

print(f"\nML dataset shape: {ml_dataset.shape}")

print(f"\nSaved to:")
print(output_file)
