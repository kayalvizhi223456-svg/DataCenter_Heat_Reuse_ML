import pandas as pd

# Load dataset
file_path = "data/Data Center Dataset Mini Project.xlsx"
df = pd.read_excel(file_path)

print("=" * 70)
print("DETAILED DATASET PROFILING")
print("=" * 70)

# --------------------------------------------------
# 1. Dataset size
# --------------------------------------------------
print("\n1. DATASET SIZE")
print("-" * 70)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# --------------------------------------------------
# 2. Numerical columns
# --------------------------------------------------
print("\n2. NUMERICAL COLUMNS")
print("-" * 70)

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numerical_columns:
    print(f"- {column}")

# --------------------------------------------------
# 3. Categorical columns
# --------------------------------------------------
print("\n3. CATEGORICAL COLUMNS")
print("-" * 70)

categorical_columns = df.select_dtypes(include=["object", "string"]).columns

for column in categorical_columns:
    print(f"- {column}")

# --------------------------------------------------
# 4. Unique values
# --------------------------------------------------
print("\n4. UNIQUE VALUES")
print("-" * 70)

for column in categorical_columns:
    print(f"\n{column}: {df[column].nunique():,} unique values")

# --------------------------------------------------
# 5. Important categorical distributions
# --------------------------------------------------
print("\n5. CATEGORICAL DISTRIBUTIONS")
print("-" * 70)

for column in [
    "Facility_Type",
    "Cooling_System_Type",
    "Surrounding_Water_Stress_Tier"
]:
    print(f"\n{column}:")
    print(df[column].value_counts())

# --------------------------------------------------
# 6. Year distribution
# --------------------------------------------------
print("\n6. RECORDS BY YEAR")
print("-" * 70)
print(df["Year"].value_counts().sort_index())

# --------------------------------------------------
# 7. Numerical range
# --------------------------------------------------
print("\n7. NUMERICAL RANGES")
print("-" * 70)

for column in numerical_columns:
    print(
        f"{column}: "
        f"Min = {df[column].min():,.3f}, "
        f"Max = {df[column].max():,.3f}"
    )

# --------------------------------------------------
# 8. Duplicate Facility IDs
# --------------------------------------------------
print("\n8. FACILITY ID ANALYSIS")
print("-" * 70)

unique_facilities = df["Facility_ID"].nunique()
total_records = len(df)

print(f"Unique Facility IDs : {unique_facilities:,}")
print(f"Total Records       : {total_records:,}")

if unique_facilities == total_records:
    print("Every record has a unique Facility_ID.")
else:
    print("Some Facility_IDs occur multiple times.")

# --------------------------------------------------
# 9. Final message
# --------------------------------------------------
print("\n" + "=" * 70)
print("DATASET PROFILING COMPLETED")
print("=" * 70)
