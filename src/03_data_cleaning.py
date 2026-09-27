import pandas as pd
import os

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/Data Center Dataset Mini Project.xlsx"
df = pd.read_excel(file_path)

print("=" * 70)
print("DATA CLEANING & VALIDATION")
print("=" * 70)

print(f"\nOriginal shape: {df.shape}")

# --------------------------------------------------
# 2. Remove completely empty rows
# --------------------------------------------------

empty_rows = df.isna().all(axis=1).sum()
print(f"\nCompletely empty rows: {empty_rows}")

df = df.dropna(how="all")

# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# --------------------------------------------------
# 4. Remove duplicate rows
# --------------------------------------------------

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")

df = df.drop_duplicates()

# --------------------------------------------------
# 5. Validate numerical values
# --------------------------------------------------

print("\nChecking numerical columns...")

numerical_columns = [
    "Estimated_Capacity_MW",
    "PUE",
    "WUE_L_per_kWh",
    "Daily_Electricity_Usage_MWh",
    "Daily_Water_Usage_Gallons"
]

for column in numerical_columns:
    negative_values = (df[column] < 0).sum()
    print(f"{column}: {negative_values} negative values")

# --------------------------------------------------
# 6. Validate Year
# --------------------------------------------------

print("\nYear range:")
print(f"Minimum year: {df['Year'].min()}")
print(f"Maximum year: {df['Year'].max()}")

# --------------------------------------------------
# 7. Validate PUE
# --------------------------------------------------

print("\nPUE range:")
print(f"Minimum PUE: {df['PUE'].min()}")
print(f"Maximum PUE: {df['PUE'].max()}")

# --------------------------------------------------
# 8. Validate Cooling System Types
# --------------------------------------------------

print("\nCooling System Types:")
print(df["Cooling_System_Type"].value_counts())

# --------------------------------------------------
# 9. Validate Facility Types
# --------------------------------------------------

print("\nFacility Types:")
print(df["Facility_Type"].value_counts())

# --------------------------------------------------
# 10. Validate Water Stress Tiers
# --------------------------------------------------

print("\nWater Stress Tiers:")
print(df["Surrounding_Water_Stress_Tier"].value_counts())

# --------------------------------------------------
# 11. Final dataset information
# --------------------------------------------------

print("\nFinal shape:")
print(df.shape)

print("\nFinal missing values:")
print(df.isnull().sum().sum())

print("\nFinal duplicate rows:")
print(df.duplicated().sum())

# --------------------------------------------------
# 12. Save cleaned dataset
# --------------------------------------------------

output_directory = "data/processed"
os.makedirs(output_directory, exist_ok=True)

output_file = "data/processed/cleaned_data.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)
print(f"Cleaned dataset saved to: {output_file}")
