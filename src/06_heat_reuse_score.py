import pandas as pd
import os

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

input_file = "data/processed/cleaned_data.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("HEAT REUSE SCORE CALCULATION")
print("=" * 70)

print(f"\nInput dataset shape: {df.shape}")

# --------------------------------------------------
# 2. Cooling System Score
# --------------------------------------------------

cooling_score_map = {
    "Liquid Cooled": 1.0,
    "Evaporative": 0.6,
    "Air Cooled": 0.2
}

df["Cooling_Score"] = df["Cooling_System_Type"].map(
    cooling_score_map
)

# Check for unmapped cooling systems
unmapped = df["Cooling_Score"].isna().sum()

print("\nCooling systems not mapped:", unmapped)

if unmapped > 0:
    print("\nUnmapped cooling system values:")
    print(
        df.loc[
            df["Cooling_Score"].isna(),
            "Cooling_System_Type"
        ].unique()
    )
    raise ValueError(
        "Some Cooling_System_Type values do not have a score."
    )

# --------------------------------------------------
# 3. PUE Normalization
# --------------------------------------------------

pue_min = df["PUE"].min()
pue_max = df["PUE"].max()

df["PUE_Normalized"] = (
    (df["PUE"] - pue_min)
    / (pue_max - pue_min)
)

# --------------------------------------------------
# 4. Capacity Normalization
# --------------------------------------------------

capacity_min = df["Estimated_Capacity_MW"].min()
capacity_max = df["Estimated_Capacity_MW"].max()

df["Capacity_Normalized"] = (
    (df["Estimated_Capacity_MW"] - capacity_min)
    / (capacity_max - capacity_min)
)

# --------------------------------------------------
# 5. Heat Reuse Score
# --------------------------------------------------

df["Heat_Reuse_Score"] = (
    0.50 * df["Cooling_Score"]
    + 0.30 * df["PUE_Normalized"]
    + 0.20 * df["Capacity_Normalized"]
)

# Round score to 2 decimal places
df["Heat_Reuse_Score"] = df["Heat_Reuse_Score"].round(2)

# --------------------------------------------------
# 6. Heat Reuse Category
# --------------------------------------------------


def assign_category(score):

    if score >= 0.66:
        return "High"

    elif score >= 0.33:
        return "Medium"

    else:
        return "Low"


df["Heat_Reuse_Category"] = (
    df["Heat_Reuse_Score"].apply(assign_category)
)

# --------------------------------------------------
# 7. Display methodology information
# --------------------------------------------------

print("\nNormalization ranges:")
print(f"PUE minimum       : {pue_min}")
print(f"PUE maximum       : {pue_max}")
print(f"Capacity minimum  : {capacity_min} MW")
print(f"Capacity maximum  : {capacity_max} MW")

# --------------------------------------------------
# 8. Score statistics
# --------------------------------------------------

print("\nHeat Reuse Score Statistics:")
print(df["Heat_Reuse_Score"].describe())

# --------------------------------------------------
# 9. Category distribution
# --------------------------------------------------

print("\nHeat Reuse Category Distribution:")
category_counts = df["Heat_Reuse_Category"].value_counts()

print(category_counts)

# --------------------------------------------------
# 10. Category percentages
# --------------------------------------------------

print("\nHeat Reuse Category Percentage:")

category_percentages = (
    df["Heat_Reuse_Category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(category_percentages)

# --------------------------------------------------
# 11. Cooling score distribution
# --------------------------------------------------

print("\nCooling Score Distribution:")
print(df["Cooling_Score"].value_counts().sort_index())

# --------------------------------------------------
# 12. Show sample results
# --------------------------------------------------

print("\nSample Heat Reuse Results:")

display_columns = [
    "Facility_ID",
    "Cooling_System_Type",
    "Estimated_Capacity_MW",
    "PUE",
    "Cooling_Score",
    "PUE_Normalized",
    "Capacity_Normalized",
    "Heat_Reuse_Score",
    "Heat_Reuse_Category"
]

print(df[display_columns].head(10).to_string(index=False))

# --------------------------------------------------
# 13. Save dataset
# --------------------------------------------------

output_directory = "data/processed"
os.makedirs(output_directory, exist_ok=True)

output_file = (
    "data/processed/heat_reuse_dataset.csv"
)

df.to_csv(output_file, index=False)

# --------------------------------------------------
# 14. Final confirmation
# --------------------------------------------------

print("\n" + "=" * 70)
print("HEAT REUSE SCORE CALCULATION COMPLETED")
print("=" * 70)

print(f"\nOutput file:")
print(output_file)

print(f"\nFinal dataset shape: {df.shape}")

print("\nNew columns added:")
print("- Cooling_Score")
print("- PUE_Normalized")
print("- Capacity_Normalized")
print("- Heat_Reuse_Score")
print("- Heat_Reuse_Category")
