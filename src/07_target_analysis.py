import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# Load heat reuse dataset
# --------------------------------------------------

file_path = "data/processed/heat_reuse_dataset.csv"
df = pd.read_csv(file_path)

print("=" * 70)
print("HEAT REUSE TARGET ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# --------------------------------------------------
# Target distribution
# --------------------------------------------------

print("\nTarget Distribution:")
print(df["Heat_Reuse_Category"].value_counts())

print("\nTarget Percentage:")
print(
    df["Heat_Reuse_Category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# Score distribution
# --------------------------------------------------

print("\nScore Statistics:")
print(df["Heat_Reuse_Score"].describe())

# --------------------------------------------------
# Create figure directory
# --------------------------------------------------

os.makedirs("results/figures", exist_ok=True)

# --------------------------------------------------
# Category count plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Heat_Reuse_Category",
    order=["Low", "Medium", "High"]
)

plt.title("Heat Reuse Category Distribution")
plt.xlabel("Heat Reuse Category")
plt.ylabel("Number of Data Centers")

plt.tight_layout()

plt.savefig(
    "results/figures/heat_reuse_category_distribution.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# Score distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Heat_Reuse_Score",
    bins=30,
    kde=True
)

plt.axvline(
    0.33,
    linestyle="--",
    label="Low/Medium Threshold"
)

plt.axvline(
    0.66,
    linestyle="--",
    label="Medium/High Threshold"
)

plt.title("Heat Reuse Score Distribution")
plt.xlabel("Heat Reuse Score")
plt.ylabel("Number of Data Centers")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/figures/heat_reuse_score_distribution.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# Target by cooling system
# --------------------------------------------------

print("\nHeat Reuse Category by Cooling System:")

cooling_target = pd.crosstab(
    df["Cooling_System_Type"],
    df["Heat_Reuse_Category"],
    normalize="index"
) * 100

print(cooling_target.round(2))

# --------------------------------------------------
# Target by facility type
# --------------------------------------------------

print("\nHeat Reuse Category by Facility Type:")

facility_target = pd.crosstab(
    df["Facility_Type"],
    df["Heat_Reuse_Category"],
    normalize="index"
) * 100

print(facility_target.round(2))

# --------------------------------------------------
# Target by water stress
# --------------------------------------------------

print("\nHeat Reuse Category by Water Stress:")

water_target = pd.crosstab(
    df["Surrounding_Water_Stress_Tier"],
    df["Heat_Reuse_Category"],
    normalize="index"
) * 100

print(water_target.round(2))

print("\n" + "=" * 70)
print("TARGET ANALYSIS COMPLETED")
print("=" * 70)
