import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

file_path = "data/processed/cleaned_data.csv"
df = pd.read_csv(file_path)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# --------------------------------------------------
# 2. Create results folders
# --------------------------------------------------

os.makedirs("results/figures", exist_ok=True)

# --------------------------------------------------
# 3. Basic numerical summary
# --------------------------------------------------

print("\nNumerical Summary:")
print(df.describe())

# --------------------------------------------------
# 4. Records by year
# --------------------------------------------------

print("\nRecords by Year:")
print(df["Year"].value_counts().sort_index())

plt.figure(figsize=(10, 6))

year_counts = df["Year"].value_counts().sort_index()

year_counts.plot(kind="bar")

plt.title("Number of Data Center Records by Year")
plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/figures/records_by_year.png", dpi=300)
plt.show()

# --------------------------------------------------
# 5. Facility Type distribution
# --------------------------------------------------

print("\nFacility Type Distribution:")
print(df["Facility_Type"].value_counts())

plt.figure(figsize=(8, 6))

df["Facility_Type"].value_counts().plot(kind="bar")

plt.title("Data Centers by Facility Type")
plt.xlabel("Facility Type")
plt.ylabel("Number of Data Centers")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("results/figures/facility_type_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 6. Cooling System distribution
# --------------------------------------------------

print("\nCooling System Distribution:")
print(df["Cooling_System_Type"].value_counts())

plt.figure(figsize=(8, 6))

df["Cooling_System_Type"].value_counts().plot(kind="bar")

plt.title("Data Centers by Cooling System Type")
plt.xlabel("Cooling System Type")
plt.ylabel("Number of Data Centers")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("results/figures/cooling_system_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7. Water Stress distribution
# --------------------------------------------------

print("\nWater Stress Distribution:")
print(df["Surrounding_Water_Stress_Tier"].value_counts())

plt.figure(figsize=(8, 6))

df["Surrounding_Water_Stress_Tier"].value_counts().plot(kind="bar")

plt.title("Data Centers by Surrounding Water Stress")
plt.xlabel("Water Stress Tier")
plt.ylabel("Number of Data Centers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("results/figures/water_stress_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 8. PUE Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(df["PUE"], bins=30)

plt.title("Distribution of PUE")
plt.xlabel("PUE")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("results/figures/pue_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 9. Capacity Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(df["Estimated_Capacity_MW"], bins=40)

plt.title("Distribution of Data Center Capacity")
plt.xlabel("Estimated Capacity (MW)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("results/figures/capacity_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 10. Electricity Usage Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(df["Daily_Electricity_Usage_MWh"], bins=40)

plt.title("Daily Electricity Usage Distribution")
plt.xlabel("Daily Electricity Usage (MWh)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("results/figures/electricity_usage_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 11. Capacity vs Electricity Usage
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Estimated_Capacity_MW"],
    df["Daily_Electricity_Usage_MWh"],
    alpha=0.3
)

plt.title("Data Center Capacity vs Daily Electricity Usage")
plt.xlabel("Estimated Capacity (MW)")
plt.ylabel("Daily Electricity Usage (MWh)")

plt.tight_layout()
plt.savefig("results/figures/capacity_vs_electricity.png", dpi=300)
plt.show()

# --------------------------------------------------
# 12. PUE vs Electricity Usage
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["PUE"],
    df["Daily_Electricity_Usage_MWh"],
    alpha=0.3
)

plt.title("PUE vs Daily Electricity Usage")
plt.xlabel("PUE")
plt.ylabel("Daily Electricity Usage (MWh)")

plt.tight_layout()
plt.savefig("results/figures/pue_vs_electricity.png", dpi=300)
plt.show()

# --------------------------------------------------
# 13. Correlation Heatmap
# --------------------------------------------------

numerical_columns = [
    "Estimated_Capacity_MW",
    "PUE",
    "WUE_L_per_kWh",
    "Daily_Electricity_Usage_MWh",
    "Daily_Water_Usage_Gallons"
]

correlation_matrix = df[numerical_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Between Numerical Variables")
plt.tight_layout()

plt.savefig(
    "results/figures/correlation_heatmap.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 14. Final message
# --------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("\nCharts saved in:")
print("results/figures/")
