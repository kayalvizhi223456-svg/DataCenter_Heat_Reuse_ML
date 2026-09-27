import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

file_path = "data/processed/cleaned_data.csv"
df = pd.read_csv(file_path)

# Create output folder
os.makedirs("results/figures", exist_ok=True)

print("=" * 70)
print("HEAT-REUSE-FOCUSED EDA")
print("=" * 70)

# --------------------------------------------------
# 2. Cooling system analysis
# --------------------------------------------------

print("\n1. COOLING SYSTEM ANALYSIS")
print("-" * 70)

cooling_summary = df.groupby("Cooling_System_Type").agg(
    Data_Centers=("Facility_ID", "count"),
    Average_PUE=("PUE", "mean"),
    Average_Capacity_MW=("Estimated_Capacity_MW", "mean"),
    Average_Electricity_MWh=("Daily_Electricity_Usage_MWh", "mean"),
    Average_Water_Gallons=("Daily_Water_Usage_Gallons", "mean")
)

print(cooling_summary.round(2))

# --------------------------------------------------
# 3. Facility type analysis
# --------------------------------------------------

print("\n2. FACILITY TYPE ANALYSIS")
print("-" * 70)

facility_summary = df.groupby("Facility_Type").agg(
    Data_Centers=("Facility_ID", "count"),
    Average_PUE=("PUE", "mean"),
    Average_Capacity_MW=("Estimated_Capacity_MW", "mean"),
    Average_Electricity_MWh=("Daily_Electricity_Usage_MWh", "mean")
)

print(facility_summary.round(2))

# --------------------------------------------------
# 4. Water stress analysis
# --------------------------------------------------

print("\n3. WATER STRESS ANALYSIS")
print("-" * 70)

water_summary = df.groupby("Surrounding_Water_Stress_Tier").agg(
    Data_Centers=("Facility_ID", "count"),
    Average_WUE=("WUE_L_per_kWh", "mean"),
    Average_Water_Usage=("Daily_Water_Usage_Gallons", "mean"),
    Average_PUE=("PUE", "mean")
)

print(water_summary.round(2))

# --------------------------------------------------
# 5. Cooling System vs PUE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Cooling_System_Type",
    y="PUE"
)

plt.title("PUE Distribution by Cooling System Type")
plt.xlabel("Cooling System Type")
plt.ylabel("PUE")

plt.tight_layout()

plt.savefig(
    "results/figures/cooling_vs_pue.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 6. Cooling System vs Capacity
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Cooling_System_Type",
    y="Estimated_Capacity_MW"
)

plt.title("Data Center Capacity by Cooling System Type")
plt.xlabel("Cooling System Type")
plt.ylabel("Estimated Capacity (MW)")

plt.tight_layout()

plt.savefig(
    "results/figures/cooling_vs_capacity.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 7. Cooling System vs Electricity Usage
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Cooling_System_Type",
    y="Daily_Electricity_Usage_MWh"
)

plt.title("Electricity Usage by Cooling System Type")
plt.xlabel("Cooling System Type")
plt.ylabel("Daily Electricity Usage (MWh)")

plt.tight_layout()

plt.savefig(
    "results/figures/cooling_vs_electricity.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 8. PUE by Facility Type
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Facility_Type",
    y="PUE"
)

plt.title("PUE Distribution by Facility Type")
plt.xlabel("Facility Type")
plt.ylabel("PUE")

plt.tight_layout()

plt.savefig(
    "results/figures/facility_vs_pue.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 9. Capacity vs PUE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(min(10000, len(df)), random_state=42),
    x="Estimated_Capacity_MW",
    y="PUE",
    alpha=0.4
)

plt.title("Data Center Capacity vs PUE")
plt.xlabel("Estimated Capacity (MW)")
plt.ylabel("PUE")

plt.tight_layout()

plt.savefig(
    "results/figures/capacity_vs_pue.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# 10. Cooling system counts
# --------------------------------------------------

cooling_counts = df["Cooling_System_Type"].value_counts()

print("\nCooling System Counts:")
print(cooling_counts)

# --------------------------------------------------
# 11. Final message
# --------------------------------------------------

print("\n" + "=" * 70)
print("HEAT-REUSE EDA COMPLETED")
print("=" * 70)

print("\nNew charts saved in:")
print("results/figures/")
