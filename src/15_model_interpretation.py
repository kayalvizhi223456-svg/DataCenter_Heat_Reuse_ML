import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import os

print("=" * 70)
print("CATBOOST MODEL INTERPRETATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
model_path = "models/catboost_heat_reuse_model.cbm"
data_path = "data/processed/ml_dataset.csv"
output_dir = "results/evaluation"

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------
print("\nLoading CatBoost model...")

model = CatBoostClassifier()
model.load_model(model_path)

print("CatBoost model loaded successfully.")

# ---------------------------------------------------------
# 3. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(data_path)

target = "Heat_Reuse_Category"

X = df.drop(columns=[target])

print(f"\nDataset shape: {df.shape}")
print(f"Number of input features: {X.shape[1]}")

# ---------------------------------------------------------
# 4. Get feature importance
# ---------------------------------------------------------
importance = model.get_feature_importance()

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# ---------------------------------------------------------
# 5. Calculate percentage importance
# ---------------------------------------------------------
importance_df["Importance_Percent"] = (
    importance_df["Importance"]
    / importance_df["Importance"].sum()
) * 100

# Sort highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# ---------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format,
            "Importance_Percent": "{:.2f}%".format
        }
    )
)

# ---------------------------------------------------------
# 7. Save CSV
# ---------------------------------------------------------
csv_path = os.path.join(
    output_dir,
    "catboost_feature_importance.csv"
)

importance_df.to_csv(csv_path, index=False)

print("\nFeature importance saved to:")
print(csv_path)

# ---------------------------------------------------------
# 8. Create visualization
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance for Heat-Reuse Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()

plot_path = os.path.join(
    output_dir,
    "catboost_feature_importance.png"
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.close()

print("\nFeature importance graph saved to:")
print(plot_path)

print("\n" + "=" * 70)
print("MODEL INTERPRETATION COMPLETED")
print("=" * 70)
