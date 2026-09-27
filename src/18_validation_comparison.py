import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("FINAL VALIDATION COMPARISON")
print("=" * 70)

output_dir = "results/evaluation"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Load validation results
# ---------------------------------------------------------

random_results = pd.DataFrame({
    "Validation": ["Random Stratified"],
    "Accuracy": [0.9462],
    "Macro_F1": [0.8082],
    "High_F1": [0.5300]
})

time_results = pd.read_csv(
    "results/evaluation/time_based_validation_results.csv"
)

time_results = time_results.rename(
    columns={
        "Validation": "Original_Validation"
    }
)

time_results["Validation"] = "2025 Time-Based"

time_results = time_results[
    ["Validation", "Accuracy", "Macro_F1", "High_F1"]
]

facility_results = pd.read_csv(
    "results/evaluation/facility_level_validation_results.csv"
)

facility_results = facility_results.rename(
    columns={
        "Validation": "Original_Validation"
    }
)

facility_results["Validation"] = "Unseen Facilities"

facility_results = facility_results[
    ["Validation", "Accuracy", "Macro_F1", "High_F1"]
]

# ---------------------------------------------------------
# 2. Combine results
# ---------------------------------------------------------

comparison = pd.concat(
    [
        random_results,
        time_results,
        facility_results
    ],
    ignore_index=True
)

print("\n" + "=" * 70)
print("VALIDATION COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro_F1": "{:.4f}".format,
            "High_F1": "{:.4f}".format
        }
    )
)

# ---------------------------------------------------------
# 3. Save CSV
# ---------------------------------------------------------

csv_path = os.path.join(
    output_dir,
    "final_validation_comparison.csv"
)

comparison.to_csv(
    csv_path,
    index=False
)

# ---------------------------------------------------------
# 4. Accuracy graph
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Validation"],
    comparison["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Validation Strategy")
plt.title("CatBoost Accuracy Across Validation Strategies")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

accuracy_path = os.path.join(
    output_dir,
    "validation_accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# 5. Macro F1 graph
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Validation"],
    comparison["Macro_F1"]
)

plt.ylabel("Macro F1")
plt.xlabel("Validation Strategy")
plt.title("CatBoost Macro F1 Across Validation Strategies")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

macro_f1_path = os.path.join(
    output_dir,
    "validation_macro_f1_comparison.png"
)

plt.savefig(
    macro_f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# 6. High-class F1 graph
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Validation"],
    comparison["High_F1"]
)

plt.ylabel("High Class F1")
plt.xlabel("Validation Strategy")
plt.title("High-Class F1 Across Validation Strategies")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

high_f1_path = os.path.join(
    output_dir,
    "validation_high_f1_comparison.png"
)

plt.savefig(
    high_f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# 7. Final output
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"CSV:              {csv_path}")
print(f"Accuracy graph:   {accuracy_path}")
print(f"Macro F1 graph:   {macro_f1_path}")
print(f"High F1 graph:    {high_f1_path}")

print("\n" + "=" * 70)
print("FINAL VALIDATION COMPARISON COMPLETED")
print("=" * 70)
