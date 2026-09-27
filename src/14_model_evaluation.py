import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. LOAD RESULTS
# ============================================================

print("=" * 70)
print("FINAL MODEL EVALUATION AND COMPARISON")
print("=" * 70)


rf = pd.read_csv(
    "results/models/random_forest_comparison.csv"
)

smote = pd.read_csv(
    "results/models/smote_random_forest.csv"
)

catboost = pd.read_csv(
    "results/models/catboost_results.csv"
)


print("\nAll model result files loaded successfully.")


# ============================================================
# 2. COMMON METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "High_Precision",
    "High_Recall",
    "High_F1"
]


# ============================================================
# 3. RANDOM FOREST RESULTS
# ============================================================

rf_final = rf[
    [
        "Model",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "High_Precision",
        "High_Recall",
        "High_F1"
    ]
].copy()


# ============================================================
# 4. SMOTE RESULT
# ============================================================

smote_final = smote[
    [
        "Model",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "High_Precision",
        "High_Recall",
        "High_F1"
    ]
].copy()


# ============================================================
# 5. CATBOOST RESULT
# ============================================================

catboost_final = catboost[
    [
        "Model",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "High_Precision",
        "High_Recall",
        "High_F1"
    ]
].copy()


# ============================================================
# 6. COMBINE ALL MODELS
# ============================================================

comparison = pd.concat(
    [
        rf_final,
        smote_final,
        catboost_final
    ],
    ignore_index=True
)


# ============================================================
# 7. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "results/evaluation",
    exist_ok=True
)


# ============================================================
# 8. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. SAVE FINAL COMPARISON
# ============================================================

comparison_file = (
    "results/evaluation/final_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print("\nFinal comparison saved to:")
print(comparison_file)


# ============================================================
# 10. ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Accuracy Comparison of Machine Learning Models")

plt.xlabel("Model")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

accuracy_graph = (
    "results/evaluation/final_accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300
)

plt.show()


# ============================================================
# 11. MACRO F1 COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["Macro_F1"]
)

plt.title("Macro F1-Score Comparison")

plt.xlabel("Model")

plt.ylabel("Macro F1-Score")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

macro_f1_graph = (
    "results/evaluation/final_macro_f1_comparison.png"
)

plt.savefig(
    macro_f1_graph,
    dpi=300
)

plt.show()


# ============================================================
# 12. HIGH-CLASS PRECISION / RECALL / F1
# ============================================================

x = range(len(comparison))

width = 0.25

plt.figure(figsize=(12, 6))

plt.bar(
    [i - width for i in x],
    comparison["High_Precision"],
    width=width,
    label="High Precision"
)

plt.bar(
    x,
    comparison["High_Recall"],
    width=width,
    label="High Recall"
)

plt.bar(
    [i + width for i in x],
    comparison["High_F1"],
    width=width,
    label="High F1"
)

plt.title(
    "High-Class Performance Comparison"
)

plt.xlabel("Model")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(
    list(x),
    comparison["Model"],
    rotation=20,
    ha="right"
)

plt.legend()

plt.tight_layout()

high_graph = (
    "results/evaluation/final_high_class_comparison.png"
)

plt.savefig(
    high_graph,
    dpi=300
)

plt.show()


# ============================================================
# 13. SAVE A REPORT-FRIENDLY TABLE
# ============================================================

report_table = comparison.copy()

for column in metrics:
    report_table[column] = (
        report_table[column] * 100
    ).round(2)


report_file = (
    "results/evaluation/model_comparison_percentage.csv"
)

report_table.to_csv(
    report_file,
    index=False
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION FILES GENERATED")
print("=" * 70)

print("\n1.", comparison_file)
print("2.", accuracy_graph)
print("3.", macro_f1_graph)
print("4.", high_graph)
print("5.", report_file)

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 70)
