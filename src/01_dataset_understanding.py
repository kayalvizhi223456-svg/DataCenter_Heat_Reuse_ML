import pandas as pd

# Dataset path
file_path = "data/Data Center Dataset Mini Project.xlsx"

# Load dataset
df = pd.read_excel(file_path)

# Basic information
print("=" * 60)
print("DATASET UNDERSTANDING")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nBasic Statistics:")
print(df.describe())
