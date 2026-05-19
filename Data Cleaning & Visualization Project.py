# =========================================================
# DATA CLEANING & VISUALIZATION PROJECT
# =========================================================
# Objective:
# Clean a raw dataset, process missing data, remove outliers,
# detect duplicates, and visualize meaningful insights.
#
# Expected Outcome:
# ✔ Learn Data Preprocessing
# ✔ Learn Data Visualization
# ✔ Learn Storytelling with Data
# =========================================================

# -------------------------
# IMPORT LIBRARIES
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# LOAD DATASET
# -------------------------
# Replace with your dataset file
df = pd.read_csv(r"D:\Data Cleaning & Visualization Project\StudentsPerformance.csv")

print("\n========== ORIGINAL DATASET ==========")
print(df.head())

# -------------------------
# DATASET INFORMATION
# -------------------------
print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

# =========================================================
# DATA CLEANING
# =========================================================

# -------------------------
# REMOVE DUPLICATES
# -------------------------
df.drop_duplicates(inplace=True)

print("\nDuplicates Removed Successfully!")

# -------------------------
# HANDLE MISSING VALUES
# -------------------------

# Numerical Columns
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Categorical Columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values Handled Successfully!")

# =========================================================
# OUTLIER DETECTION & REMOVAL
# =========================================================

print("\n========== OUTLIER REMOVAL ==========")

for col in num_cols:

    # Calculate IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    # Define Limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Remove Outliers
    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]

print("Outliers Removed Successfully!")

# =========================================================
# DATA VISUALIZATION
# =========================================================

sns.set_style("whitegrid")

# -------------------------
# HISTOGRAMS
# -------------------------
for col in num_cols:

    plt.figure(figsize=(8,5))

    sns.histplot(df[col], kde=True)

    plt.title(f"Distribution of {col}")

    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()

# -------------------------
# BOXPLOTS
# -------------------------
for col in num_cols:

    plt.figure(figsize=(8,4))

    sns.boxplot(x=df[col])

    plt.title(f"Boxplot of {col}")

    plt.show()

# -------------------------
# CORRELATION HEATMAP
# -------------------------
plt.figure(figsize=(12,8))

correlation = df[num_cols].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# -------------------------
# BAR CHARTS
# -------------------------
for col in cat_cols:

    plt.figure(figsize=(10,5))

    df[col].value_counts().head(10).plot(
        kind='bar'
    )

    plt.title(f"Top Categories in {col}")

    plt.xlabel(col)
    plt.ylabel("Count")

    plt.show()

# =========================================================
# STORYTELLING WITH DATA
# =========================================================

print("\n========== KEY INSIGHTS ==========")

print("\nSummary Statistics:")
print(df.describe())

print("\nTop Categories:")

for col in cat_cols:
    print(f"\nMost Common in {col}:")
    print(df[col].value_counts().head(3))

# =========================================================
# SAVE CLEANED DATASET
# =========================================================

df.to_csv("cleaned_dataset.csv", index=False)

print("\n========== PROJECT COMPLETED ==========")

print("""
Project Outcomes Achieved:
✔ Missing Values Handled
✔ Duplicate Rows Removed
✔ Outliers Removed
✔ Data Visualized Successfully
✔ Insights Generated
✔ Cleaned Dataset Saved

File Saved As:
cleaned_dataset.csv
""")