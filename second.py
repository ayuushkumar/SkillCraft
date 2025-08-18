# Titanic Survival Prediction - with Visualizations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("titanic")

# Data cleaning
df = df.drop(
    ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], 
    axis=1
)

# Fill missing values safely
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# -------------------------------
# 1. Survival Rate by Gender
# -------------------------------
plt.figure(figsize=(6,4))
sns.barplot(x="sex", y="survived", data=df, hue="sex", palette="Set2", legend=False)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

# -------------------------------
# 2. Survival Rate by Passenger Class
# -------------------------------
plt.figure(figsize=(6,4))
sns.barplot(x="pclass", y="survived", data=df, hue="pclass", palette="pastel", legend=False)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# -------------------------------
# 3. Age Distribution with Survival Overlay
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="age", hue="survived", multiple="stack", bins=30, palette="Set1")
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 4. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(
    df.select_dtypes(include="number").corr(),  # ✅ only numeric columns
    annot=True, cmap="coolwarm", linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.show()