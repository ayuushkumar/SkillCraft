# ------------------- Import Libraries -------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub
import os

# ------------------- Load Dataset from Kaggle Hub -------------------
print("Downloading dataset from Kaggle Hub...")
try:
    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
    print("Dataset downloaded to:", path)

    # The dataset contains multiple files; find the main CSV file
    dataset_file = ""
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv") and "accidents" in file.lower() and os.path.getsize(os.path.join(root, file)) > 1000000:
                dataset_file = os.path.join(root, file)
                break
        if dataset_file:
            break

    if not dataset_file:
        print("Error: Could not find a suitable CSV file in the downloaded dataset.")
        exit()

    print(f"Loading dataset from: {dataset_file}")
    df = pd.read_csv(dataset_file)
    print("Dataset loaded successfully!")

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)

# ------------------- Basic EDA -------------------
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ------------------- Data Preprocessing -------------------
# Drop mostly empty or unnecessary columns
df.drop(columns=['ID', 'Source', 'TMC', 'End_Time', 'End_Lat', 'End_Lng',
                 'Description', 'Street', 'Side', 'County', 'State', 'Zipcode',
                 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp'],
        inplace=True, errors='ignore')

# Convert Start_Time to datetime (handles nanoseconds)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce', infer_datetime_format=True)

# Extract day of week and hour
df['Day_of_Week'] = df['Start_Time'].dt.day_name()
df['Hour_of_Day'] = df['Start_Time'].dt.hour

# Encode categorical columns (excluding some high-cardinality ones)
categorical_cols = df.select_dtypes(include=['object']).columns
cols_to_encode = [col for col in categorical_cols if col not in ['City', 'Wind_Direction', 'Sunrise_Sunset']]
label_encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ------------------- Exploratory Visualizations -------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 12))

# 1. Accident counts by severity
plt.subplot(2, 2, 1)
if 'Severity' in df.columns:
    sns.countplot(x='Severity', data=df, palette='viridis')
    plt.title("Accident Counts by Severity")

# 2. Accidents by weather conditions (Top 10)
plt.subplot(2, 2, 2)
if 'Weather_Condition' in df.columns:
    top_weather = df['Weather_Condition'].value_counts().nlargest(10).index
    sns.countplot(y='Weather_Condition', data=df[df['Weather_Condition'].isin(top_weather)],
                  order=top_weather, palette='coolwarm')
    plt.title("Top 10 Accidents by Weather Conditions")

# 3. Accidents by day of the week
plt.subplot(2, 2, 3)
if 'Day_of_Week' in df.columns:
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='Day_of_Week', data=df, order=days_order, palette='magma')
    plt.title("Accidents by Day of the Week")
    plt.xticks(rotation=45)

# 4. Accidents by hour of day
plt.subplot(2, 2, 4)
if 'Hour_of_Day' in df.columns:
    sns.histplot(df['Hour_of_Day'].dropna(), bins=24, kde=False, color='skyblue')
    plt.title("Accidents by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

# ------------------- Correlation Heatmap -------------------
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ------------------- Predictive Modeling -------------------
if 'Severity' in df.columns:
    columns_to_drop = ['Start_Time', 'City', 'Wind_Direction', 'Sunrise_Sunset']
    X = df.drop(columns_to_drop + ['Severity'], axis=1, errors='ignore')
    y = df['Severity']

    # Fill missing numeric values
    X = X.fillna(X.mean())

    if len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()