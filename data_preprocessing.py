import numpy as np
import pandas as pd
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
from dataprep.eda.missing import plot_missing
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the Dataset
df = load_dataset("titanic")

# Step 2: Data Cleaning
# Replace placeholder for missing values with NaN
df = df.replace(" ?", np.NaN)

# Step 3: Exploratory Data Analysis (EDA)
# Generate a report to understand the dataset
report = create_report(df)
report.show_browser()

# Step 4: Drop Irrelevant Columns
# Drop columns that are not needed for modeling
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Step 5: Visualize Missing Values
# Plot missing values to understand which features need handling
plot_missing(df)

# Step 6: Drop Columns with High Percentage of Missing Values
# Drop the 'Cabin' column due to too many missing values
df = df.drop(columns=['Cabin'])

# Step 7: Handle Missing Values
# Fill missing values in 'Age' with the median
df['Age'] = df['Age'].fillna(df['Age'].median())
# Fill missing values in 'Embarked' with the most frequent value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Step 8: Outlier Detection and Removal
# Handle outliers in 'Fare' using the IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Fare'] = np.where(df['Fare'] < lower_bound, lower_bound, df['Fare'])
df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])

# Step 9: Feature Engineering
# Create new feature 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Drop 'SibSp' and 'Parch' after creating 'FamilySize'
df = df.drop(columns=['SibSp', 'Parch'])

# Create new feature 'AgeGroup'
bins = [0, 18, 35, 60, np.inf]
labels = ['child', 'young_adult', 'adult', 'elderly']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
# Drop the original 'Age' column
df = df.drop(columns=['Age'])

# Step 10: Encode Categorical Variables
# One-hot encoding for 'Sex', 'Embarked', and 'AgeGroup'
encoded_cols = ['Sex', 'Embarked', 'AgeGroup']
df = pd.get_dummies(df, columns=encoded_cols, drop_first=True)  # drop_first=True to avoid multicollinearity

# Step 11: Scale Numerical Features
# Scale 'Fare' and 'FamilySize' using Min-Max Scaler
scaler = MinMaxScaler()
df[['Fare', 'FamilySize']] = scaler.fit_transform(df[['Fare', 'FamilySize']])
