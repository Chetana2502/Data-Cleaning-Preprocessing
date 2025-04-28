# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

print("Step 1: Load and Explore the Dataset")
# Load the dataset
df = pd.read_csv('titanic.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics for numerical columns
print("\nBasic statistics:")
print(df.describe())

# Check unique values in categorical columns
for col in df.select_dtypes(include=['object']).columns:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

# Visualize distribution of target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Distribution of Survival')
plt.savefig('survival_distribution.png')
plt.close()

print("\nStep 2: Handle Missing Values")
# Create a copy of the dataframe to work with
df_clean = df.copy()

# Handle missing Age values with median imputation
print(f"Missing Age values before: {df_clean['Age'].isnull().sum()}")
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
print(f"Missing Age values after: {df_clean['Age'].isnull().sum()}")

# Create binary feature for Cabin
df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)

# Handle missing Embarked values
print(f"Missing Embarked values before: {df_clean['Embarked'].isnull().sum()}")
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
print(f"Missing Embarked values after: {df_clean['Embarked'].isnull().sum()}")

# Drop columns not needed for modeling
df_clean.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Verify no missing values remain
print("\nMissing values after handling:")
print(df_clean.isnull().sum())

print("\nStep 3: Convert Categorical Features to Numerical")
# Encode Sex column
print("Encoding Sex column...")
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked column
print("One-hot encoding Embarked column...")
embarked_dummies = pd.get_dummies(df_clean['Embarked'], prefix='Embarked', drop_first=True)
df_clean = pd.concat([df_clean, embarked_dummies], axis=1)
df_clean.drop('Embarked', axis=1, inplace=True)

# Check the transformed data
print("Data after categorical encoding:")
print(df_clean.head())

print("\nStep 4: Normalize/Standardize Numerical Features")
# Identify numerical columns to standardize
num_cols = ['Age', 'Fare']

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize numerical features
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

# Check standardized data
print("Data after standardization:")
print(df_clean.head())

print("\nStep 5: Handle Outliers")
# Create boxplots to visualize outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[['Age', 'Fare', 'SibSp', 'Parch']])
plt.title('Boxplot to Detect Outliers')
plt.savefig('outliers_before.png')
plt.close()

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    print(f"Removing {outliers} outliers from {column}")
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from selected columns
print(f"Dataset shape before removing outliers: {df_clean.shape}")
df_clean = remove_outliers(df_clean, 'Fare')
df_clean = remove_outliers(df_clean, 'SibSp')
print(f"Dataset shape after removing outliers: {df_clean.shape}")

# Visualize after handling outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[['Age', 'Fare', 'SibSp', 'Parch']])
plt.title('Boxplot After Handling Outliers')
plt.savefig('outliers_after.png')
plt.close()

print("\nStep 6: Final Preprocessing and Save the Dataset")
# Split features and target
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Save the preprocessed dataset
df_clean.to_csv('preprocessed_titanic.csv', index=False)

print("\nPreprocessing completed!")
print(f"Final dataset shape: {df_clean.shape}")
print(f"Features: {X.columns.tolist()}")
print("\nThe preprocessed dataset has been saved as 'preprocessed_titanic.csv'")
