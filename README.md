Data Cleaning & Preprocessing on Titanic Dataset

Overview
This project focuses on data cleaning and preprocessing of the Titanic dataset as part of an AI & ML internship task. The preprocessing steps include handling missing values, encoding categorical variables, feature scaling, and handling outliers. The prepared dataset can then be used for building machine learning models.

Dataset
The dataset used for this project is the Titanic dataset, which contains information about the passengers aboard the Titanic, including demographic information and whether they survived the disaster. The dataset was obtained from Kaggle.

Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Steps Performed
1. Data Loading and Exploration
- Loaded the Titanic dataset using Pandas
- Displayed basic information about the dataset (shape, data types)
- Checked for missing values
- Generated summary statistics for numerical features
- Explored unique values in categorical columns
- Visualized the distribution of the target variable (Survived)
2. Handling Missing Values
- Used median imputation for missing Age values
- Created a binary feature 'Has_Cabin' to capture information from the Cabin column
- Filled missing Embarked values with the most frequent value
- Removed unnecessary columns (Cabin, Ticket, Name, PassengerId)
3. Encoding Categorical Features
- Applied label encoding to the Sex column (male: 0, female: 1)
- Used one-hot encoding for the Embarked column to avoid introducing ordinal relationships
4. Feature Scaling
- Applied standardization to numerical features (Age, Fare)
- Transformed features to have mean = 0 and standard deviation = 1
5. Handling Outliers
- Visualized outliers using boxplots
- Used the IQR (Interquartile Range) method to detect and remove outliers in Fare and SibSp columns
- Created visualizations before and after outlier removal
6. Final Dataset Preparation
- Split the data into features (X) and target (y)
- Generated a correlation matrix to understand relationships between features
- Saved the preprocessed dataset to CSV format

Insights
- The preprocessing revealed several patterns in the data, particularly the relationship between survival and features such as Sex, Class, and Age
- The correlation matrix highlighted important relationships between features that can be useful for feature selection in modeling
- Outlier removal improved the distribution of features like Fare

Conclusion
This preprocessing pipeline prepares the Titanic dataset for machine learning modeling by addressing common data quality issues. The clean dataset is now ready for further analysis and model building.

How to Run
- Ensure you have Python installed along with the required libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Download the Titanic dataset from Kaggle and place it in the project directory
- Run the preprocessing script to clean and prepare the data
- The preprocessed dataset will be saved as 'preprocessed_titanic.csv'
